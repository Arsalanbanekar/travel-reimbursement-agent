"""
The LangGraph reimbursement workflow.

    intake -> precheck -> retrieve -> agent <-> tools -> decide -> guardrail -> END

Shape of the design:

- precheck always runs. Safety checks cannot be skipped by a model deciding
  they are unnecessary.
- agent <-> tools is the agentic part: the model chooses what to investigate,
  reads the result, and decides whether to look further.
- guardrail runs last and has the final say. The model can escalate a claim
  toward Manual Review but can never approve more money than the rules allow.

The whole graph degrades gracefully: if the LLM is unavailable, the rule
engine's own decision is returned instead of failing.
"""

import operator
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.config import (
    GROQ_FALLBACK_MODEL,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    MAX_AGENT_STEPS,
    PRIMARY_COOLDOWN_SECONDS,
    require_api_key,
)
from agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    DECISION_SYSTEM_PROMPT,
    build_claim_brief,
    format_facts,
)
from agent.rules import evaluate_claim, explain_from_rules, is_rules_conclusive
from agent.tools import AGENT_TOOLS
from models.schema import (
    AuditTrailStep,
    ClaimFacts,
    ClaimInput,
    Decision,
    LLMDecision,
    ReimbursementDecision,
)
from rag.retriever import (
    format_sections,
    get_retriever,
    merge_references,
    verify_policy_references,
)


class AgentState(TypedDict, total=False):
    """Everything that flows through the graph."""

    claim: dict
    facts: ClaimFacts
    policy_context: str
    allowed_sections: list[str]
    messages: Annotated[list, add_messages]
    audit: Annotated[list[AuditTrailStep], operator.add]
    llm_decision: LLMDecision | None
    decision: dict
    llm_error: str | None
    llm_skipped: str | None


def _step(state: AgentState, tool_name: str, status: str, details: str) -> AuditTrailStep:
    return AuditTrailStep(
        step=len(state.get("audit", [])) + 1,
        tool_name=tool_name,
        status=status,
        details=details,
    )


# When the primary model returns a rate limit, stop re-testing it on every
# call. The SDK retries a 429 internally with backoff before raising, so each
# pointless attempt costs tens of seconds — and a claim makes several calls.
_primary_unavailable_until = 0.0


def _build_llm(model: str, max_retries: int | None = None):
    from langchain_groq import ChatGroq

    kwargs = {
        "model": model,
        "api_key": require_api_key(),
        "temperature": LLM_TEMPERATURE,
    }

    if max_retries is not None:
        kwargs["max_retries"] = max_retries

    return ChatGroq(**kwargs)


def _invoke_with_fallback(build, *args, **kwargs):
    """
    Try the primary model, then the lighter one.

    The free tier rate-limits the larger model, and answering with the smaller
    one beats losing the run. Once the primary has failed, it is skipped for a
    cooldown rather than retried on every subsequent call.
    """

    global _primary_unavailable_until

    now = time.monotonic()

    if now >= _primary_unavailable_until:
        try:
            # No SDK-level retries here: a 429 should surface at once so the
            # fallback can answer, instead of backing off for ~40 seconds.
            return build(GROQ_MODEL, 0)(*args, **kwargs), GROQ_MODEL
        except Exception:
            _primary_unavailable_until = now + PRIMARY_COOLDOWN_SECONDS

    return build(GROQ_FALLBACK_MODEL, None)(*args, **kwargs), GROQ_FALLBACK_MODEL


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def intake_node(state: AgentState) -> dict:
    """Validate the incoming claim against the schema."""

    claim = ClaimInput.model_validate(state["claim"])

    return {
        "audit": [
            _step(
                state,
                "intake",
                "ok",
                f"Validated claim {claim.claim_id} with "
                f"{len(claim.expenses)} expense line(s).",
            )
        ]
    }


def precheck_node(state: AgentState) -> dict:
    """Run every deterministic policy check. Always runs."""

    facts = evaluate_claim(state["claim"])

    codes = ", ".join(c.value for c in facts.reason_codes) or "none"

    return {
        "facts": facts,
        "audit": [
            _step(
                state,
                "rule_engine",
                "ok",
                f"Baseline {facts.baseline_decision.value}; "
                f"allowed ₹{facts.total_allowed:,.0f} of ₹{facts.total_claimed:,.0f}; "
                f"reason codes: {codes}.",
            )
        ],
    }


def retrieve_node(state: AgentState) -> dict:
    """Fetch the policy clauses the rules applied, plus semantic context."""

    facts: ClaimFacts = state["facts"]

    queries = [
        f"{finding.category} reimbursement limit"
        for finding in facts.expense_findings
    ]
    queries += [
        finding.description
        for finding in facts.expense_findings
        if finding.description
    ]
    queries.append("non-reimbursable expenses")

    documents = get_retriever().retrieve_for_claim(
        rule_references=facts.policy_references,
        queries=queries,
    )

    sections = [doc.metadata["section"] for doc in documents]

    return {
        "policy_context": format_sections(documents),
        "allowed_sections": sections,
        "audit": [
            _step(
                state,
                "policy_retrieval",
                "ok",
                f"Retrieved {len(documents)} clause(s): {', '.join(sections)}.",
            )
        ],
    }


def should_investigate_at_all(state: AgentState) -> str:
    """
    Decide whether the agent loop is worth running.

    On a late submission or a duplicate receipt the guardrail has already
    fixed the outcome, so several sequential model calls would cost time and
    tokens to reach a conclusion the rules reached instantly. Knowing when not
    to call the model is part of the design, not a shortcut around it.
    """

    conclusive, _ = is_rules_conclusive(state["facts"])

    return "settled" if conclusive else "investigate"


def record_skip_node(state: AgentState) -> dict:
    """Note in the audit trail why the investigation was skipped."""

    _, reason = is_rules_conclusive(state["facts"])

    return {
        "llm_skipped": reason,
        "audit": [
            _step(state, "agent", "skipped", reason or "Outcome already settled by the rules."),
        ],
    }


def agent_node(state: AgentState) -> dict:
    """The model investigates, optionally calling tools."""

    messages = state.get("messages") or []

    if not messages:
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=build_claim_brief(
                    state["claim"], state["facts"], state["policy_context"]
                )
            ),
        ]

    try:
        reply, model_used = _invoke_with_fallback(
            lambda model, retries: _build_llm(model, retries).bind_tools(AGENT_TOOLS).invoke,
            messages,
        )
    except Exception as exc:
        # No key, no network, or both models refused — fall through to the
        # rule-engine-only path in the guardrail.
        return {
            "llm_error": f"{type(exc).__name__}: {exc}",
            "audit": [
                _step(state, "agent", "failed", f"LLM unavailable: {type(exc).__name__}.")
            ],
        }

    new_messages = messages if not state.get("messages") else []
    new_messages = [*new_messages, reply]

    if reply.tool_calls:
        details = "Requested: " + ", ".join(c["name"] for c in reply.tool_calls)
    else:
        details = (reply.content or "").strip()[:300] or "No further investigation needed."

    return {
        "messages": new_messages,
        "audit": [_step(state, f"agent ({model_used})", "ok", details)],
    }


def should_investigate(state: AgentState) -> str:
    """Route to the tools node while the model keeps asking for tools."""

    if state.get("llm_error"):
        return "decide"

    messages = state.get("messages") or []
    last = messages[-1] if messages else None

    if not isinstance(last, AIMessage) or not last.tool_calls:
        return "decide"

    # Hard ceiling so a confused model cannot loop against the free tier.
    tool_turns = sum(
        1 for m in messages if isinstance(m, AIMessage) and m.tool_calls
    )

    return "tools" if tool_turns <= MAX_AGENT_STEPS else "decide"


def tools_audit_node(state: AgentState) -> dict:
    """Record which tools ran, for the audit trail."""

    messages = state.get("messages") or []
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls),
        None,
    )

    if last_ai is None:
        return {}

    return {
        "audit": [
            _step(
                state,
                call["name"],
                "ok",
                f"args={call['args']}",
            )
            for call in last_ai.tool_calls
        ]
    }


def decide_node(state: AgentState) -> dict:
    """Ask the model for a decision label, explanation and citations."""

    if state.get("llm_error"):
        return {"llm_decision": None}

    facts: ClaimFacts = state["facts"]

    investigation = "\n".join(
        (m.content or "").strip()
        for m in state.get("messages", [])
        if isinstance(m, AIMessage) and m.content
    )

    prompt = f"""
VERIFIED FACTS
{format_facts(facts)}

RETRIEVED POLICY CLAUSES
{state['policy_context']}

INVESTIGATION NOTES
{investigation or 'No additional findings.'}

Issue the final decision.
""".strip()

    messages = [
        SystemMessage(content=DECISION_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        decision, model_used = _invoke_with_fallback(
            lambda model, retries: _build_llm(model, retries)
            .with_structured_output(LLMDecision)
            .invoke,
            messages,
        )
    except Exception as exc:
        return {
            "llm_decision": None,
            "llm_error": f"{type(exc).__name__}: {exc}",
            "audit": [
                _step(
                    state,
                    "decide",
                    "failed",
                    f"Structured decision failed: {type(exc).__name__}.",
                )
            ],
        }

    return {
        "llm_decision": decision,
        "audit": [
            _step(
                state,
                f"decide ({model_used})",
                "ok",
                f"Model proposed {decision.decision.value}.",
            )
        ],
    }


def _resolve_decision(
    facts: ClaimFacts,
    proposed: Decision | None,
) -> tuple[Decision, str | None]:
    """
    Reconcile the model's decision with the rules.

    The model may escalate toward review. It may never release more money than
    the rules allow, and it may never clear a blocking issue.
    """

    baseline = facts.baseline_decision

    if proposed is None:
        # The judgement layer did not run. Rules alone cannot see a
        # non-reimbursable item in a description or a destination that
        # contradicts the trip purpose, so releasing money here would be
        # failing open. Rejections and reviews are rule-determined and stand.
        if baseline in (Decision.APPROVE, Decision.PARTIALLY_APPROVE):
            return Decision.MANUAL_REVIEW, (
                "The language model was unavailable, so the checks only it can "
                "perform did not run. The claim is not auto-approved; the "
                "amounts below are what policy would allow, pending review."
            )

        return baseline, None

    if proposed == baseline:
        return baseline, None

    if baseline == Decision.MANUAL_REVIEW:
        return baseline, (
            f"Model proposed {proposed.value}, but unresolved blocking issues "
            "require a human reviewer."
        )

    if proposed == Decision.MANUAL_REVIEW:
        # Escalation is always allowed — the model saw something the rules did not.
        return proposed, "Model escalated to Manual Review after investigation."

    generosity = {
        Decision.REJECT: 0,
        Decision.MANUAL_REVIEW: 1,
        Decision.PARTIALLY_APPROVE: 2,
        Decision.APPROVE: 3,
    }

    if generosity[proposed] > generosity[baseline]:
        return baseline, (
            f"Model proposed {proposed.value}, which is more generous than the "
            f"rules allow ({baseline.value}). Rules applied."
        )

    # Model is stricter than the rules but gave no blocking issue: that is a
    # disagreement, and disagreement means a human should look.
    return Decision.MANUAL_REVIEW, (
        f"Model proposed {proposed.value} against a rules baseline of "
        f"{baseline.value}. Routed to Manual Review because they disagree."
    )


def _confidence(
    facts: ClaimFacts,
    proposed: Decision | None,
    final: Decision,
    unsupported: list[str],
) -> float:
    """
    Derive a confidence score from real signals.

    Deliberately not asked of the model: a self-reported confidence carries no
    information about whether the answer is actually right.
    """

    if proposed is None:
        score = 0.55  # rules only, no model corroboration
    elif proposed == facts.baseline_decision:
        score = 0.95  # model and rules independently agree
    else:
        score = 0.55  # they disagreed and the guardrail intervened

    if unsupported:
        score -= 0.15

    if final == Decision.MANUAL_REVIEW:
        score = min(score, 0.60)

    score -= 0.05 * len(facts.blocking_issues)

    return round(max(score, 0.10), 2)


def guardrail_node(state: AgentState) -> dict:
    """
    Assemble the final answer and enforce the rules over the model.

    Amounts, reason codes and the approver come from the rule engine. Citations
    are filtered to sections that exist and were actually retrieved.
    """

    facts: ClaimFacts = state["facts"]
    llm_decision: LLMDecision | None = state.get("llm_decision")
    skip_reason = state.get("llm_skipped")

    proposed = llm_decision.decision if llm_decision else None

    if skip_reason:
        # The investigation was skipped because it could not change anything,
        # which is different from the model having failed. The rules decided,
        # and they are deterministic, so this is a confident answer.
        final_decision, override_note = facts.baseline_decision, None
    else:
        final_decision, override_note = _resolve_decision(facts, proposed)

    supported, unsupported = verify_policy_references(
        llm_decision.policy_references if llm_decision else [],
        allowed_sections=set(state.get("allowed_sections", [])),
    )

    # The rule engine's own citations are trustworthy by construction; the
    # model's are added only after passing verification.
    references = merge_references(facts.policy_references, supported)

    if llm_decision:
        explanation = llm_decision.explanation
    elif skip_reason:
        explanation = explain_from_rules(facts)
    else:
        reason = state.get("llm_error", "the language model was unavailable")
        explanation = (
            f"Decided by the rule engine alone because {reason}. "
            f"{facts.baseline_decision.value}: "
            f"₹{facts.total_allowed:,.0f} of ₹{facts.total_claimed:,.0f} is payable under policy."
        )

    if override_note:
        explanation = f"{explanation}\n\n[Guardrail] {override_note}"

    if unsupported:
        explanation = (
            f"{explanation}\n\n[Guardrail] Removed unverifiable citation(s): "
            f"{', '.join(unsupported)}."
        )

    payable = final_decision != Decision.REJECT

    decision = ReimbursementDecision(
        decision=final_decision,
        approved_amount=facts.total_allowed if payable else 0.0,
        rejected_amount=facts.total_rejected if payable else facts.total_claimed,
        missing_documents=facts.missing_documents,
        policy_references=references,
        reason_codes=facts.reason_codes,
        confidence=(
            # Fully rule-determined: no model opinion to agree or disagree with.
            0.95
            if skip_reason
            else _confidence(facts, proposed, final_decision, unsupported)
        ),
        explanation=explanation,
        required_approver=facts.required_approver,
    )

    audit = [
        _step(
            state,
            "guardrail",
            "override" if override_note else "ok",
            override_note or f"Model agreed with the rules ({final_decision.value}).",
        )
    ]

    decision.audit_trail = [*state.get("audit", []), *audit]

    return {"decision": decision.model_dump(mode="json"), "audit": audit}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    """Compile the workflow."""

    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("precheck", precheck_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(AGENT_TOOLS))
    graph.add_node("tools_audit", tools_audit_node)
    graph.add_node("decide", decide_node)
    graph.add_node("record_skip", record_skip_node)
    graph.add_node("guardrail", guardrail_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "precheck")
    graph.add_edge("precheck", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        should_investigate_at_all,
        {"investigate": "agent", "settled": "record_skip"},
    )

    graph.add_conditional_edges(
        "agent",
        should_investigate,
        {"tools": "tools_audit", "decide": "decide"},
    )

    graph.add_edge("tools_audit", "tools")
    graph.add_edge("tools", "agent")
    graph.add_edge("record_skip", "guardrail")
    graph.add_edge("decide", "guardrail")
    graph.add_edge("guardrail", END)

    return graph.compile()


_compiled = None


def get_graph():
    """Compile once and reuse."""

    global _compiled

    if _compiled is None:
        _compiled = build_graph()

    return _compiled


def run_claim(claim: dict, on_node=None) -> dict:
    """
    Evaluate one claim end to end.

    Returns the structured decision, the rule-engine facts, the retrieved
    policy context and the audit trail.

    Pass on_node to receive each node's name as it finishes, which the UI uses
    to show progress during a run that takes tens of seconds.
    """

    inputs = {"claim": claim}

    if on_node is None:
        final_state: dict[str, Any] = get_graph().invoke(inputs)
    else:
        final_state = {}

        for mode, chunk in get_graph().stream(
            inputs, stream_mode=["updates", "values"]
        ):
            if mode == "updates":
                for node_name in chunk:
                    on_node(node_name)
            else:
                final_state = chunk

    facts: ClaimFacts = final_state["facts"]

    return {
        "decision": final_state["decision"],
        "facts": facts.model_dump(mode="json"),
        "policy_context": final_state.get("policy_context", ""),
        "retrieved_sections": final_state.get("allowed_sections", []),
        "audit_trail": [
            step.model_dump() if hasattr(step, "model_dump") else step
            for step in final_state.get("audit", [])
        ],
        "llm_error": final_state.get("llm_error"),
    }


if __name__ == "__main__":

    import json

    from agent.rules import _load

    for raw in _load("sample_claims.json")["claims"]:
        result = run_claim(raw)
        decision = result["decision"]

        print("\n" + "=" * 78)
        print(f"{raw['claim_id']}  expected={raw['expected_decision']}")
        print("=" * 78)
        print(json.dumps(decision, indent=2, ensure_ascii=False))
