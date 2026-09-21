"""
Streamlit front end for the Travel Reimbursement Approval Agent.

The layout deliberately mirrors the architecture: what the rules computed,
what the agent investigated, and what the guardrail finally decided.
"""

import json

import streamlit as st

from agent.config import DATA_DIR, GROQ_MODEL
from agent.graph import run_claim

st.set_page_config(
    page_title="Travel Reimbursement Approval Agent",
    page_icon="💼",
    layout="wide",
)


WORKFLOW_DOT = """
digraph {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11
          color="#cbd5e1" fillcolor="#f1f5f9" fontcolor="#0f172a"];
    edge [color="#94a3b8" fontname="Helvetica" fontsize=9];

    intake     [label="intake\\nvalidate claim"];
    precheck   [label="precheck\\nrule engine" fillcolor="#dbeafe"];
    retrieve   [label="retrieve\\npolicy clauses" fillcolor="#dbeafe"];
    agent      [label="agent\\nLLM investigates" fillcolor="#fef3c7"];
    tools      [label="tools\\n5 lookups" fillcolor="#fef3c7"];
    decide     [label="decide\\nstructured output" fillcolor="#fef3c7"];
    guardrail  [label="guardrail\\nrules override LLM" fillcolor="#dcfce7"];

    intake -> precheck -> retrieve -> agent;
    agent -> tools [label="needs a lookup"];
    tools -> agent;
    agent -> decide [label="done"];
    decide -> guardrail;
}
"""

NODE_LABELS = {
    "intake": "Validating claim",
    "precheck": "Running rule engine",
    "retrieve": "Retrieving policy clauses",
    "agent": "Agent investigating",
    "tools": "Calling tools",
    "tools_audit": "Recording tool calls",
    "decide": "Drafting decision",
    "guardrail": "Applying guardrail",
}

DECISION_STYLE = {
    "Approve": ("🟢", st.success),
    "Partially Approve": ("🟡", st.warning),
    "Reject": ("🔴", st.error),
    "Manual Review": ("🟠", st.info),
}


@st.cache_data
def load_sample_claims() -> dict:
    with open(DATA_DIR / "sample_claims.json", encoding="utf-8") as f:
        return {c["claim_id"]: c for c in json.load(f)["claims"]}


def reset_result() -> None:
    st.session_state.result = None
    st.session_state.evaluated_claim_id = None


st.session_state.setdefault("result", None)
st.session_state.setdefault("evaluated_claim_id", None)
st.session_state.setdefault("current_claim_id", None)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Claim input")

claim_mode = st.sidebar.radio("Source", ["Sample claim", "Upload JSON"])

claim_lookup = load_sample_claims()
selected_claim = None

if claim_mode == "Sample claim":
    options = ["Select a claim"] + list(claim_lookup)
    chosen = st.sidebar.selectbox("Claim", options, index=0)

    if chosen != "Select a claim":
        selected_claim = claim_lookup[chosen]

        note = selected_claim.get("expected_notes")
        if note:
            st.sidebar.caption(note)

else:
    uploaded = st.sidebar.file_uploader("Claim JSON", type=["json"])

    if uploaded is not None:
        try:
            selected_claim = json.load(uploaded)
        except json.JSONDecodeError as exc:
            st.sidebar.error(f"Not valid JSON: {exc.msg}")

if selected_claim and st.session_state.current_claim_id != selected_claim.get("claim_id"):
    st.session_state.current_claim_id = selected_claim.get("claim_id")
    reset_result()

run_clicked = st.sidebar.button(
    "Evaluate claim",
    use_container_width=True,
    type="primary",
    disabled=selected_claim is None,
)

st.sidebar.divider()
st.sidebar.caption(f"Model: `{GROQ_MODEL}`")
st.sidebar.caption("Amounts are computed in Python. The LLM explains and judges.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("💼 Travel Reimbursement Approval Agent")
st.caption(
    "A LangGraph agent that checks travel claims against policy. "
    "Deterministic rules compute every amount; the LLM investigates ambiguity; "
    "a guardrail has the final say."
)

with st.expander("How it works", expanded=st.session_state.result is None):
    st.graphviz_chart(WORKFLOW_DOT, use_container_width=True)

    left, middle, right = st.columns(3)
    with left:
        st.markdown(
            "**Rules first**  \nLimits, receipts, duplicates and the deadline "
            "are checked in plain Python before the model sees anything."
        )
    with middle:
        st.markdown(
            "**Then the agent**  \nThe LLM chooses which of five tools to call, "
            "reads the results, and can look further."
        )
    with right:
        st.markdown(
            "**Guardrail last**  \nThe model may escalate to Manual Review, but "
            "can never approve more than the rules allow."
        )


# ---------------------------------------------------------------------------
# Claim details
# ---------------------------------------------------------------------------

if selected_claim is not None:
    st.subheader("Claim")

    top = st.columns(4)
    top[0].metric("Claim ID", selected_claim.get("claim_id", "—"))
    top[1].metric("Employee", selected_claim.get("employee_name", "—"))
    top[2].metric("Department", selected_claim.get("department", "—"))
    top[3].metric("Total claimed", f"₹{selected_claim.get('total_amount', 0):,.0f}")

    meta = st.columns(4)
    meta[0].metric("Travel start", selected_claim.get("travel_start_date", "—"))
    meta[1].metric("Travel end", selected_claim.get("travel_end_date", "—"))
    meta[2].metric("Submitted", selected_claim.get("submission_date", "—"))
    meta[3].metric("Trip type", selected_claim.get("travel_category", "—").title())

    if selected_claim.get("trip_purpose"):
        st.caption(f"Purpose: {selected_claim['trip_purpose']}")

    st.dataframe(
        selected_claim.get("expenses", []),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if run_clicked and selected_claim is not None:
    with st.status("Evaluating claim...", expanded=True) as status:
        seen: set[str] = set()

        def report(node_name: str) -> None:
            if node_name not in seen:
                seen.add(node_name)
                status.write(f"✓ {NODE_LABELS.get(node_name, node_name)}")

        try:
            st.session_state.result = run_claim(selected_claim, on_node=report)
            st.session_state.evaluated_claim_id = selected_claim.get("claim_id")
            status.update(label="Evaluation complete", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Evaluation failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

result = st.session_state.result

if result is not None:
    decision = result["decision"]
    facts = result["facts"]

    st.divider()

    icon, render = DECISION_STYLE.get(decision["decision"], ("⚪", st.info))
    render(f"{icon}  **{decision['decision']}**")

    if result.get("llm_error"):
        st.warning(
            "The language model was unavailable, so this decision came from the "
            "rule engine alone.",
            icon="⚠️",
        )

    summary = st.columns(4)
    summary[0].metric("Approved", f"₹{decision['approved_amount']:,.0f}")
    summary[1].metric("Rejected", f"₹{decision['rejected_amount']:,.0f}")
    summary[2].metric("Confidence", f"{decision['confidence'] * 100:.0f}%")
    summary[3].metric("Approver", decision.get("required_approver") or "—")

    if decision["reason_codes"]:
        st.write(" ".join(f"`{code}`" for code in decision["reason_codes"]))

    decision_tab, rules_tab, trail_tab, policy_tab, json_tab = st.tabs(
        ["Decision", "Rule engine", "Agent trail", "Policy", "JSON"]
    )

    # -- Decision ----------------------------------------------------------

    with decision_tab:
        st.markdown("#### Explanation")
        st.write(decision["explanation"])

        left, right = st.columns(2)

        with left:
            st.markdown("#### Missing documents")
            if decision["missing_documents"]:
                for document in decision["missing_documents"]:
                    st.write(f"- {document}")
            else:
                st.caption("None.")

        with right:
            st.markdown("#### Policy references")
            if decision["policy_references"]:
                for reference in decision["policy_references"]:
                    st.write(f"- {reference}")
            else:
                st.caption("None cited.")

    # -- Rule engine -------------------------------------------------------

    with rules_tab:
        st.caption(
            "Computed in Python before the model ran. These numbers are not "
            "the model's to change."
        )

        cols = st.columns(4)
        cols[0].metric("Baseline decision", facts["baseline_decision"])
        cols[1].metric("Policy allows", f"₹{facts['total_allowed']:,.0f}")
        cols[2].metric("Nights", facts["nights"])
        cols[3].metric(
            "Days to submit",
            f"{facts['days_to_submit']} / {facts['submission_window_days']}",
        )

        st.markdown("#### Expense lines")
        st.dataframe(
            [
                {
                    "Category": finding["category"],
                    "Receipt": finding["receipt_id"],
                    "Claimed": finding["claimed_amount"],
                    "Allowed": finding["allowed_amount"],
                    "Excess": finding["excess_amount"],
                    "Limit applied": finding["limit_basis"] or "—",
                    "Flags": ", ".join(finding["reason_codes"]) or "—",
                }
                for finding in facts["expense_findings"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        notes = [
            note
            for finding in facts["expense_findings"]
            for note in finding["notes"]
        ]
        if notes:
            st.markdown("#### Notes")
            for note in notes:
                st.write(f"- {note}")

    # -- Agent trail -------------------------------------------------------

    with trail_tab:
        st.caption(
            "Every step taken, in order, including which tools the model chose "
            "to call."
        )

        for step in result["audit_trail"]:
            marker = {"ok": "✓", "override": "⚠", "failed": "✗"}.get(
                step["status"], "•"
            )
            with st.expander(
                f"{marker} {step['step']}. {step['tool_name']}",
                expanded=step["status"] != "ok",
            ):
                st.write(step["details"])

    # -- Policy ------------------------------------------------------------

    with policy_tab:
        st.caption(
            f"{len(result['retrieved_sections'])} clause(s) retrieved. The agent "
            "may only cite sections from this set."
        )
        st.write(" ".join(f"`{s}`" for s in result["retrieved_sections"]))
        st.markdown("---")
        st.markdown(result["policy_context"])

    # -- JSON --------------------------------------------------------------

    with json_tab:
        st.json(result)

    st.download_button(
        "⬇ Download decision JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name=f"{st.session_state.evaluated_claim_id}_decision.json",
        mime="application/json",
    )

elif selected_claim is None:
    st.info("Pick a sample claim or upload one to begin.", icon="👈")
