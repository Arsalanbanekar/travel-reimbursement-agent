"""
Tests for the graph's failure behaviour.

These cover the paths that are easy to leave untested because they only happen
when something goes wrong: a missing API key, a rate limit, a malformed claim.
None of them call the LLM, so they run offline.
"""

import json

import pytest

from agent.config import DATA_DIR
from agent.graph import _confidence, _resolve_decision, run_claim
from agent.rules import evaluate_claim
from models.schema import Decision


def load_claim(claim_id: str) -> dict:
    with open(DATA_DIR / "sample_claims.json", encoding="utf-8") as f:
        claims = json.load(f)["claims"]
    return next(c for c in claims if c["claim_id"] == claim_id)


@pytest.fixture
def no_llm(monkeypatch):
    """Simulate the LLM being unreachable."""

    def unavailable(*args, **kwargs):
        raise RuntimeError("GROQ_API_KEY is not set.")

    monkeypatch.setattr("agent.graph.require_api_key", unavailable)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_llm_outage_fails_safe_rather_than_auto_approving(no_llm):
    """
    Losing the model must not release money.

    The rules alone cannot spot a non-reimbursable item in a description or a
    destination that contradicts the trip purpose. If that layer did not run,
    the claim goes to a human instead of being approved.
    """

    result = run_claim(load_claim("CLM-002"))  # baseline Partially Approve
    decision = result["decision"]

    assert decision["decision"] == "Manual Review"
    assert result["llm_error"]


def test_llm_outage_still_computes_the_amounts(no_llm):
    """Failing safe must not mean losing work — the numbers still stand."""

    decision = run_claim(load_claim("CLM-002"))["decision"]

    assert decision["approved_amount"] == 12650
    assert decision["rejected_amount"] == 800


def test_llm_outage_does_not_soften_a_rule_based_rejection(no_llm):
    """A late claim is rejected by the rules; no model is needed to know that."""

    decision = run_claim(load_claim("CLM-003"))["decision"]

    assert decision["decision"] == "Reject"
    assert decision["approved_amount"] == 0


def test_degraded_run_reports_lower_confidence(no_llm):
    """A decision with no model corroboration must not claim high confidence."""

    decision = run_claim(load_claim("CLM-002"))["decision"]

    assert decision["confidence"] <= 0.6


def test_degraded_run_explains_why(no_llm):
    decision = run_claim(load_claim("CLM-001"))["decision"]

    assert "rule engine" in decision["explanation"].lower()


def test_degraded_run_still_produces_an_audit_trail(no_llm):
    result = run_claim(load_claim("CLM-001"))

    tools = [s["tool_name"] for s in result["audit_trail"]]

    assert "intake" in tools
    assert "rule_engine" in tools
    assert "guardrail" in tools


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description,claim",
    [
        (
            "no expense lines",
            {"claim_id": "X", "travel_start_date": "2025-03-01",
             "travel_end_date": "2025-03-02", "submission_date": "2025-03-03",
             "expenses": []},
        ),
        (
            "negative amount",
            {"claim_id": "X", "travel_start_date": "2025-03-01",
             "travel_end_date": "2025-03-02", "submission_date": "2025-03-03",
             "expenses": [{"category": "hotel", "amount": -50, "receipt_id": "R1"}]},
        ),
        (
            "unparseable date",
            {"claim_id": "X", "travel_start_date": "not-a-date",
             "travel_end_date": "2025-03-02", "submission_date": "2025-03-03",
             "expenses": [{"category": "hotel", "amount": 50, "receipt_id": "R1"}]},
        ),
        (
            "missing claim_id",
            {"travel_start_date": "2025-03-01", "travel_end_date": "2025-03-02",
             "submission_date": "2025-03-03",
             "expenses": [{"category": "hotel", "amount": 50, "receipt_id": "R1"}]},
        ),
    ],
)
def test_malformed_claims_are_rejected_cleanly(description, claim, no_llm):
    """Bad input must raise a validation error, never reach the decision stage."""

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        run_claim(claim)


# ---------------------------------------------------------------------------
# Guardrail arithmetic — the core safety property
# ---------------------------------------------------------------------------


def test_model_cannot_be_more_generous_than_the_rules():
    """A model proposing Approve over a Partially Approve baseline is overruled."""

    facts = evaluate_claim(load_claim("CLM-002"))

    final, note = _resolve_decision(facts, Decision.APPROVE)

    assert final == Decision.PARTIALLY_APPROVE
    assert "more generous" in note


def test_model_cannot_clear_a_blocking_issue():
    """A blocked claim stays blocked no matter what the model proposes."""

    facts = evaluate_claim(load_claim("CLM-004"))  # missing receipt

    for proposed in (Decision.APPROVE, Decision.PARTIALLY_APPROVE, Decision.REJECT):
        final, note = _resolve_decision(facts, proposed)

        assert final == Decision.MANUAL_REVIEW
        assert "human reviewer" in note


def test_model_may_escalate_to_manual_review():
    """Escalation is the one direction the model is trusted in."""

    facts = evaluate_claim(load_claim("CLM-001"))  # baseline Approve

    final, note = _resolve_decision(facts, Decision.MANUAL_REVIEW)

    assert final == Decision.MANUAL_REVIEW
    assert "escalated" in note


def test_disagreement_routes_to_a_human():
    """Model stricter than the rules, with no blocking issue, means uncertainty."""

    facts = evaluate_claim(load_claim("CLM-001"))  # baseline Approve

    final, note = _resolve_decision(facts, Decision.REJECT)

    assert final == Decision.MANUAL_REVIEW
    assert "disagree" in note


def test_agreement_passes_through_untouched():
    facts = evaluate_claim(load_claim("CLM-002"))

    final, note = _resolve_decision(facts, Decision.PARTIALLY_APPROVE)

    assert final == Decision.PARTIALLY_APPROVE
    assert note is None


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_agreement_scores_higher_than_disagreement():
    facts = evaluate_claim(load_claim("CLM-001"))

    agreed = _confidence(facts, Decision.APPROVE, Decision.APPROVE, [])
    disagreed = _confidence(facts, Decision.REJECT, Decision.MANUAL_REVIEW, [])

    assert agreed > disagreed


def test_unverifiable_citations_reduce_confidence():
    facts = evaluate_claim(load_claim("CLM-001"))

    clean = _confidence(facts, Decision.APPROVE, Decision.APPROVE, [])
    dirty = _confidence(facts, Decision.APPROVE, Decision.APPROVE, ["Section 9.9"])

    assert dirty < clean


def test_confidence_stays_in_range():
    facts = evaluate_claim(load_claim("CLM-004"))

    for proposed in (None, Decision.APPROVE, Decision.MANUAL_REVIEW):
        score = _confidence(facts, proposed, Decision.MANUAL_REVIEW, ["bad"] * 5)

        assert 0.0 <= score <= 1.0


def test_outage_on_an_approvable_claim_does_not_approve_it():
    """
    The specific bug this guards: CLM-007 contains alcohol, which only the LLM
    can notice. With the rules alone the baseline is Approve, so an outage once
    auto-approved it. It must now route to a human instead.
    """

    facts = evaluate_claim(load_claim("CLM-007"))

    assert facts.baseline_decision == Decision.APPROVE  # rules see nothing wrong

    final, note = _resolve_decision(facts, None)  # model unavailable

    assert final == Decision.MANUAL_REVIEW
    assert "not auto-approved" in note
