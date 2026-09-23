"""
Tests for the deterministic rule engine.

These run without an API key, because the rule engine never calls an LLM.
Each test below corresponds to a rule in data/travel_policy.md.
"""

import json

import pytest

from agent.config import DATA_DIR
from agent.rules import evaluate_claim
from models.schema import Decision, ReasonCode


def load_sample_claims() -> list[dict]:
    with open(DATA_DIR / "sample_claims.json", encoding="utf-8") as f:
        return json.load(f)["claims"]


SAMPLE_CLAIMS = load_sample_claims()


def evaluate_isolated(claim: dict):
    """
    Evaluate a synthetic claim with no prior claim history.

    Synthetic claims reuse receipt IDs from the sample data, which would
    otherwise trip the duplicate check and mask what the test is asserting.
    """

    return evaluate_claim(claim, claim_history=[])


def make_claim(**overrides) -> dict:
    """A minimal, fully compliant claim that individual tests can bend."""

    claim = {
        "claim_id": "TEST-001",
        "employee_id": "EMP-0001",
        "employee_name": "Test Employee",
        "department": "Engineering",
        "travel_category": "domestic",
        "travel_start_date": "2025-03-14",
        "travel_end_date": "2025-03-15",
        "submission_date": "2025-03-18",
        "manager_pre_approved": True,
        "currency": "INR",
        "expenses": [
            {
                "category": "hotel",
                "description": "1 night",
                "amount": 4200,
                "currency": "INR",
                "receipt_id": "RCP-001",
            }
        ],
    }
    claim.update(overrides)
    return claim


# ---------------------------------------------------------------------------
# Golden set — the five documented sample claims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("claim", SAMPLE_CLAIMS, ids=[c["claim_id"] for c in SAMPLE_CLAIMS])
def test_sample_claims_match_expected_baseline(claim):
    """
    The rule engine alone reproduces the documented baseline for every claim.

    For most claims the baseline is also the final decision. CLM-006 and
    CLM-007 pass every numeric rule, so the baseline is Approve and only the
    LLM can escalate them — that gap is the point of those two fixtures.
    """

    facts = evaluate_claim(claim)

    assert facts.baseline_decision.value == claim["expected_baseline_decision"]


def test_judgement_cases_are_invisible_to_the_rules():
    """
    Guards the fixtures that prove the LLM adds something.

    If a future rule change made these decidable by rules alone, they would
    stop testing what they were written to test.
    """

    judgement_cases = [
        c
        for c in SAMPLE_CLAIMS
        if c["expected_baseline_decision"] != c["expected_decision"]
    ]

    assert {c["claim_id"] for c in judgement_cases} == {"CLM-006", "CLM-007"}

    for claim in judgement_cases:
        facts = evaluate_claim(claim)

        assert facts.baseline_decision == Decision.APPROVE
        assert facts.reason_codes == []


@pytest.mark.parametrize(
    "claim",
    [c for c in SAMPLE_CLAIMS if c.get("expected_approved_amount") is not None],
    ids=[
        c["claim_id"]
        for c in SAMPLE_CLAIMS
        if c.get("expected_approved_amount") is not None
    ],
)
def test_sample_claim_amounts(claim):
    """Approved and rejected amounts match the documented expectations."""

    facts = evaluate_claim(claim)

    assert facts.total_allowed == claim["expected_approved_amount"]
    assert facts.total_rejected == claim["expected_rejected_amount"]


def test_amounts_always_reconcile():
    """Allowed + rejected must equal the total claimed, for every sample."""

    for claim in SAMPLE_CLAIMS:
        facts = evaluate_claim(claim)

        assert facts.total_allowed + facts.total_rejected == pytest.approx(
            facts.total_claimed
        ), f"{facts.claim_id} does not reconcile"


# ---------------------------------------------------------------------------
# Limits (Sections 4-9)
# ---------------------------------------------------------------------------


def test_hotel_limit_is_per_night_not_per_line():
    """A 3-night stay gets 3x the nightly cap, not 1x (Section 4.1)."""

    claim = make_claim(
        travel_start_date="2025-03-10",
        travel_end_date="2025-03-13",  # 3 nights
        expenses=[
            {
                "category": "hotel",
                "amount": 14000,
                "currency": "INR",
                "receipt_id": "RCP-UNKNOWN-1",
            }
        ],
    )

    facts = evaluate_isolated(claim)

    assert facts.nights == 3
    # 3 nights x 5,000 = 15,000 allowance, so 14,000 is fully covered.
    assert facts.total_allowed == 14000
    assert ReasonCode.LIMIT_EXCEEDED not in facts.reason_codes


def test_hotel_over_the_nightly_allowance_is_partially_approved():
    claim = make_claim(
        expenses=[
            {
                "category": "hotel",
                "amount": 5800,
                "currency": "INR",
                "receipt_id": "RCP-005",
            }
        ]
    )

    facts = evaluate_isolated(claim)

    assert facts.baseline_decision == Decision.PARTIALLY_APPROVE
    assert facts.total_allowed == 5000
    assert facts.total_rejected == 800


def test_meals_daily_cap_aggregates_across_lines():
    """Two meals on the same day share one daily cap (Section 6.1)."""

    claim = make_claim(
        expenses=[
            {
                "category": "meals",
                "amount": 700,
                "currency": "INR",
                "receipt_id": "RCP-MEAL-A",
                "date": "2025-03-14",
            },
            {
                "category": "meals",
                "amount": 600,
                "currency": "INR",
                "receipt_id": "RCP-MEAL-B",
                "date": "2025-03-14",
            },
        ]
    )

    facts = evaluate_isolated(claim)

    # 1,300 claimed against a 1,000/day cap.
    assert facts.total_allowed == 1000
    assert facts.total_rejected == 300
    assert ReasonCode.LIMIT_EXCEEDED in facts.reason_codes


def test_meals_on_separate_days_each_get_their_own_cap():
    claim = make_claim(
        expenses=[
            {
                "category": "meals",
                "amount": 900,
                "currency": "INR",
                "receipt_id": "RCP-MEAL-A",
                "date": "2025-03-14",
            },
            {
                "category": "meals",
                "amount": 900,
                "currency": "INR",
                "receipt_id": "RCP-MEAL-B",
                "date": "2025-03-15",
            },
        ]
    )

    facts = evaluate_isolated(claim)

    assert facts.total_allowed == 1800
    assert facts.total_rejected == 0


def test_taxi_trip_ceiling_and_daily_cap_both_apply():
    """
    A 3,000 taxi ride is first cut to the 2,500 per-trip ceiling
    (Section 7.2), then to the 2,000 daily cap (Section 7.1), which is the
    tighter of the two. The per-trip note is still recorded for the auditor.
    """

    claim = make_claim(
        expenses=[
            {
                "category": "taxi",
                "amount": 3000,
                "currency": "INR",
                "receipt_id": "RCP-TAXI-A",
                "date": "2025-03-14",
            }
        ]
    )

    facts = evaluate_isolated(claim)

    assert facts.total_allowed == 2000
    assert facts.total_rejected == 1000
    assert any(
        "per-trip ceiling" in note
        for finding in facts.expense_findings
        for note in finding.notes
    )


def test_international_travel_uses_international_limits():
    """The same hotel amount is fine internationally (Section 4.2)."""

    claim = make_claim(
        travel_category="international",
        expenses=[
            {
                "category": "hotel",
                "amount": 11000,
                "currency": "INR",
                "receipt_id": "RCP-INTL-1",
            }
        ],
    )

    facts = evaluate_isolated(claim)

    assert facts.total_allowed == 11000
    assert facts.total_rejected == 0


# ---------------------------------------------------------------------------
# Receipts (Section 10) and Manual Review (Section 14)
# ---------------------------------------------------------------------------


def test_missing_attachment_above_threshold_goes_to_manual_review():
    """RCP-008 exists but has no attachment, and 680 > 500 (Section 14.1)."""

    claim = make_claim(
        expenses=[
            {
                "category": "meals",
                "amount": 680,
                "currency": "INR",
                "receipt_id": "RCP-008",
            }
        ]
    )

    facts = evaluate_isolated(claim)

    assert facts.baseline_decision == Decision.MANUAL_REVIEW
    assert ReasonCode.MISSING_RECEIPT in facts.reason_codes
    assert facts.missing_documents


def test_missing_attachment_below_threshold_is_tolerated():
    """Under 500 a receipt is optional, so this should not block (Section 10.4)."""

    claim = make_claim(
        expenses=[
            {
                "category": "taxi",
                "amount": 300,
                "currency": "INR",
                "receipt_id": "RCP-NONE-AT-ALL",
            }
        ]
    )

    facts = evaluate_isolated(claim)

    assert facts.baseline_decision == Decision.APPROVE
    assert ReasonCode.MISSING_RECEIPT not in facts.reason_codes


def test_receipt_amount_mismatch_is_flagged():
    """Claiming 9,000 against a 4,200 receipt must not pass silently."""

    claim = make_claim(
        expenses=[
            {
                "category": "hotel",
                "amount": 9000,
                "currency": "INR",
                "receipt_id": "RCP-001",  # registry says 4,200
            }
        ]
    )

    facts = evaluate_isolated(claim)

    assert ReasonCode.RECEIPT_MISMATCH in facts.reason_codes
    assert facts.baseline_decision == Decision.MANUAL_REVIEW


def test_unknown_category_is_not_silently_approved():
    """The old tool approved unknown categories in full; it must not."""

    claim = make_claim(
        expenses=[
            {
                "category": "spa",
                "amount": 3000,
                "currency": "INR",
                "receipt_id": "RCP-SPA-1",
            }
        ]
    )

    facts = evaluate_isolated(claim)

    assert facts.baseline_decision == Decision.MANUAL_REVIEW
    assert ReasonCode.UNKNOWN_CATEGORY in facts.reason_codes
    assert facts.total_allowed == 0


def test_currency_mismatch_routes_to_manual_review():
    claim = make_claim(
        expenses=[
            {
                "category": "hotel",
                "amount": 200,
                "currency": "USD",
                "receipt_id": "RCP-USD-1",
            }
        ]
    )

    facts = evaluate_isolated(claim)

    assert ReasonCode.CURRENCY_MISMATCH in facts.reason_codes
    assert facts.baseline_decision == Decision.MANUAL_REVIEW


# ---------------------------------------------------------------------------
# Duplicates (Section 12)
# ---------------------------------------------------------------------------


def test_duplicate_flags_only_the_later_claim():
    """
    The original claim must stay clean.

    CLM-001 (submitted 18 Mar) and CLM-005 (submitted 25 Mar) share RCP-003.
    Only CLM-005 should be flagged.
    """

    original = evaluate_claim(next(c for c in SAMPLE_CLAIMS if c["claim_id"] == "CLM-001"))
    later = evaluate_claim(next(c for c in SAMPLE_CLAIMS if c["claim_id"] == "CLM-005"))

    assert ReasonCode.DUPLICATE_RECEIPT not in original.reason_codes
    assert ReasonCode.DUPLICATE_RECEIPT in later.reason_codes


def test_no_duplicate_when_history_is_empty():
    """A receipt is only a duplicate relative to prior claims."""

    claim = next(c for c in SAMPLE_CLAIMS if c["claim_id"] == "CLM-005")

    facts = evaluate_claim(claim, claim_history=[])

    assert ReasonCode.DUPLICATE_RECEIPT not in facts.reason_codes


# ---------------------------------------------------------------------------
# Submission window (Section 11)
# ---------------------------------------------------------------------------


def test_late_submission_is_rejected_and_reimburses_nothing():
    claim = make_claim(
        travel_end_date="2025-03-08",
        submission_date="2025-04-22",  # 45 days later
    )

    facts = evaluate_isolated(claim)

    assert facts.baseline_decision == Decision.REJECT
    assert facts.days_to_submit == 45
    assert facts.total_allowed == 0
    assert facts.total_rejected == facts.total_claimed


def test_submission_exactly_on_the_deadline_is_accepted():
    """30 days is within the window; 31 is not (Section 11.1)."""

    on_time = evaluate_isolated(
        make_claim(travel_end_date="2025-03-01", submission_date="2025-03-31")
    )
    too_late = evaluate_isolated(
        make_claim(travel_end_date="2025-03-01", submission_date="2025-04-01")
    )

    assert on_time.within_submission_window is True
    assert too_late.within_submission_window is False


def test_manual_review_takes_precedence_over_late_rejection():
    """
    Section 14 says route to a human 'rather than automatic rejection',
    so a late claim that also has a missing receipt is not auto-rejected.
    """

    claim = make_claim(
        travel_end_date="2025-03-08",
        submission_date="2025-04-22",
        expenses=[
            {
                "category": "meals",
                "amount": 680,
                "currency": "INR",
                "receipt_id": "RCP-008",  # no attachment
            }
        ],
    )

    facts = evaluate_isolated(claim)

    assert facts.baseline_decision == Decision.MANUAL_REVIEW


# ---------------------------------------------------------------------------
# Approval matrix (Section 2.2 / 14.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total,expected_approver",
    [
        (18280, "Manager"),
        (25000, "Manager"),
        (60000, "Senior Manager"),
        (150000, "Finance Head"),
    ],
)
def test_approval_bands(total, expected_approver):
    claim = make_claim(
        expenses=[
            {
                "category": "flight",
                "amount": total,
                "currency": "INR",
                "receipt_id": "RCP-FLIGHT-X",
            }
        ]
    )

    facts = evaluate_isolated(claim)

    assert facts.required_approver == expected_approver


def test_amount_in_the_gap_between_bands_still_resolves():
    """
    The matrix jumps from 25,000 to 25,001, so 25,000.50 sits in a gap.
    The old lookup returned 'Unknown'; it must now land in a real band.
    """

    claim = make_claim(
        expenses=[
            {
                "category": "flight",
                "amount": 25000.50,
                "currency": "INR",
                "receipt_id": "RCP-FLIGHT-GAP",
            }
        ]
    )

    facts = evaluate_isolated(claim)

    assert facts.required_approver != "Unknown"


def test_very_large_claim_requires_executive_review():
    claim = make_claim(
        travel_category="international",
        expenses=[
            {
                "category": "flight",
                "amount": 300000,
                "currency": "INR",
                "receipt_id": "RCP-FLIGHT-BIG",
            }
        ],
    )

    facts = evaluate_isolated(claim)

    assert facts.requires_executive_review is True
    assert facts.baseline_decision == Decision.MANUAL_REVIEW


# ---------------------------------------------------------------------------
# Cached data files
# ---------------------------------------------------------------------------


def test_data_files_are_parsed_once():
    """
    The loader is cached, so repeated reads return the same object.

    This is what makes it cheap, and it is also the risk: anything that mutated
    the result would corrupt every later claim.
    """

    from agent.config import load_data_file

    assert load_data_file("receipts.json") is load_data_file("receipts.json")


def test_evaluating_a_claim_does_not_mutate_the_shared_data():
    """Guards the read-only contract the cache depends on."""

    from agent.config import load_data_file

    before = json.dumps(load_data_file("receipts.json"), sort_keys=True)

    for claim in SAMPLE_CLAIMS:
        evaluate_claim(claim)

    after = json.dumps(load_data_file("receipts.json"), sort_keys=True)

    assert before == after
