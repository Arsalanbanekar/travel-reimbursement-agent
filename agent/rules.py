"""
Deterministic reimbursement rule engine.

This module contains every calculation that decides money: limits, per-day and
per-night aggregation, receipt requirements, duplicate detection, the
submission deadline and the approval threshold.

It never calls an LLM. Given the same claim it always returns the same facts,
which is what makes the agent's output auditable and testable.
"""

import json
from collections import defaultdict
from datetime import date
from typing import Iterable

from agent.config import DATA_DIR
from models.schema import (
    ClaimFacts,
    ClaimInput,
    Decision,
    ExpenseFinding,
    ReasonCode,
)


def _load(filename: str) -> dict:
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


# How each category's cap is applied:
#   per_item  — the cap applies to each individual expense line
#   per_night — the cap is multiplied by the number of nights stayed
#   per_day   — the cap applies to the sum of that category on a given date
LIMIT_RULES = {
    "hotel": {"basis": "per_night", "domestic": "domestic_per_night", "international": "international_per_night"},
    "meals": {"basis": "per_day", "domestic": "domestic_per_day", "international": "international_per_day"},
    "taxi": {"basis": "per_day", "domestic": "per_day", "international": "per_day"},
    "parking": {"basis": "per_day", "domestic": "per_day", "international": "per_day"},
    "internet": {"basis": "per_day", "domestic": "per_day", "international": "per_day"},
    "flight": {"basis": "per_item", "domestic": "domestic_max", "international": "international_max"},
    "train": {"basis": "per_item", "domestic": "domestic_per_trip", "international": "domestic_per_trip"},
}


class RuleEngine:
    """Evaluates a claim against the policy data files."""

    def __init__(self, claim_history: list[dict] | None = None):
        self.limits_data = _load("reimbursement_limits.json")
        self.limits = self.limits_data["limits"]
        self.receipts = {
            r["receipt_id"]: r for r in _load("receipts.json")["receipts"]
        }
        self.approval_thresholds = _load("approval_matrix.json")["thresholds"]

        # Previously submitted claims, used for duplicate-receipt detection.
        if claim_history is None:
            claim_history = _load("sample_claims.json")["claims"]
        self.claim_history = claim_history

    # -- helpers ------------------------------------------------------------

    def _limit_for(self, category: str, travel_category: str) -> tuple[float | None, str | None]:
        """Return the cap and policy reference for a category, or (None, None)."""

        rule = LIMIT_RULES.get(category)
        if rule is None:
            return None, None

        config = self.limits[category]
        field = rule["international" if travel_category == "international" else "domestic"]

        return config[field], config["policy_reference"]

    def _expense_date(self, expense, claim: ClaimInput) -> date:
        """
        Resolve the date an expense was incurred.

        Preference order: the expense's own date, then the linked receipt's
        date, then the trip start date. Per-day limits are meaningless without
        a date, so there is always a fallback.
        """

        if expense.expense_date:
            return expense.expense_date

        receipt = self.receipts.get(expense.receipt_id or "")
        if receipt and receipt.get("date"):
            return date.fromisoformat(receipt["date"])

        return claim.travel_start_date

    @staticmethod
    def _allocate(amounts: list[tuple[int, float]], cap: float) -> dict[int, float]:
        """
        Spread a shared cap across several expense lines.

        Lines are filled in order until the cap is used up, so the earliest
        expense is reimbursed first and the overflow lands on later lines.
        """

        allowed: dict[int, float] = {}
        remaining = cap

        for index, amount in amounts:
            grant = min(amount, max(remaining, 0.0))
            allowed[index] = grant
            remaining -= grant

        return allowed

    # -- individual checks --------------------------------------------------

    def _check_receipt(self, finding: ExpenseFinding, expense) -> None:
        """Receipt existence, attachment and agreement with the claim line."""

        threshold = self.limits_data["receipt_threshold"]
        mandatory_above = threshold["mandatory_above"]
        threshold_ref = threshold["policy_reference"]

        receipt = self.receipts.get(expense.receipt_id or "")

        if receipt is None:
            # Below the threshold a receipt is optional (Section 10.4).
            if finding.claimed_amount > mandatory_above:
                finding.reason_codes.append(ReasonCode.RECEIPT_NOT_FOUND)
                finding.policy_references.append(threshold_ref)
                finding.notes.append(
                    f"No receipt on file for {expense.receipt_id or 'this expense'}, "
                    f"which is required above ₹{mandatory_above:,}."
                )
            return

        if not receipt.get("attachment_present", False):
            if finding.claimed_amount > mandatory_above:
                finding.reason_codes.append(ReasonCode.MISSING_RECEIPT)
                finding.policy_references.extend([threshold_ref, "Section 14.1"])
                finding.notes.append(
                    f"Receipt {expense.receipt_id} has no attachment, required "
                    f"above ₹{mandatory_above:,}."
                )
            else:
                finding.notes.append(
                    f"Receipt {expense.receipt_id} has no attachment, but the amount "
                    f"is within the ₹{mandatory_above:,} threshold (Section 10.4)."
                )
            return

        # The receipt exists and is attached — does it actually match the claim?
        if abs(receipt["amount"] - finding.claimed_amount) > 0.01:
            finding.reason_codes.append(ReasonCode.RECEIPT_MISMATCH)
            finding.policy_references.append("Section 14.2")
            finding.notes.append(
                f"Claimed ₹{finding.claimed_amount:,.0f} but receipt "
                f"{expense.receipt_id} is for ₹{receipt['amount']:,.0f}."
            )

        if receipt.get("category", "").lower() != finding.category:
            finding.reason_codes.append(ReasonCode.RECEIPT_MISMATCH)
            finding.policy_references.append("Section 14.2")
            finding.notes.append(
                f"Claimed as {finding.category} but receipt {expense.receipt_id} "
                f"is categorised as {receipt.get('category')}."
            )

    def _check_duplicate(self, finding: ExpenseFinding, claim: ClaimInput) -> None:
        """
        Flag a receipt already used on an earlier claim.

        Only the *later* submission is flagged. Comparing submission dates
        prevents the original, legitimate claim from being penalised.
        """

        if not finding.receipt_id:
            return

        for prior in self.claim_history:
            if prior["claim_id"] == claim.claim_id:
                continue

            uses_receipt = any(
                e.get("receipt_id") == finding.receipt_id
                for e in prior.get("expenses", [])
            )
            if not uses_receipt:
                continue

            prior_submitted = date.fromisoformat(prior["submission_date"])
            # Ties break on claim_id so the result is stable either way.
            is_later = (claim.submission_date, claim.claim_id) > (
                prior_submitted,
                prior["claim_id"],
            )

            if is_later:
                finding.reason_codes.append(ReasonCode.DUPLICATE_RECEIPT)
                finding.policy_references.extend(["Section 12.1", "Section 12.2"])
                finding.notes.append(
                    f"Receipt {finding.receipt_id} was already submitted on "
                    f"{prior['claim_id']} ({prior['submission_date']})."
                )

    def _apply_limits(
        self,
        findings: list[ExpenseFinding],
        claim: ClaimInput,
        nights: int,
    ) -> None:
        """Apply per-item, per-night and per-day caps and set allowed amounts."""

        travel_category = claim.travel_category

        per_day_groups: dict[tuple[str, date], list[tuple[int, float]]] = defaultdict(list)
        per_night_groups: dict[str, list[tuple[int, float]]] = defaultdict(list)

        for finding in findings:
            category = finding.category
            rule = LIMIT_RULES.get(category)

            if rule is None:
                # An unrecognised category is never silently approved.
                finding.allowed_amount = 0.0
                finding.excess_amount = 0.0
                finding.reason_codes.append(ReasonCode.UNKNOWN_CATEGORY)
                finding.policy_references.append("Section 14.6")
                finding.notes.append(
                    f"'{category}' is not a recognised expense category and needs a human decision."
                )
                continue

            limit, policy_ref = self._limit_for(category, travel_category)
            finding.limit_applied = limit
            if policy_ref:
                finding.policy_references.append(policy_ref)

            if rule["basis"] == "per_item":
                finding.limit_basis = f"₹{limit:,.0f} per item"
                finding.allowed_amount = min(finding.claimed_amount, limit)

            elif rule["basis"] == "per_night":
                per_night_groups[category].append((finding.index, finding.claimed_amount))

            else:
                key = (category, finding.expense_date)
                per_day_groups[key].append((finding.index, finding.claimed_amount))

        by_index = {f.index: f for f in findings}

        # Hotels: the nightly cap covers the whole stay.
        for category, items in per_night_groups.items():
            limit, _ = self._limit_for(category, travel_category)
            cap = limit * max(nights, 1)

            for index, allowed in self._allocate(items, cap).items():
                by_index[index].allowed_amount = allowed
                by_index[index].limit_basis = (
                    f"₹{limit:,.0f} per night × {max(nights, 1)} night(s) = ₹{cap:,.0f}"
                )

        # Meals, taxi, parking, internet: the cap is a daily total.
        for (category, day), items in per_day_groups.items():
            limit, _ = self._limit_for(category, travel_category)
            cap = limit

            # Taxi also has a hard per-trip ceiling on top of the daily cap.
            if category == "taxi":
                trip_max = self.limits["taxi"]["single_trip_max"]
                capped_items = []
                for index, amount in items:
                    if amount > trip_max:
                        by_index[index].notes.append(
                            f"Single trip of ₹{amount:,.0f} exceeds the ₹{trip_max:,} "
                            f"per-trip ceiling (Section 7.2)."
                        )
                        capped_items.append((index, trip_max))
                    else:
                        capped_items.append((index, amount))
                items = capped_items

            for index, allowed in self._allocate(items, cap).items():
                by_index[index].allowed_amount = allowed
                by_index[index].limit_basis = (
                    f"₹{limit:,.0f} per day for {category} on {day.isoformat()}"
                )

        # Excess is whatever the policy would not cover.
        for finding in findings:
            finding.excess_amount = round(
                max(finding.claimed_amount - finding.allowed_amount, 0.0), 2
            )
            if finding.excess_amount > 0 and ReasonCode.UNKNOWN_CATEGORY not in finding.reason_codes:
                finding.reason_codes.append(ReasonCode.LIMIT_EXCEEDED)

    def _approver_for(self, total: float) -> dict:
        """
        Find the approval band for a total.

        Bands in the matrix have gaps (…25000 then 25001…), so rather than
        requiring an exact hit the first band whose ceiling covers the total
        wins. Without this, ₹25,000.50 would fall through to 'Unknown'.
        """

        for threshold in sorted(self.approval_thresholds, key=lambda t: t["min_amount"]):
            ceiling = threshold["max_amount"]

            if ceiling is None or total <= ceiling:
                return {
                    "required_approver": threshold["required_approver"],
                    "policy_reference": threshold["policy_reference"],
                    "requires_manual_review": threshold.get("requires_manual_review", False),
                }

        return {
            "required_approver": "Finance Head",
            "policy_reference": "Section 14.4",
            "requires_manual_review": True,
        }

    # -- entry point --------------------------------------------------------

    def evaluate(self, claim: ClaimInput) -> ClaimFacts:
        """Run every deterministic check and return the complete facts."""

        nights = max((claim.travel_end_date - claim.travel_start_date).days, 0)

        findings: list[ExpenseFinding] = []

        for index, expense in enumerate(claim.expenses):
            finding = ExpenseFinding(
                index=index,
                category=expense.category.strip().lower(),
                description=expense.description,
                receipt_id=expense.receipt_id,
                expense_date=self._expense_date(expense, claim),
                claimed_amount=expense.amount,
                allowed_amount=0.0,
                excess_amount=0.0,
            )

            if expense.currency.upper() != claim.currency.upper():
                finding.reason_codes.append(ReasonCode.CURRENCY_MISMATCH)
                finding.notes.append(
                    f"Expense is in {expense.currency} but the claim is in {claim.currency}; "
                    "no conversion rate is available."
                )

            self._check_receipt(finding, expense)
            self._check_duplicate(finding, claim)

            findings.append(finding)

        self._apply_limits(findings, claim, nights)

        # --- claim-level aggregation ---

        total_claimed = round(sum(f.claimed_amount for f in findings), 2)
        total_allowed = round(sum(f.allowed_amount for f in findings), 2)
        total_rejected = round(total_claimed - total_allowed, 2)

        window = self.limits_data["submission_window_days"]
        days_to_submit = (claim.submission_date - claim.travel_end_date).days
        within_window = days_to_submit <= window

        approval = self._approver_for(total_claimed)

        reason_codes: list[ReasonCode] = []
        for finding in findings:
            for code in finding.reason_codes:
                if code not in reason_codes:
                    reason_codes.append(code)

        if not within_window:
            reason_codes.insert(0, ReasonCode.LATE_SUBMISSION)

        if approval["requires_manual_review"]:
            reason_codes.append(ReasonCode.EXCEEDS_APPROVAL_THRESHOLD)

        # --- decide ---

        # Section 14 conditions take precedence: the policy says to route these
        # to a human "rather than automatic approval or rejection".
        manual_review_codes = {
            ReasonCode.MISSING_RECEIPT,
            ReasonCode.RECEIPT_NOT_FOUND,
            ReasonCode.RECEIPT_MISMATCH,
            ReasonCode.DUPLICATE_RECEIPT,
            ReasonCode.UNKNOWN_CATEGORY,
            ReasonCode.CURRENCY_MISMATCH,
            ReasonCode.EXCEEDS_APPROVAL_THRESHOLD,
        }

        blocking = [
            note
            for finding in findings
            for note in finding.notes
            if set(finding.reason_codes) & manual_review_codes
        ]

        if set(reason_codes) & manual_review_codes:
            baseline = Decision.MANUAL_REVIEW
        elif not within_window:
            baseline = Decision.REJECT
            blocking.append(
                f"Submitted {days_to_submit} days after travel ended; the limit "
                f"is {window} days (Section 11.1, Section 11.2)."
            )
        elif total_rejected > 0:
            baseline = Decision.PARTIALLY_APPROVE
        else:
            baseline = Decision.APPROVE

        # A rejected claim reimburses nothing, regardless of the line maths.
        if baseline == Decision.REJECT:
            total_allowed = 0.0
            total_rejected = total_claimed

        missing_documents = [
            f"Itemised receipt for {f.receipt_id or f.category} (₹{f.claimed_amount:,.0f})"
            for f in findings
            if {ReasonCode.MISSING_RECEIPT, ReasonCode.RECEIPT_NOT_FOUND} & set(f.reason_codes)
        ]

        policy_references: list[str] = []
        for finding in findings:
            for ref in finding.policy_references:
                if ref not in policy_references:
                    policy_references.append(ref)

        if not within_window:
            policy_references.append(self.limits_data["submission_policy_reference"])
            policy_references.append("Section 11.2")

        if approval.get("policy_reference") and approval["policy_reference"] not in policy_references:
            policy_references.append(approval["policy_reference"])

        return ClaimFacts(
            claim_id=claim.claim_id,
            travel_category=claim.travel_category,
            total_claimed=total_claimed,
            total_allowed=total_allowed,
            total_rejected=total_rejected,
            days_to_submit=days_to_submit,
            submission_window_days=window,
            within_submission_window=within_window,
            required_approver=approval["required_approver"],
            approval_policy_reference=approval.get("policy_reference"),
            requires_executive_review=approval["requires_manual_review"],
            nights=nights,
            expense_findings=findings,
            reason_codes=reason_codes,
            missing_documents=missing_documents,
            policy_references=policy_references,
            blocking_issues=blocking,
            baseline_decision=baseline,
        )


def evaluate_claim(claim: dict | ClaimInput, claim_history: Iterable[dict] | None = None) -> ClaimFacts:
    """Convenience wrapper: validate a raw claim dict, then evaluate it."""

    validated = claim if isinstance(claim, ClaimInput) else ClaimInput.model_validate(claim)

    history = list(claim_history) if claim_history is not None else None

    return RuleEngine(claim_history=history).evaluate(validated)


if __name__ == "__main__":

    claims = _load("sample_claims.json")["claims"]

    for raw in claims:
        facts = evaluate_claim(raw)
        expected = raw.get("expected_decision")
        match = "OK " if facts.baseline_decision.value == expected else "DIFF"

        print(
            f"{match} {facts.claim_id}  "
            f"rules={facts.baseline_decision.value:<18} expected={expected:<18} "
            f"allowed=₹{facts.total_allowed:>9,.0f}  rejected=₹{facts.total_rejected:>8,.0f}  "
            f"codes={[c.value for c in facts.reason_codes]}"
        )
