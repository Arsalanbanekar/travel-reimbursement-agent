"""
Shared Pydantic schemas for the Travel Reimbursement Approval Agent.

These models define the data contracts shared between the Streamlit UI, the
LangGraph agent, the rule engine and the tools.

Business logic does NOT belong here — see agent/rules.py.
"""

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Decision(str, Enum):
    """Possible reimbursement outcomes."""

    APPROVE = "Approve"
    PARTIALLY_APPROVE = "Partially Approve"
    REJECT = "Reject"
    MANUAL_REVIEW = "Manual Review"


class ReasonCode(str, Enum):
    """
    Machine-readable reason a claim was not straightforwardly approved.

    Reason codes are produced only by the rule engine, never by the LLM, so
    they are always traceable to a concrete check.
    """

    LATE_SUBMISSION = "LATE_SUBMISSION"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    MISSING_RECEIPT = "MISSING_RECEIPT"
    RECEIPT_NOT_FOUND = "RECEIPT_NOT_FOUND"
    RECEIPT_MISMATCH = "RECEIPT_MISMATCH"
    DUPLICATE_RECEIPT = "DUPLICATE_RECEIPT"
    UNKNOWN_CATEGORY = "UNKNOWN_CATEGORY"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    EXCEEDS_APPROVAL_THRESHOLD = "EXCEEDS_APPROVAL_THRESHOLD"


# ---------------------------------------------------------------------------
# Claim intake
# ---------------------------------------------------------------------------


class ExpenseItem(BaseModel):
    """A single expense line inside a reimbursement claim."""

    model_config = ConfigDict(populate_by_name=True)

    category: str = Field(..., description="hotel, meals, taxi, flight, train, parking, internet")
    amount: float = Field(..., gt=0, description="Claimed amount")
    currency: str = Field(default="INR")
    receipt_id: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)

    # Optional: when absent, the engine falls back to the receipt's own date.
    # Named expense_date to avoid shadowing datetime.date; accepts "date" in JSON.
    expense_date: Optional[date] = Field(default=None, alias="date")


class ClaimInput(BaseModel):
    """
    An employee reimbursement claim.

    Field names mirror data/sample_claims.json so uploaded files validate
    without translation. Unknown extra keys (expected_decision and the other
    evaluation fields) are ignored rather than rejected.
    """

    claim_id: str
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None
    department: Optional[str] = None

    travel_category: str = Field(default="domestic")
    trip_purpose: Optional[str] = None

    travel_start_date: date
    travel_end_date: date
    submission_date: date

    manager_pre_approved: bool = Field(default=False)
    currency: str = Field(default="INR")

    expenses: List[ExpenseItem] = Field(..., min_length=1)

    @field_validator("travel_category")
    @classmethod
    def _normalise_category(cls, value: str) -> str:
        return value.strip().lower()


# ---------------------------------------------------------------------------
# Rule engine output ("the facts")
# ---------------------------------------------------------------------------


class ExpenseFinding(BaseModel):
    """What the rule engine determined about one expense line."""

    index: int
    category: str
    description: Optional[str] = None
    receipt_id: Optional[str] = None
    expense_date: Optional[date] = None

    claimed_amount: float
    allowed_amount: float
    excess_amount: float

    limit_applied: Optional[float] = None
    limit_basis: Optional[str] = Field(
        default=None,
        description="Human-readable explanation, e.g. '₹5,000/night × 1 night'",
    )

    reason_codes: List[ReasonCode] = Field(default_factory=list)
    policy_references: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ClaimFacts(BaseModel):
    """
    The complete deterministic assessment of a claim.

    Every number here is computed in Python. The LLM may explain these facts
    but is never allowed to change them.
    """

    claim_id: str
    travel_category: str

    total_claimed: float
    total_allowed: float
    total_rejected: float

    days_to_submit: int
    submission_window_days: int
    within_submission_window: bool

    required_approver: str
    approval_policy_reference: Optional[str] = None
    requires_executive_review: bool = False

    nights: int = 0

    expense_findings: List[ExpenseFinding] = Field(default_factory=list)

    reason_codes: List[ReasonCode] = Field(default_factory=list)
    missing_documents: List[str] = Field(default_factory=list)
    policy_references: List[str] = Field(default_factory=list)
    blocking_issues: List[str] = Field(
        default_factory=list,
        description="Plain-language issues that force Manual Review.",
    )

    baseline_decision: Decision = Field(
        description="Decision implied by the rules alone, before the LLM reasons about it.",
    )


# ---------------------------------------------------------------------------
# Agent output
# ---------------------------------------------------------------------------


class AuditTrailStep(BaseModel):
    """One step executed while evaluating a claim."""

    step: int
    tool_name: str
    status: str
    details: str


class ReimbursementDecision(BaseModel):
    """Final structured response returned by the agent."""

    decision: Decision

    approved_amount: float = Field(default=0.0, ge=0.0)
    rejected_amount: float = Field(default=0.0, ge=0.0)

    missing_documents: List[str] = Field(default_factory=list)
    policy_references: List[str] = Field(default_factory=list)
    reason_codes: List[ReasonCode] = Field(default_factory=list)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    explanation: str

    required_approver: Optional[str] = None
    audit_trail: List[AuditTrailStep] = Field(default_factory=list)


class LLMDecision(BaseModel):
    """
    The narrow slice of the decision the LLM is actually trusted to produce.

    Amounts, reason codes and approver are deliberately excluded — those come
    from the rule engine, so the model cannot alter them.
    """

    decision: Decision
    explanation: str = Field(..., description="2-4 sentences citing the policy sections used.")
    policy_references: List[str] = Field(default_factory=list)
