# """
# Pydantic schemas for reimbursement claims and agent decisions.

# Future purpose:
# - ClaimInput: fields from form/JSON (employee ID, name, department, category, etc.).
# - ReimbursementDecision: decision enum (Approve | Partially Approve | Reject | Manual Review),
#   approved_amount, rejected_amount, missing_documents, policy_references, confidence,
#   reasoning_summary, audit_trail (tool calls, retrieved policy).
# - Tool-specific response models for consistent tool outputs.

# Assignment mapping:
# - Structured output (Section 3): decision, amounts, missing docs, policy refs,
#   confidence, explanation.
# - Output validation via Pydantic (Section 6: Reliability).
# - Manual Review as a first-class decision outcome (Section 3).
# """



"""
Shared Pydantic schemas for the Travel Reimbursement Approval Agent.

These models define the common data contracts shared between:
- Streamlit UI
- LangChain Agent
- RAG Pipeline
- Tool Functions

Business logic should NOT be implemented here.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Decision(str, Enum):
    """Possible reimbursement outcomes."""

    APPROVE = "Approve"
    PARTIALLY_APPROVE = "Partially Approve"
    REJECT = "Reject"
    MANUAL_REVIEW = "Manual Review"


class ExpenseItem(BaseModel):
    """Represents a single expense item in a reimbursement claim."""

    category: str = Field(..., description="Expense category (Hotel, Meals, Taxi, Flight, etc.)")
    amount: float = Field(..., gt=0, description="Expense amount")
    currency: str = Field(default="INR", description="Currency code")
    receipt_id: str = Field(..., description="Receipt identifier")
    travel_date: str = Field(..., description="Date of the expense (YYYY-MM-DD)")
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the expense"
    )


class ClaimInput(BaseModel):
    """Employee reimbursement claim."""

    claim_id: str
    employee_id: str
    employee_name: str
    department: str

    trip_start_date: str
    trip_end_date: str
    submission_date: str

    expenses: List[ExpenseItem]


class AuditTrailStep(BaseModel):
    """Represents one step executed during claim evaluation."""

    step: int
    tool_name: str
    status: str
    details: str


class ReimbursementDecision(BaseModel):
    """Final structured response returned by the AI agent."""

    decision: Decision

    approved_amount: float = Field(default=0.0)
    rejected_amount: float = Field(default=0.0)

    missing_documents: List[str] = Field(default_factory=list)

    policy_references: List[str] = Field(default_factory=list)

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1."
    )

    explanation: str

    audit_trail: List[AuditTrailStep] = Field(default_factory=list)