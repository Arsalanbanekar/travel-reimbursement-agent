# """
# LangChain tool definitions for the reimbursement approval agent.

# Future purpose — implement each as a separate @tool:

# 1. Policy Lookup Tool
#    - Retrieve relevant travel policy sections from FAISS (RAG).

# 2. Receipt Validation Tool
#    - Check missing receipt, vendor, amount, category, travel date.

# 3. Reimbursement Limit Checker
#    - Validate against reimbursement_limits.json; compute approved/rejected amounts.

# 4. Duplicate Receipt Checker
#    - Detect duplicate receipt IDs against receipts.json / claim history.

# 5. Approval Threshold Checker
#    - Determine required approver from approval_matrix.json based on amount.

# 6. Output Validator
#    - Validate final JSON response against Pydantic schema in models/schema.py.

# Assignment mapping:
# - Tool/function usage — at least two meaningful tools required (Section 3);
#   all six listed above satisfy and exceed the minimum.
# - Business correctness: limits, receipts, duplicates, approval thresholds (Section 6).
# """






"""
Business tools for the Travel Reimbursement Approval Agent.

Each tool performs one focused task.

The LLM will decide which tools to call.
"""

import json
from pathlib import Path

from langchain.tools import tool

from rag.retriever import PolicyRetriever


DATA_DIR = Path("data")


def load_json(filename: str):
    """Load JSON data from the data directory."""
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


policy_retriever = PolicyRetriever()

@tool
def policy_lookup_tool(query: str) -> str:
    """
    Retrieve relevant travel policy sections.
    """

    docs = policy_retriever.retrieve(query)

    if not docs:
        return "No relevant policy found."

    return "\n\n".join(doc.page_content for doc in docs)


@tool
def receipt_validation_tool(receipt_id: str) -> dict:
    """
    Validate whether a receipt exists and whether the required attachment is present.
    """

    receipts_data = load_json("receipts.json")

    receipts = receipts_data.get("receipts", [])

    receipt = next(
        (r for r in receipts if r["receipt_id"] == receipt_id),
        None,
    )

    if receipt is None:
        return {
            "valid": False,
            "message": "Receipt not found.",
        }

    if not receipt.get("attachment_present", False):
        return {
            "valid": False,
            "message": "Receipt attachment missing.",
            "receipt": receipt,
        }

    return {
        "valid": True,
        "message": "Receipt validated successfully.",
        "receipt": receipt,
    }




@tool
def reimbursement_limit_checker_tool(expenses: list) -> dict:
    """
    Check each expense against reimbursement limits.
    Returns approved amount, rejected amount and exceeded categories.
    """

    limits_data = load_json("reimbursement_limits.json")
    limits = limits_data["limits"]

    approved_amount = 0
    rejected_amount = 0
    exceeded_categories = []
    policy_references = []

    for expense in expenses:

        category = expense["category"].lower()
        amount = expense["amount"]

        if category == "hotel":
            limit = limits["hotel"]["domestic_per_night"]
            policy_ref = limits["hotel"]["policy_reference"]

        elif category == "meals":
            limit = limits["meals"]["domestic_per_day"]
            policy_ref = limits["meals"]["policy_reference"]

        elif category == "taxi":
            limit = limits["taxi"]["per_day"]
            policy_ref = limits["taxi"]["policy_reference"]

        elif category == "parking":
            limit = limits["parking"]["per_day"]
            policy_ref = limits["parking"]["policy_reference"]

        elif category == "internet":
            limit = limits["internet"]["per_day"]
            policy_ref = limits["internet"]["policy_reference"]

        elif category == "flight":
            limit = limits["flight"]["domestic_max"]
            policy_ref = limits["flight"]["policy_reference"]

        elif category == "train":
            limit = limits["train"]["domestic_per_trip"]
            policy_ref = limits["train"]["policy_reference"]

        else:
            approved_amount += amount
            continue

        if amount <= limit:
            approved_amount += amount
        else:
            approved_amount += limit
            rejected_amount += amount - limit
            exceeded_categories.append(category)
            policy_references.append(policy_ref)

    return {
        "approved_amount": approved_amount,
        "rejected_amount": rejected_amount,
        "exceeded_categories": exceeded_categories,
        "policy_references": policy_references,
    }





# @tool
# def duplicate_receipt_checker_tool(receipt_id: str) -> dict:
#     """
#     Check whether a receipt ID has been used in more than one claim.
#     """

#     claims_data = load_json("sample_claims.json")
#     claims = claims_data["claims"]

#     used_in_claims = []

#     for claim in claims:
#         for expense in claim.get("expenses", []):
#             if expense.get("receipt_id") == receipt_id:
#                 used_in_claims.append(claim["claim_id"])

#     if len(used_in_claims) > 1:
#         return {
#             "duplicate": True,
#             "receipt_id": receipt_id,
#             "used_in_claims": used_in_claims,
#             "message": "Duplicate receipt detected."
#         }

#     return {
#         "duplicate": False,
#         "receipt_id": receipt_id,
#         "used_in_claims": used_in_claims,
#         "message": "Receipt is unique."
#     }




# @tool
# def duplicate_receipt_checker_tool(receipt_id: str, current_claim_id: str):

#     claims_data = load_json("sample_claims.json")
#     claims = claims_data["claims"]

#     used_in_claims = []

#     for claim in claims:

#         # Ignore the claim currently being evaluated
#         if claim["claim_id"] == current_claim_id:
#             continue

#         for expense in claim["expenses"]:
#             if expense["receipt_id"] == receipt_id:
#                 used_in_claims.append(claim["claim_id"])

#     return {
#         "duplicate": len(used_in_claims) > 0,
#         "receipt_id": receipt_id,
#         "used_in_claims": used_in_claims,
#         "message": (
#             "Duplicate receipt detected."
#             if used_in_claims
#             else "Receipt is unique."
#         )
#     }




@tool
def duplicate_receipt_checker_tool(
    receipt_id: str,
    current_claim_id: str,
) -> dict:
    """
    Check whether a receipt ID has already been used
    in another reimbursement claim.
    """

    claims_data = load_json("sample_claims.json")
    claims = claims_data["claims"]

    used_in_claims = []

    for claim in claims:

        # Ignore the claim currently being evaluated
        if claim["claim_id"] == current_claim_id:
            continue

        for expense in claim["expenses"]:
            if expense["receipt_id"] == receipt_id:
                used_in_claims.append(claim["claim_id"])

    return {
        "duplicate": len(used_in_claims) > 0,
        "receipt_id": receipt_id,
        "used_in_claims": used_in_claims,
        "message": (
            "Duplicate receipt detected."
            if used_in_claims
            else "Receipt is unique."
        ),
    }



@tool
def approval_threshold_checker_tool(total_amount: float) -> dict:
    """
    Determine the required approver based on the total claim amount.
    """

    approval_data = load_json("approval_matrix.json")
    thresholds = approval_data["thresholds"]

    for threshold in thresholds:

        min_amount = threshold["min_amount"]
        max_amount = threshold["max_amount"]

        if max_amount is None:
            if total_amount >= min_amount:
                return {
                    "required_approver": threshold["required_approver"],
                    "manual_review_required": threshold.get(
                        "requires_manual_review",
                        False,
                    ),
                    "policy_reference": threshold["policy_reference"],
                }

        elif min_amount <= total_amount <= max_amount:
            return {
                "required_approver": threshold["required_approver"],
                "manual_review_required": threshold.get(
                    "requires_manual_review",
                    False,
                ),
                "policy_reference": threshold["policy_reference"],
            }

    return {
        "required_approver": "Unknown",
        "manual_review_required": True,
        "policy_reference": None,
    }