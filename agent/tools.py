"""
Tools the agent can call while investigating a claim.

Every tool is read-only. None of them decide anything or compute money — the
rule engine in agent/rules.py has already done that before the agent runs.
These exist so the model can look things up and check its reasoning against
source data instead of guessing.
"""


from langchain_core.tools import tool

from agent.config import load_data_file as _load
from rag.retriever import format_sections, get_retriever

_retriever = get_retriever()


@tool
def search_policy(query: str) -> str:
    """
    Search the travel policy by meaning and return the most relevant clauses.

    Use this when you need to check whether something is covered by policy and
    you do not already know the section number — for example an unusual expense
    description, or whether a category is reimbursable at all.
    """

    try:
        return format_sections(_retriever.retrieve(query, k=3))
    except Exception as exc:
        return f"Policy search unavailable: {type(exc).__name__}"


@tool
def get_policy_section(section_id: str) -> str:
    """
    Return the exact text of one policy section, for example "4.1" or "14.2".

    Use this to read the full wording of a clause before citing it.
    """

    documents = _retriever.get_sections([section_id])

    if not documents:
        return (
            f"Section {section_id} does not exist in this policy. "
            "Do not cite it."
        )

    return format_sections(documents)


@tool
def get_receipt(receipt_id: str) -> dict:
    """
    Look up one receipt in the receipt registry.

    Returns the vendor, date, amount, category and whether the attachment is
    present. Use this to check whether a receipt agrees with the claim line.
    """

    receipts = _load("receipts.json")["receipts"]

    receipt = next(
        (r for r in receipts if r["receipt_id"] == receipt_id),
        None,
    )

    if receipt is None:
        return {
            "found": False,
            "receipt_id": receipt_id,
            "message": "No receipt with this ID exists in the registry.",
        }

    return {"found": True, **receipt}


@tool
def find_claims_using_receipt(receipt_id: str) -> dict:
    """
    List every claim that has used a given receipt ID, with submission dates.

    Use this to confirm a suspected duplicate and to see which claim submitted
    the receipt first.
    """

    claims = _load("sample_claims.json")["claims"]

    matches = [
        {
            "claim_id": claim["claim_id"],
            "employee_name": claim.get("employee_name"),
            "submission_date": claim.get("submission_date"),
        }
        for claim in claims
        if any(e.get("receipt_id") == receipt_id for e in claim.get("expenses", []))
    ]

    return {
        "receipt_id": receipt_id,
        "claim_count": len(matches),
        "claims": sorted(matches, key=lambda c: c["submission_date"] or ""),
    }


@tool
def get_category_limits(category: str) -> dict:
    """
    Return the reimbursement limits for one expense category.

    Categories: hotel, meals, taxi, parking, internet, flight, train.
    Use this to confirm which cap applies and on what basis.
    """

    limits = _load("reimbursement_limits.json")["limits"]

    key = category.strip().lower()

    if key not in limits:
        return {
            "found": False,
            "category": category,
            "known_categories": sorted(limits),
            "message": (
                f"'{category}' is not a reimbursable category in this policy. "
                "Claims containing it require Manual Review."
            ),
        }

    return {"found": True, **limits[key]}


AGENT_TOOLS = [
    search_policy,
    get_policy_section,
    get_receipt,
    find_claims_using_receipt,
    get_category_limits,
]
