"""
Evaluation harness.

Runs every claim in the golden set through the full agent and scores three
things that matter independently:

  decision accuracy — did it reach the right outcome?
  amount accuracy   — are approved/rejected amounts exactly right?
  citation validity — is every cited section real and actually retrieved?

Writes one decision file per claim to outputs/ and a summary, which doubles as
the "sample outputs" deliverable.

Usage:
    py -3.13 -m evals.run_eval
    py -3.13 -m evals.run_eval --claim CLM-002
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from agent.config import DATA_DIR, GROQ_MODEL, OUTPUTS_DIR
from agent.graph import run_claim
from rag.retriever import verify_policy_references


def load_golden_set() -> list[dict]:
    with open(DATA_DIR / "sample_claims.json", encoding="utf-8") as f:
        return json.load(f)["claims"]


def score_claim(claim: dict) -> dict:
    """Run one claim and compare the result with its expected outcome."""

    started = time.perf_counter()
    result = run_claim(claim)
    elapsed = time.perf_counter() - started

    decision = result["decision"]

    expected_decision = claim.get("expected_decision")
    decision_ok = decision["decision"] == expected_decision

    # Claims where the rules alone cannot reach the right answer: the LLM has
    # to notice something. Tracked separately because they are the ones that
    # justify having an LLM in the loop at all.
    needs_judgement = (
        claim.get("expected_baseline_decision", expected_decision) != expected_decision
    )

    # Manual Review claims deliberately carry no expected amounts.
    expected_approved = claim.get("expected_approved_amount")
    if expected_approved is None:
        amount_ok = None
    else:
        amount_ok = (
            decision["approved_amount"] == expected_approved
            and decision["rejected_amount"] == claim.get("expected_rejected_amount")
        )

    _, unsupported = verify_policy_references(
        decision["policy_references"],
        allowed_sections=set(result["retrieved_sections"]),
    )

    tools_used = [
        step["tool_name"]
        for step in result["audit_trail"]
        if step["tool_name"]
        in {
            "search_policy",
            "get_policy_section",
            "get_receipt",
            "find_claims_using_receipt",
            "get_category_limits",
        }
    ]

    # A run where the LLM never answered did not test what this eval is for.
    # It is especially misleading on judgement cases: the fail-safe routes them
    # to Manual Review, which is the expected answer, so they "pass" without the
    # model having reasoned at all. Degraded runs are excluded from scoring.
    degraded = bool(result.get("llm_error"))

    return {
        "claim_id": claim["claim_id"],
        "expected_decision": expected_decision,
        "actual_decision": decision["decision"],
        "decision_ok": decision_ok,
        "degraded": degraded,
        "needs_judgement": needs_judgement,
        "expected_approved": expected_approved,
        "actual_approved": decision["approved_amount"],
        "expected_rejected": claim.get("expected_rejected_amount"),
        "actual_rejected": decision["rejected_amount"],
        "amount_ok": amount_ok,
        "confidence": decision["confidence"],
        "reason_codes": decision["reason_codes"],
        "unsupported_citations": unsupported,
        "tools_used": tools_used,
        "steps": len(result["audit_trail"]),
        "seconds": round(elapsed, 1),
        "llm_error": result.get("llm_error"),
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the reimbursement agent.")
    parser.add_argument("--claim", help="Evaluate a single claim ID.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write decision files to outputs/.",
    )
    args = parser.parse_args()

    claims = load_golden_set()
    if args.claim:
        claims = [c for c in claims if c["claim_id"] == args.claim]
        if not claims:
            print(f"No claim with ID {args.claim}")
            return 2

    OUTPUTS_DIR.mkdir(exist_ok=True)

    print(f"Model: {GROQ_MODEL}")
    print(f"Evaluating {len(claims)} claim(s)\n")

    header = (
        f"{'CLAIM':<9} {'BY':<6} {'EXPECTED':<18} {'ACTUAL':<18} "
        f"{'DEC':<5} {'AMT':<5} {'CITE':<5} {'CONF':<5} {'TOOLS':<6} {'SEC'}"
    )
    print(header)
    print("-" * len(header))

    scores = []

    for claim in claims:
        score = score_claim(claim)
        scores.append(score)

        amount_mark = {True: "OK", False: "FAIL", None: "-"}[score["amount_ok"]]

        print(
            f"{score['claim_id']:<9} "
            f"{'LLM' if score['needs_judgement'] else 'rules':<6} "
            f"{score['expected_decision']:<18} "
            f"{score['actual_decision']:<18} "
            f"{'n/a' if score['degraded'] else ('OK' if score['decision_ok'] else 'FAIL'):<5} "
            f"{amount_mark:<5} "
            f"{'OK' if not score['unsupported_citations'] else 'BAD':<5} "
            f"{score['confidence']:<5} "
            f"{len(score['tools_used']):<6} "
            f"{score['seconds']}"
        )

        if not args.no_save:
            path = OUTPUTS_DIR / f"{score['claim_id']}_decision.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(score["result"], f, indent=2, ensure_ascii=False)

    # --- summary ---

    degraded = [s for s in scores if s["degraded"]]
    valid = [s for s in scores if not s["degraded"]]

    total = len(valid)
    decisions_right = sum(1 for s in valid if s["decision_ok"])
    scored_amounts = [s for s in valid if s["amount_ok"] is not None]
    amounts_right = sum(1 for s in scored_amounts if s["amount_ok"])
    clean_citations = sum(1 for s in valid if not s["unsupported_citations"])

    judgement = [s for s in valid if s["needs_judgement"]]
    judgement_right = sum(1 for s in judgement if s["decision_ok"])
    rules_only = [s for s in valid if not s["needs_judgement"]]
    rules_right = sum(1 for s in rules_only if s["decision_ok"])

    def ratio(right: int, out_of: int) -> str:
        return f"{right}/{out_of}" if out_of else "no valid runs"

    print()
    if degraded:
        print(
            f"!! {len(degraded)} of {len(scores)} run(s) degraded - the LLM did not "
            "answer, so they are EXCLUDED from scoring below."
        )
        print("   Usually a rate limit. Wait for the quota to reset and re-run.")
        for score in degraded:
            print(f"     {score['claim_id']}: {str(score['llm_error'])[:70]}")
        print()

    print(f"Decision accuracy      : {ratio(decisions_right, total)}")
    print(f"  rule-decidable       : {ratio(rules_right, len(rules_only))}")
    print(f"  needs LLM judgement  : {ratio(judgement_right, len(judgement))}")
    print(f"Amount accuracy        : {ratio(amounts_right, len(scored_amounts))}")
    print(f"Citation validity      : {ratio(clean_citations, total)}")

    failures = [s for s in valid if not s["decision_ok"]]
    if failures:
        print("\nFailures:")
        for score in failures:
            print(
                f"  {score['claim_id']}: expected {score['expected_decision']}, "
                f"got {score['actual_decision']}"
            )

    if not args.no_save:
        summary = {
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": GROQ_MODEL,
            "degraded_runs": len(degraded),
            "decision_accuracy": f"{decisions_right}/{total}",
            "rule_decidable_accuracy": f"{rules_right}/{len(rules_only)}",
            "judgement_case_accuracy": f"{judgement_right}/{len(judgement)}",
            "amount_accuracy": f"{amounts_right}/{len(scored_amounts)}",
            "citation_validity": f"{clean_citations}/{total}",
            "claims": [
                {k: v for k, v in score.items() if k != "result"} for score in scores
            ],
        }

        with open(OUTPUTS_DIR / "eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nWrote decisions and eval_summary.json to {OUTPUTS_DIR}")

    # Degraded runs are a failure of the run, not of the agent, but either way
    # this did not demonstrate what it set out to.
    return 0 if (decisions_right == total and not degraded) else 1


if __name__ == "__main__":
    sys.exit(main())
