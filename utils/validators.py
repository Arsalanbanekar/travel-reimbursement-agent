"""
Input and business-rule validation helpers (non-LLM).

Future purpose:
- Pre-validate claim form fields before sending to the agent.
- Shared validation logic used by Receipt Validation Tool and Limit Checker.
- Flag conditions that should strongly suggest Manual Review routing.

Assignment mapping:
- Receipt completeness checks (Section 3, tool requirements).
- Reliability and fallbacks (Section 6).
- Manual review for incomplete cases (Section 3).
"""
