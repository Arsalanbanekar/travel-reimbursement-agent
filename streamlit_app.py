# import json
# from pathlib import Path
# import streamlit as st
# from agent.agent import ReimbursementAgent

# # -------------------------------------------------------
# # Page Configuration
# # -------------------------------------------------------
# st.set_page_config(
#     page_title="Travel Reimbursement Approval Agent",
#     page_icon="💼",
#     layout="wide",
# )

# st.title("💼 Travel Reimbursement Approval Agent")
# st.markdown(
#     """
# This application demonstrates an AI-powered reimbursement approval workflow using:
# - Planner (LLM)
# - Retrieval-Augmented Generation (RAG)
# - Business Validation Tools
# - Structured AI Decision Making
# """
# )

# # -------------------------------------------------------
# # Load Sample Claims
# # -------------------------------------------------------
# DATA_PATH = Path("data") / "sample_claims.json"

# with open(DATA_PATH, "r", encoding="utf-8") as f:
#     sample_claims = json.load(f)["claims"]

# claim_lookup = {
#     claim["claim_id"]: claim
#     for claim in sample_claims
# }

# # -------------------------------------------------------
# # Sidebar
# # -------------------------------------------------------
# st.sidebar.header("Claim Input")

# claim_mode = st.sidebar.radio(
#     "Choose Input Method",
#     [
#         "Sample Claim",
#         "Upload JSON",
#     ],
# )

# selected_claim = None

# if claim_mode == "Sample Claim":
#     claim_id = st.sidebar.selectbox(
#         "Select Claim",
#         list(claim_lookup.keys()),
#     )
#     selected_claim = claim_lookup[claim_id]
# else:
#     uploaded_file = st.sidebar.file_uploader(
#         "Upload Claim JSON",
#         type=["json"],
#     )
#     if uploaded_file is not None:
#         selected_claim = json.load(uploaded_file)

# # -------------------------------------------------------
# # Agent
# # -------------------------------------------------------
# agent = ReimbursementAgent()

# # -------------------------------------------------------
# # Evaluate Button
# # -------------------------------------------------------
# run_clicked = st.sidebar.button(
#     "Evaluate Claim",
#     use_container_width=True,
# )

# # -------------------------------------------------------
# # Main UI — Claim Details
# # -------------------------------------------------------
# if selected_claim is not None:

#     st.subheader("Claim Details")

#     left, right = st.columns(2)

#     with left:
#         st.metric(
#             "Claim ID",
#             selected_claim["claim_id"],
#         )
#         st.metric(
#             "Employee",
#             selected_claim["employee_name"],
#         )
#         st.metric(
#             "Department",
#             selected_claim["department"],
#         )

#     with right:
#         st.metric(
#             "Trip Start",
#             selected_claim["travel_start_date"],
#         )
#         st.metric(
#             "Trip End",
#             selected_claim["travel_end_date"],
#         )
#         st.metric(
#             "Submission",
#             selected_claim["submission_date"],
#         )

#     st.divider()

#     st.subheader("Expense Items")
#     st.dataframe(
#         selected_claim["expenses"],
#         use_container_width=True,
#     )

# # -------------------------------------------------------
# # Run Agent
# # -------------------------------------------------------
# if run_clicked and selected_claim is not None:
#     with st.spinner("Running AI Reimbursement Agent..."):
#         result = agent.run_claim(selected_claim)

#     st.success("Evaluation completed.")
#     st.session_state["result"] = result
#     st.session_state["evaluated_claim_id"] = selected_claim["claim_id"]

# # -------------------------------------------------------
# # Display Results
# # -------------------------------------------------------
# if "result" in st.session_state:

#     result = st.session_state["result"]
#     planner_tools = result["selected_tools"]
#     tool_results = result["tool_results"]
#     decision = result["decision"]

#     st.divider()

#     # ===================================================
#     # Planner
#     # ===================================================
#     st.subheader("🧠 Planner Decision")
#     st.write(
#         "The planner LLM selected the following tools before evaluating the claim."
#     )

#     cols = st.columns(len(planner_tools))
#     for i, tool in enumerate(planner_tools):
#         with cols[i]:
#             st.success(tool)

#     st.divider()

#     # ===================================================
#     # Tool Outputs
#     # ===================================================
#     st.subheader("🛠 Tool Execution Results")

#     if "policy" in tool_results:
#         with st.expander("📘 Retrieved Policy", expanded=False):
#             st.markdown(tool_results["policy"])

#     if "receipt_validation" in tool_results:
#         with st.expander("🧾 Receipt Validation", expanded=False):
#             for receipt in tool_results["receipt_validation"]:
#                 st.json(receipt)

#     if "limit_check" in tool_results:
#         with st.expander("💰 Reimbursement Limit Check", expanded=False):
#             st.json(tool_results["limit_check"])

#     if "duplicate_check" in tool_results:
#         with st.expander("🔁 Duplicate Receipt Check", expanded=False):
#             for duplicate in tool_results["duplicate_check"]:
#                 st.json(duplicate)

#     if "threshold_check" in tool_results:
#         with st.expander("👤 Approval Threshold", expanded=False):
#             st.json(tool_results["threshold_check"])

#     st.divider()

#     # ===================================================
#     # Final Decision
#     # ===================================================
#     st.subheader("✅ Final Decision")

#     decision_value = decision["decision"]

#     if decision_value == "Approve":
#         st.success(f"🟢 {decision_value}")
#     elif decision_value == "Partially Approve":
#         st.warning(f"🟡 {decision_value}")
#     elif decision_value == "Reject":
#         st.error(f"🔴 {decision_value}")
#     else:
#         st.info(f"🟠 {decision_value}")

#     st.divider()

#     # -------------------------------------------------------
#     # Decision Summary
#     # -------------------------------------------------------
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric(
#             "Approved Amount",
#             f"₹ {decision['approved_amount']:,}"
#         )

#     with col2:
#         st.metric(
#             "Rejected Amount",
#             f"₹ {decision['rejected_amount']:,}"
#         )

#     with col3:
#         st.metric(
#             "Confidence",
#             f"{decision['confidence'] * 100:.1f}%"
#         )

#     st.divider()

#     # -------------------------------------------------------
#     # Missing Documents
#     # -------------------------------------------------------
#     st.subheader("📄 Missing Documents")

#     missing_docs = decision.get("missing_documents", [])

#     if missing_docs:
#         for doc in missing_docs:
#             st.write(f"• {doc}")
#     else:
#         st.success("No missing documents.")

#     st.divider()

#     # -------------------------------------------------------
#     # Policy References
#     # -------------------------------------------------------
#     st.subheader("📚 Policy References")

#     policy_refs = decision.get("policy_references", [])

#     if policy_refs:
#         for ref in policy_refs:
#             st.write(f"• {ref}")
#     else:
#         st.info("No policy references returned.")

#     st.divider()

#     # -------------------------------------------------------
#     # Explanation
#     # -------------------------------------------------------
#     st.subheader("💬 AI Explanation")

#     st.write(
#         decision.get(
#             "explanation",
#             "No explanation available."
#         )
#     )

#     st.divider()

#     # -------------------------------------------------------
#     # Raw JSON
#     # -------------------------------------------------------
#     with st.expander("📦 Full JSON Output", expanded=False):
#         st.json(result)

#     # -------------------------------------------------------
#     # Download Button
#     # -------------------------------------------------------
#     st.download_button(
#         label="⬇ Download Decision JSON",
#         data=json.dumps(result, indent=4),
#         file_name=f"{st.session_state['evaluated_claim_id']}_decision.json",
#         mime="application/json",
#     )

# # -------------------------------------------------------
# # Footer
# # -------------------------------------------------------
# st.divider()
# st.caption(
#     "Built for the HCLTech GenAI Developer Assignment | "
#     "Planner → RAG → Business Tools → LLM Decision"
# )











import json
from pathlib import Path

import streamlit as st

from agent.agent import ReimbursementAgent


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Travel Reimbursement Approval Agent",
    page_icon="💼",
    layout="wide",
)

st.title("💼 Travel Reimbursement Approval Agent")

st.markdown(
    """
This application demonstrates an AI-powered reimbursement approval workflow using:

- Planner (LLM)
- Retrieval-Augmented Generation (RAG)
- Business Validation Tools
- Structured AI Decision Making
"""
)


# ==========================================================
# Session State Initialization
# ==========================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "evaluated_claim_id" not in st.session_state:
    st.session_state.evaluated_claim_id = None

if "current_claim_id" not in st.session_state:
    st.session_state.current_claim_id = None


# ==========================================================
# Load Sample Claims
# ==========================================================

DATA_PATH = Path("data") / "sample_claims.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    sample_claims = json.load(f)["claims"]

claim_lookup = {
    claim["claim_id"]: claim
    for claim in sample_claims
}


# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("Claim Input")

claim_mode = st.sidebar.radio(
    "Choose Input Method",
    [
        "Sample Claim",
        "Upload JSON",
    ],
)

selected_claim = None


# ==========================================================
# Sample Claim Mode
# ==========================================================

if claim_mode == "Sample Claim":

    claim_options = ["Select a Claim"] + list(claim_lookup.keys())

    selected_option = st.sidebar.selectbox(
        "Select Claim",
        claim_options,
        index=0,
    )

    if selected_option != "Select a Claim":
        selected_claim = claim_lookup[selected_option]

        if (
            st.session_state.current_claim_id
            != selected_claim["claim_id"]
        ):

            st.session_state.current_claim_id = (
                selected_claim["claim_id"]
            )

            st.session_state.result = None
            st.session_state.evaluated_claim_id = None


# ==========================================================
# Upload JSON Mode
# ==========================================================

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload Claim JSON",
        type=["json"],
    )

    if uploaded_file is not None:

        selected_claim = json.load(uploaded_file)

        if (
            st.session_state.current_claim_id
            != selected_claim["claim_id"]
        ):

            st.session_state.current_claim_id = (
                selected_claim["claim_id"]
            )

            st.session_state.result = None
            st.session_state.evaluated_claim_id = None


# ==========================================================
# Agent
# ==========================================================

agent = ReimbursementAgent()


# ==========================================================
# Evaluate Button
# ==========================================================

run_clicked = st.sidebar.button(
    "Evaluate Claim",
    use_container_width=True,
)


# ==========================================================
# Claim Details
# (Shown immediately after selecting a claim)
# ==========================================================

if selected_claim is not None:

    st.subheader("Claim Details")

    left, right = st.columns(2)

    with left:

        st.metric(
            "Claim ID",
            selected_claim["claim_id"],
        )

        st.metric(
            "Employee",
            selected_claim["employee_name"],
        )

        st.metric(
            "Department",
            selected_claim["department"],
        )

    with right:

        st.metric(
            "Trip Start",
            selected_claim["travel_start_date"],
        )

        st.metric(
            "Trip End",
            selected_claim["travel_end_date"],
        )

        st.metric(
            "Submission",
            selected_claim["submission_date"],
        )

    st.divider()

    st.subheader("Expense Items")

    st.dataframe(
        selected_claim["expenses"],
        use_container_width=True,
    )


# ==========================================================
# Run Agent
# ==========================================================

if run_clicked:
    if selected_claim is None:
        st.warning("Please select or upload a claim first.")
    else:
        with st.spinner("Running AI Reimbursement Agent..."):
            result = agent.run_claim(selected_claim)
        st.session_state.result = result
        st.session_state.evaluated_claim_id = (
            selected_claim["claim_id"]
        )
        st.success("Evaluation completed.")


# ==========================================================
# Display Evaluation
# (Only after Evaluate button is clicked)
# ==========================================================

if st.session_state.result is not None:

    result = st.session_state.result
    planner_tools = result["selected_tools"]
    tool_results = result["tool_results"]
    decision = result["decision"]

    st.divider()

    # ======================================================
    # Planner
    # ======================================================

    st.subheader("🧠 Planner Decision")

    st.write(
        "The planner selected the following business tools before evaluating the claim."
    )

    planner_cols = st.columns(max(len(planner_tools), 1))

    for i, tool in enumerate(planner_tools):
        with planner_cols[i]:
            st.success(tool)

    st.divider()

    # ======================================================
    # Tool Results
    # ======================================================

    st.subheader("🛠 Tool Execution Results")

    if "policy" in tool_results:
        with st.expander(
            "📘 Retrieved Policy",
            expanded=False,
        ):
            st.markdown(tool_results["policy"])

    if "receipt_validation" in tool_results:
        with st.expander(
            "🧾 Receipt Validation",
            expanded=False,
        ):
            for receipt in tool_results["receipt_validation"]:
                st.json(receipt)

    if "limit_check" in tool_results:
        with st.expander(
            "💰 Reimbursement Limit Check",
            expanded=False,
        ):
            st.json(tool_results["limit_check"])

    if "duplicate_check" in tool_results:
        with st.expander(
            "🔁 Duplicate Receipt Check",
            expanded=False,
        ):
            for duplicate in tool_results["duplicate_check"]:
                st.json(duplicate)

    if "threshold_check" in tool_results:
        with st.expander(
            "👤 Approval Threshold",
            expanded=False,
        ):
            st.json(tool_results["threshold_check"])

    st.divider()

    # ======================================================
    # Final Decision
    # ======================================================

    st.subheader("✅ Final Decision")

    decision_value = decision["decision"]

    if decision_value == "Approve":
        st.success(f"🟢 {decision_value}")
    elif decision_value == "Partially Approve":
        st.warning(f"🟡 {decision_value}")
    elif decision_value == "Reject":
        st.error(f"🔴 {decision_value}")
    else:
        st.info(f"🟠 {decision_value}")

    st.divider()

    # ======================================================
    # Decision Summary
    # ======================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Approved Amount",
            f"₹ {decision['approved_amount']:,}",
        )

    with col2:
        st.metric(
            "Rejected Amount",
            f"₹ {decision['rejected_amount']:,}",
        )

    with col3:
        st.metric(
            "Confidence",
            f"{decision['confidence'] * 100:.1f}%",
        )

    st.divider()

    # ======================================================
    # Missing Documents
    # ======================================================

    st.subheader("📄 Missing Documents")

    missing_docs = decision.get(
        "missing_documents",
        [],
    )

    if missing_docs:
        for doc in missing_docs:
            st.write(f"• {doc}")
    else:
        st.success("No missing documents.")

    st.divider()

    # ======================================================
    # Policy References
    # ======================================================

    st.subheader("📚 Policy References")

    policy_refs = decision.get(
        "policy_references",
        [],
    )

    if policy_refs:
        for ref in policy_refs:
            st.write(f"• {ref}")
    else:
        st.info("No policy references returned.")

    st.divider()

    # ======================================================
    # AI Explanation
    # ======================================================

    st.subheader("💬 AI Explanation")

    explanation = decision.get(
        "explanation",
        "",
    )

    if explanation:
        st.info(explanation)
    else:
        st.warning("No explanation returned by the AI.")

    st.divider()

    # ======================================================
    # Raw JSON
    # ======================================================

    with st.expander(
        "📦 Full JSON Output",
        expanded=False,
    ):
        st.json(result)

    # ======================================================
    # Download Decision
    # ======================================================

    st.download_button(
        label="⬇ Download Decision JSON",
        data=json.dumps(result, indent=4),
        file_name=f"{st.session_state.evaluated_claim_id}_decision.json",
        mime="application/json",
    )


# ==========================================================
# Footer
# ==========================================================

st.divider()

# st.caption(
#     "Built for the HCLTech GenAI Developer Assignment | "
#     "Planner → RAG → Business Tools → LLM Decision"
# )