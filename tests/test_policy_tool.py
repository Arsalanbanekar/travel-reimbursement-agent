from agent.tools import policy_lookup_tool

print(
    policy_lookup_tool.invoke(
        {"query": "What is the hotel reimbursement limit?"}
    )
)