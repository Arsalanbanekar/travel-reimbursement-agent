from agent.tools import duplicate_receipt_checker_tool

print(
    duplicate_receipt_checker_tool.invoke(
        {"receipt_id": "RCP-003"}
    )
)

print(
    duplicate_receipt_checker_tool.invoke(
        {"receipt_id": "RCP-001"}
    )
)