from agent.tools import receipt_validation_tool

print(
    receipt_validation_tool.invoke(
        {"receipt_id": "RCP-001"}
    )
)

print(
    receipt_validation_tool.invoke(
        {"receipt_id": "RCP-008"}
    )
)

print(
    receipt_validation_tool.invoke(
        {"receipt_id": "RCP-999"}
    )
)