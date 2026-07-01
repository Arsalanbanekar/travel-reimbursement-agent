from agent.tools import approval_threshold_checker_tool

print(
    approval_threshold_checker_tool.invoke(
        {"total_amount": 18280}
    )
)

print(
    approval_threshold_checker_tool.invoke(
        {"total_amount": 150000}
    )
)

print(
    approval_threshold_checker_tool.invoke(
        {"total_amount": 300000}
    )
)