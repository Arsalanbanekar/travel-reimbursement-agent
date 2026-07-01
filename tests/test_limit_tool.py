from agent.tools import reimbursement_limit_checker_tool

expenses = [
    {
        "category": "hotel",
        "amount": 5800
    },
    {
        "category": "meals",
        "amount": 700
    },
    {
        "category": "taxi",
        "amount": 450
    }
]

result = reimbursement_limit_checker_tool.invoke(
    {"expenses": expenses}
)

print(result)