# # """
# # Agent factory and runtime configuration.

# # Future purpose:
# # - Initialize the Groq LLM (llama-3.3-70b-versatile) with LangChain bindings.
# # - Bind all LangChain tools from tools.py to the model.
# # - Compile and expose the LangGraph workflow from workflow.py.
# # - Provide a single `run_claim(claim: dict) -> ReimbursementDecision` interface
# #   for the Streamlit UI and test harness.

# # Assignment mapping:
# # - Practical use of GenAI and Agentic AI: tool calling and workflow control (Section 6).
# # - Developer judgement: keep agent setup simple and readable (Section 6).
# # """





# """
# Main LangChain Agent for the Travel Reimbursement Approval Agent.
# """

# import os

# from dotenv import load_dotenv
# from langchain import hub
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_groq import ChatGroq

# from agent.prompts import SYSTEM_PROMPT
# from agent.tools import (
#     approval_threshold_checker_tool,
#     duplicate_receipt_checker_tool,
#     policy_lookup_tool,
#     receipt_validation_tool,
#     reimbursement_limit_checker_tool,
# )

# load_dotenv()


# llm = ChatGroq(
#     model=os.getenv("MODEL_NAME"),
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0,
# )


# tools = [
#     policy_lookup_tool,
#     receipt_validation_tool,
#     reimbursement_limit_checker_tool,
#     duplicate_receipt_checker_tool,
#     approval_threshold_checker_tool,
# ]


# prompt = hub.pull("hwchase17/openai-tools-agent")

# prompt.messages[0].prompt.template = SYSTEM_PROMPT


# agent = create_tool_calling_agent(
#     llm=llm,
#     tools=tools,
#     prompt=prompt,
# )


# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
# )


# def run_claim(claim: dict):
#     """
#     Run the reimbursement agent.
#     """

#     response = agent_executor.invoke(
#         {
#             "input": f"""
# Evaluate the following reimbursement claim.

# Claim:

# {claim}

# Return the final reimbursement decision.
# """
#         }
#     )

#     return response["output"]


# if __name__ == "__main__":

#     sample_claim = {
#         "claim_id": "CLM-001",
#         "employee_name": "Priya Sharma",
#         "expenses": [
#             {
#                 "category": "hotel",
#                 "amount": 4200,
#                 "receipt_id": "RCP-001",
#             }
#         ],
#     }

#     result = run_claim(sample_claim)

#     print("\n")
#     print(result)







# import json
# import os
# from dotenv import load_dotenv
# from groq import Groq

# from agent.prompts import SYSTEM_PROMPT, FINAL_DECISION_PROMPT

# from agent.tools import (
#     policy_lookup_tool,
#     receipt_validation_tool,
#     reimbursement_limit_checker_tool,
#     duplicate_receipt_checker_tool,
#     approval_threshold_checker_tool,
# )

# load_dotenv()

# client = Groq(
#     api_key=os.getenv("GROQ_API_KEY")
# )

# MODEL = "llama-3.3-70b-versatile"


# class ReimbursementAgent:

#     def __init__(self):
#         pass

#     def plan_tools(self, claim: dict):

#         response = client.chat.completions.create(
#             model=MODEL,
#             temperature=0,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": SYSTEM_PROMPT
#                 },
#                 {
#                     "role": "user",
#                     "content": json.dumps(claim, indent=2)
#                 }
#             ]
#         )

#         content = response.choices[0].message.content

#         try:
#             plan = json.loads(content)
#             return plan.get("tools", [])
#         except Exception:
#             return [
#                 "policy_lookup",
#                 "receipt_validation",
#                 "reimbursement_limit_checker",
#                 "duplicate_receipt_checker",
#                 "approval_threshold_checker"
#             ]

#     def execute_tools(self, claim: dict, tools: list):

#         results = {}

#         if "policy_lookup" in tools:

#             results["policy"] = policy_lookup_tool.invoke(
#     {
#         "query": "Travel reimbursement policy"
#     }
# )

#         if "receipt_validation" in tools:

#             receipt_results = []

#             for expense in claim["expenses"]:

#                 receipt_results.append(

#                     receipt_validation_tool.invoke(
#     {
#         "receipt_id": expense["receipt_id"]
#     }
# )

#                 )

#             results["receipt_validation"] = receipt_results

#         if "reimbursement_limit_checker" in tools:

#             results["limit_check"] = reimbursement_limit_checker_tool.invoke(
#     {
#         "expenses": claim["expenses"]
#     }
# )

#         if "duplicate_receipt_checker" in tools:

#             duplicate_results = []

#             for expense in claim["expenses"]:

#                 duplicate_results.append(

#                     duplicate_receipt_checker_tool.invoke(
#     {
#         "receipt_id": expense["receipt_id"]
#     }
# )

#                 )

#             results["duplicate_check"] = duplicate_results

#         if "approval_threshold_checker" in tools:

#             total = sum(
#                 expense["amount"]
#                 for expense in claim["expenses"]
#             )

#            approval_threshold_checker_tool.invoke(
#     {
#         "total_amount": total
#     }
# )

#         return results

#     def make_decision(self, claim: dict, tool_results: dict):

#         prompt = f"""
# Claim:

# {json.dumps(claim, indent=2)}

# Tool Results:

# {json.dumps(tool_results, indent=2)}

# Return only valid JSON.
# """

#         response = client.chat.completions.create(
#             model=MODEL,
#             temperature=0,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": FINAL_DECISION_PROMPT
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ]
#         )

#         return response.choices[0].message.content

#     def run_claim(self, claim: dict):

#         selected_tools = self.plan_tools(claim)

#         tool_results = self.execute_tools(
#             claim,
#             selected_tools
#         )

#         final_response = self.make_decision(
#             claim,
#             tool_results
#         )

#         return {
#             "selected_tools": selected_tools,
#             "tool_results": tool_results,
#             "decision": final_response
#         }


# if __name__ == "__main__":

#     with open("data/sample_claims.json", "r") as f:
#         claims = json.load(f)["claims"]

#     agent = ReimbursementAgent()

#     result = agent.run_claim(claims[0])

#     print(json.dumps(result, indent=2))







# import json
# import os
# from dotenv import load_dotenv
# from groq import Groq

# from agent.prompts import SYSTEM_PROMPT, FINAL_DECISION_PROMPT

# from agent.tools import (
#     policy_lookup_tool,
#     receipt_validation_tool,
#     reimbursement_limit_checker_tool,
#     duplicate_receipt_checker_tool,
#     approval_threshold_checker_tool,
# )

# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# MODEL = "llama-3.3-70b-versatile"


# class ReimbursementAgent:

#     def __init__(self):
#         pass

#     def plan_tools(self, claim: dict):

#         response = client.chat.completions.create(
#             model=MODEL,
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": json.dumps(claim, indent=2)},
#             ],
#         )

#         content = response.choices[0].message.content

#         try:
#             plan = json.loads(content)
#             return plan.get("tools", [])
#         except Exception:
#             return [
#                 "policy_lookup",
#                 "receipt_validation",
#                 "reimbursement_limit_checker",
#                 "duplicate_receipt_checker",
#                 "approval_threshold_checker",
#             ]

#     def execute_tools(self, claim: dict, tools: list):

#         results = {}

#         if "policy_lookup" in tools:

#             results["policy"] = policy_lookup_tool.invoke(
#                 {"query": "Travel reimbursement policy"}
#             )

#         if "receipt_validation" in tools:

#             receipt_results = []

#             for expense in claim["expenses"]:

#                 receipt_results.append(
#                     receipt_validation_tool.invoke(
#                         {"receipt_id": expense["receipt_id"]}
#                     )
#                 )

#             results["receipt_validation"] = receipt_results

#         if "reimbursement_limit_checker" in tools:

#             results["limit_check"] = reimbursement_limit_checker_tool.invoke(
#                 {"expenses": claim["expenses"]}
#             )

#         if "duplicate_receipt_checker" in tools:

#             duplicate_results = []

#             for expense in claim["expenses"]:

#                 duplicate_results.append(
#                   duplicate_receipt_checker_tool.invoke(
#                   {
#                         "receipt_id": expense["receipt_id"],
#                         "current_claim_id": claim["claim_id"]
#     }
# )

#             results["duplicate_check"] = duplicate_results

#         if "approval_threshold_checker" in tools:

#             total = sum(expense["amount"] for expense in claim["expenses"])

#             results["threshold_check"] = approval_threshold_checker_tool.invoke(
#                 {"total_amount": total}
#             )

#         return results

#     def make_decision(self, claim: dict, tool_results: dict):

#         prompt = f"""
# Claim:

# {json.dumps(claim, indent=2)}

# Tool Results:

# {json.dumps(tool_results, indent=2)}

# Return only valid JSON.
# """

#         response = client.chat.completions.create(
#             model=MODEL,
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": FINAL_DECISION_PROMPT},
#                 {"role": "user", "content": prompt},
#             ],
#         )

#         return response.choices[0].message.content

#     def run_claim(self, claim: dict):

#         selected_tools = self.plan_tools(claim)

#         tool_results = self.execute_tools(claim, selected_tools)

#         final_response = self.make_decision(claim, tool_results)

#         return {
#             "selected_tools": selected_tools,
#             "tool_results": tool_results,
#             "decision": final_response,
#         }


# if __name__ == "__main__":

#     with open("data/sample_claims.json", "r") as f:
#         claims = json.load(f)["claims"]

#     agent = ReimbursementAgent()

#     result = agent.run_claim(claims[0])

#     print(json.dumps(result, indent=2))







# import json
# import os
# from dotenv import load_dotenv
# from groq import Groq

# from agent.prompts import SYSTEM_PROMPT, FINAL_DECISION_PROMPT

# from agent.tools import (
#     policy_lookup_tool,
#     receipt_validation_tool,
#     reimbursement_limit_checker_tool,
#     duplicate_receipt_checker_tool,
#     approval_threshold_checker_tool,
# )

# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# MODEL = "llama-3.3-70b-versatile"


# class ReimbursementAgent:

#     def __init__(self):
#         pass

#     # def plan_tools(self, claim: dict):

#     #     response = client.chat.completions.create(
#     #         model=MODEL,
#     #         temperature=0,
#     #         messages=[
#     #             {"role": "system", "content": SYSTEM_PROMPT},
#     #             # {"role": "user", "content": json.dumps(claim, indent=2)},
#     #             planner_prompt = f"""
#     #             Decide which tools are required to evaluate the following reimbursement claim.

#     #             Select ONLY the minimum number of tools required.

#     #             Claim:

#     #             {json.dumps(claim, indent=2)}

#     #             Return ONLY valid JSON.
#     #             """

#     #             response = client.chat.completions.create(
#     #                model=MODEL,
#     #                temperature=0,
#     #                messages=[
#     #                   {"role": "system", "content": SYSTEM_PROMPT},
#     #                   {"role": "user", "content": planner_prompt},
#     #                 ],
#     #             )
#     #         ],
#     #     )

#     #     content = response.choices[0].message.content

#     #     try:
#     #         plan = json.loads(content)
#     #         return plan.get("tools", [])
#     #     except Exception:
#     #         return [
#     #             "policy_lookup",
#     #             "receipt_validation",
#     #             "reimbursement_limit_checker",
#     #             "duplicate_receipt_checker",
#     #             "approval_threshold_checker",
#     #         ]
    
#     def plan_tools(self, claim: dict):

#     planner_prompt = f"""
# Decide which tools are required to evaluate the following reimbursement claim.

# Select ONLY the minimum number of tools required.

# Claim:

# {json.dumps(claim, indent=2)}

# Return ONLY valid JSON.
# """

#     response = client.chat.completions.create(
#         model=MODEL,
#         temperature=0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": SYSTEM_PROMPT,
#             },
#             {
#                 "role": "user",
#                 "content": planner_prompt,
#             },
#         ],
#     )

#     content = response.choices[0].message.content

#     try:
#         plan = json.loads(content)
#         return plan.get("tools", [])
#     except Exception:
#         return [
#             "policy_lookup",
#             "receipt_validation",
#             "reimbursement_limit_checker",
#             "duplicate_receipt_checker",
#             "approval_threshold_checker",
#         ]

#     def execute_tools(self, claim: dict, tools: list):

#         results = {}

#         if "policy_lookup" in tools:

#             results["policy"] = policy_lookup_tool.invoke(
#                 {"query": "Travel reimbursement policy"}
#             )

#         if "receipt_validation" in tools:

#             receipt_results = []

#             for expense in claim["expenses"]:

#                 receipt_results.append(
#                     receipt_validation_tool.invoke(
#                         {"receipt_id": expense["receipt_id"]}
#                     )
#                 )

#             results["receipt_validation"] = receipt_results

#         if "reimbursement_limit_checker" in tools:

#             results["limit_check"] = reimbursement_limit_checker_tool.invoke(
#                 {"expenses": claim["expenses"]}
#             )

#         if "duplicate_receipt_checker" in tools:

#             duplicate_results = []

#             for expense in claim["expenses"]:

#                 duplicate_results.append(
#                     duplicate_receipt_checker_tool.invoke(
#                         {
#                             "receipt_id": expense["receipt_id"],
#                             "current_claim_id": claim["claim_id"],
#                         }
#                     )
#                 )

#             results["duplicate_check"] = duplicate_results

#         if "approval_threshold_checker" in tools:

#             total = sum(expense["amount"] for expense in claim["expenses"])

#             results["threshold_check"] = approval_threshold_checker_tool.invoke(
#                 {"total_amount": total}
#             )

#         return results

#     def make_decision(self, claim: dict, tool_results: dict):

#         prompt = f"""
# Claim:

# {json.dumps(claim, indent=2)}

# Tool Results:

# {json.dumps(tool_results, indent=2)}

# Return only valid JSON.
# """

#         response = client.chat.completions.create(
#             model=MODEL,
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": FINAL_DECISION_PROMPT},
#                 {"role": "user", "content": prompt},
#             ],
#         )

#         return response.choices[0].message.content

#     def run_claim(self, claim: dict):

#         selected_tools = self.plan_tools(claim)
#         print("\nPlanner selected tools:")
#         print(selected_tools)

#         tool_results = self.execute_tools(claim, selected_tools)

#         final_response = self.make_decision(claim, tool_results)

#         return {
#             "selected_tools": selected_tools,
#             "tool_results": tool_results,
#             "decision": final_response,
#         }


# if __name__ == "__main__":

#     with open("data/sample_claims.json", "r") as f:
#         claims = json.load(f)["claims"]

#     agent = ReimbursementAgent()
    
#     for claim in claims:

#         print("\n" + "=" * 80)
#         print(f"Processing Claim: {claim['claim_id']}")
#         print("=" * 80)

#         result = agent.run_claim(claim)

#         print(json.dumps(result, indent=2))

#     # result = agent.run_claim(claims[0])

#     # print(json.dumps(result, indent=2))






# import json
# import os
# from dotenv import load_dotenv
# from groq import Groq
# from models.schema import ReimbursementDecision

# from agent.prompts import SYSTEM_PROMPT, FINAL_DECISION_PROMPT

# from agent.tools import (
#     policy_lookup_tool,
#     receipt_validation_tool,
#     reimbursement_limit_checker_tool,
#     duplicate_receipt_checker_tool,
#     approval_threshold_checker_tool,
# )

# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# MODEL = "llama-3.3-70b-versatile"


# class ReimbursementAgent:

#     def __init__(self):
#         pass

#     def plan_tools(self, claim: dict):

#         planner_prompt = f"""
# Decide which tools are required to evaluate the following reimbursement claim.

# Select ONLY the minimum number of tools required.

# Claim:

# {json.dumps(claim, indent=2)}

# Return ONLY valid JSON.
# """

#         response = client.chat.completions.create(
#             model=MODEL,
#             temperature=0,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": SYSTEM_PROMPT,
#                 },
#                 {
#                     "role": "user",
#                     "content": planner_prompt,
#                 },
#             ],
#         )

#         content = response.choices[0].message.content

#         try:
#             plan = json.loads(content)
#             return plan.get("tools", [])
#         except Exception:
#             return [
#                 "policy_lookup",
#                 "receipt_validation",
#                 "reimbursement_limit_checker",
#                 "duplicate_receipt_checker",
#                 "approval_threshold_checker",
#             ]

#     def execute_tools(self, claim: dict, tools: list):

#         results = {}

#         if "policy_lookup" in tools:

#             results["policy"] = policy_lookup_tool.invoke(
#                 {"query": "Travel reimbursement policy"}
#             )

#         if "receipt_validation" in tools:

#             receipt_results = []

#             for expense in claim["expenses"]:

#                 receipt_results.append(
#                     receipt_validation_tool.invoke(
#                         {"receipt_id": expense["receipt_id"]}
#                     )
#                 )

#             results["receipt_validation"] = receipt_results

#         if "reimbursement_limit_checker" in tools:

#             results["limit_check"] = reimbursement_limit_checker_tool.invoke(
#                 {"expenses": claim["expenses"]}
#             )

#         if "duplicate_receipt_checker" in tools:

#             duplicate_results = []

#             for expense in claim["expenses"]:

#                 duplicate_results.append(
#                     duplicate_receipt_checker_tool.invoke(
#                         {
#                             "receipt_id": expense["receipt_id"],
#                             "current_claim_id": claim["claim_id"],
#                         }
#                     )
#                 )

#             results["duplicate_check"] = duplicate_results

#         if "approval_threshold_checker" in tools:

#             total = sum(expense["amount"] for expense in claim["expenses"])

#             results["threshold_check"] = approval_threshold_checker_tool.invoke(
#                 {"total_amount": total}
#             )

#         return results

# #     def make_decision(self, claim: dict, tool_results: dict):

# #         prompt = f"""
# # Claim:

# # {json.dumps(claim, indent=2)}

# # Tool Results:

# # {json.dumps(tool_results, indent=2)}

# # Return only valid JSON.
# # """

# #         response = client.chat.completions.create(
# #             model=MODEL,
# #             temperature=0,
# #             messages=[
# #                 {"role": "system", "content": FINAL_DECISION_PROMPT},
# #                 {"role": "user", "content": prompt},
# #             ],
# #         )

# #         return response.choices[0].message.content

# #     def run_claim(self, claim: dict):

# #         selected_tools = self.plan_tools(claim)
# #         print("\nPlanner selected tools:")
# #         print(selected_tools)

# #         tool_results = self.execute_tools(claim, selected_tools)

# #         final_response = self.make_decision(claim, tool_results)

# #         return {
# #             "selected_tools": selected_tools,
# #             "tool_results": tool_results,
# #             "decision": final_response,
# #         }


# def make_decision(self, claim: dict, tool_results: dict):

#     prompt = f"""
# Claim:

# {json.dumps(claim, indent=2)}

# Tool Results:

# {json.dumps(tool_results, indent=2)}

# Return only valid JSON.
# """

#     response = client.chat.completions.create(
#         model=MODEL,
#         temperature=0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": FINAL_DECISION_PROMPT,
#             },
#             {
#                 "role": "user",
#                 "content": prompt,
#             },
#         ],
#     )

#     response_text = response.choices[0].message.content

#     try:

#         parsed_response = json.loads(response_text)

#         validated_response = ReimbursementDecision.model_validate(
#             parsed_response
#         )

#         return validated_response.model_dump()

#     except Exception:

#         fallback = {
#             "decision": "Manual Review",
#             "approved_amount": 0,
#             "rejected_amount": 0,
#             "missing_documents": [],
#             "policy_references": [],
#             "confidence": 0.0,
#             "explanation": response_text,
#         }

#         validated_response = ReimbursementDecision.model_validate(
#             fallback
#         )

#         return validated_response.model_dump()


# if __name__ == "__main__":

#     with open("data/sample_claims.json", "r") as f:
#         claims = json.load(f)["claims"]

#     agent = ReimbursementAgent()

#     for claim in claims:

#         print("\n" + "=" * 80)
#         print(f"Processing Claim: {claim['claim_id']}")
#         print("=" * 80)

#         result = agent.run_claim(claim)

#         print(json.dumps(result, indent=2))









import json
import os
from dotenv import load_dotenv
from groq import Groq
from models.schema import ReimbursementDecision

from agent.prompts import SYSTEM_PROMPT, FINAL_DECISION_PROMPT

from agent.tools import (
    policy_lookup_tool,
    receipt_validation_tool,
    reimbursement_limit_checker_tool,
    duplicate_receipt_checker_tool,
    approval_threshold_checker_tool,
)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"


class ReimbursementAgent:

    def __init__(self):
        pass

    def plan_tools(self, claim: dict):

        planner_prompt = f"""
Decide which tools are required to evaluate the following reimbursement claim.

Select ONLY the minimum number of tools required.

Claim:

{json.dumps(claim, indent=2)}

Return ONLY valid JSON.
"""

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": planner_prompt,
                },
            ],
        )

        content = response.choices[0].message.content

        try:
            plan = json.loads(content)
            return plan.get("tools", [])
        except Exception:
            return [
                "policy_lookup",
                "receipt_validation",
                "reimbursement_limit_checker",
                "duplicate_receipt_checker",
                "approval_threshold_checker",
            ]

    def execute_tools(self, claim: dict, tools: list):

        results = {}

        if "policy_lookup" in tools:

            results["policy"] = policy_lookup_tool.invoke(
                {"query": "Travel reimbursement policy"}
            )

        if "receipt_validation" in tools:

            receipt_results = []

            for expense in claim["expenses"]:

                receipt_results.append(
                    receipt_validation_tool.invoke(
                        {"receipt_id": expense["receipt_id"]}
                    )
                )

            results["receipt_validation"] = receipt_results

        if "reimbursement_limit_checker" in tools:

            results["limit_check"] = reimbursement_limit_checker_tool.invoke(
                {"expenses": claim["expenses"]}
            )

        if "duplicate_receipt_checker" in tools:

            duplicate_results = []

            for expense in claim["expenses"]:

                duplicate_results.append(
                    duplicate_receipt_checker_tool.invoke(
                        {
                            "receipt_id": expense["receipt_id"],
                            "current_claim_id": claim["claim_id"],
                        }
                    )
                )

            results["duplicate_check"] = duplicate_results

        if "approval_threshold_checker" in tools:

            total = sum(expense["amount"] for expense in claim["expenses"])

            results["threshold_check"] = approval_threshold_checker_tool.invoke(
                {"total_amount": total}
            )

        return results

    def make_decision(self, claim: dict, tool_results: dict):

        prompt = f"""
Claim:

{json.dumps(claim, indent=2)}

Tool Results:

{json.dumps(tool_results, indent=2)}

Return only valid JSON.
"""

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": FINAL_DECISION_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        response_text = response.choices[0].message.content

        try:

            parsed_response = json.loads(response_text)

            validated_response = ReimbursementDecision.model_validate(
                parsed_response
            )

            # return validated_response.model_dump()
            return validated_response.model_dump(mode="json")

        except Exception:

            fallback = {
                "decision": "Manual Review",
                "approved_amount": 0,
                "rejected_amount": 0,
                "missing_documents": [],
                "policy_references": [],
                "confidence": 0.0,
                "explanation": response_text,
            }

            validated_response = ReimbursementDecision.model_validate(
                fallback
            )

            return validated_response.model_dump()

    def run_claim(self, claim: dict):

        selected_tools = self.plan_tools(claim)
        print("\nPlanner selected tools:")
        print(selected_tools)

        tool_results = self.execute_tools(claim, selected_tools)

        final_response = self.make_decision(claim, tool_results)

        return {
            "selected_tools": selected_tools,
            "tool_results": tool_results,
            "decision": final_response,
        }


if __name__ == "__main__":

    with open("data/sample_claims.json", "r") as f:
        claims = json.load(f)["claims"]

    agent = ReimbursementAgent()

    for claim in claims:

        print("\n" + "=" * 80)
        print(f"Processing Claim: {claim['claim_id']}")
        print("=" * 80)

        result = agent.run_claim(claim)

        print(json.dumps(result, indent=2))