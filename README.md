# 💼 Travel Reimbursement Approval Agent

An AI-powered **Travel Reimbursement Approval System** developed as part of the **HCLTech GenAI Developer Assignment**.

The application leverages a **Planner → Retrieval-Augmented Generation (RAG) → Business Validation Tools → LLM Decision** workflow to evaluate employee travel reimbursement claims and generate structured reimbursement decisions grounded in company travel policies.

---

# 🚀 Features

- Accept reimbursement claims from:
  - Sample claims
  - Uploaded JSON files
- Planner LLM dynamically selects only the required business tools
- Retrieval-Augmented Generation (RAG) using FAISS and sentence-transformers
- Business validation tools for policy compliance
- Structured JSON output validated with Pydantic
- Interactive Streamlit dashboard
- Downloadable reimbursement decision reports
- Manual Review routing for uncertain or incomplete cases

---

# 🏗️ System Architecture

```
                    User
                      │
                      ▼
            Streamlit Web Interface
                      │
                      ▼
          Reimbursement Agent
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
      Planner LLM          Policy Retriever
   (Tool Selection)          (FAISS + RAG)
          │                       │
          └───────────┬───────────┘
                      ▼
              Business Tools
      • Policy Lookup
      • Receipt Validation
      • Reimbursement Limit Checker
      • Duplicate Receipt Checker
      • Approval Threshold Checker
                      │
                      ▼
             Final LLM Decision
                      │
                      ▼
       Structured JSON Response
                      │
                      ▼
          Streamlit Dashboard
```

---

# 📌 Key Highlights

- Planner-based AI workflow
- Retrieval-Augmented Generation (FAISS)
- Dynamic business tool selection
- Structured JSON decision output
- Pydantic schema validation
- Interactive Streamlit dashboard
- Downloadable reimbursement reports

---

# ⚙️ Technology Stack

| Layer | Technology |
|--------|------------|
| Programming Language | Python |
| UI | Streamlit |
| LLM | Groq (Llama 3.3 70B Versatile) |
| AI Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Validation | Pydantic |
| Environment | python-dotenv |
| Data Storage | JSON + Markdown |

---

# 📂 Project Structure

```
travel-reimbursement-agent/
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── prompts.py
│   └── tools.py
│
├── data/
│   ├── approval_matrix.json
│   ├── receipts.json
│   ├── reimbursement_limits.json
│   ├── sample_claims.json
│   └── travel_policy.md
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── models/
│   ├── __init__.py
│   └── schema.py
│
├── outputs/
│
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   ├── retriever.py
│   └── vector_store.py
│
├── screenshots/
│
├── tests/
│   ├── test_cases.py
│   ├── test_policy_tool.py
│   ├── test_receipt_tool.py
│   ├── test_limit_tool.py
│   ├── test_duplicate_tool.py
│   ├── test_approval_tool.py
│   └── verify_setup.py
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   ├── parser.py
│   └── validators.py
│
├── streamlit_app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

# 🤖 Workflow

## 1. Claim Intake

The application accepts reimbursement claims through:

- Sample reimbursement claims
- Uploaded JSON claim files

---

## 2. Planner (LLM)

The Planner LLM analyzes the reimbursement claim and dynamically determines which business tools are required.

Example planner output:

```json
{
  "tools": [
    "policy_lookup",
    "receipt_validation",
    "reimbursement_limit_checker"
  ]
}
```

Only the selected tools are executed.

---

## 3. Policy Retrieval (RAG)

The Policy Lookup Tool retrieves relevant reimbursement policy sections from a FAISS vector database created using sentence-transformer embeddings.

This ensures that reimbursement decisions are grounded in company travel policies.

---

## 4. Business Tool Execution

Depending on the planner's output, the agent may execute:

- 📘 Policy Lookup Tool
- 🧾 Receipt Validation Tool
- 💰 Reimbursement Limit Checker
- 🔁 Duplicate Receipt Checker
- 👤 Approval Threshold Checker

---

## 5. Final Decision

The LLM combines:

- Original reimbursement claim
- Retrieved policy
- Tool execution results

to generate one of the following decisions:

- Approve
- Partially Approve
- Reject
- Manual Review

---

## 6. Structured Output

The final decision is validated using a Pydantic schema.

Example:

```json
{
  "decision": "Approve",
  "approved_amount": 18280,
  "rejected_amount": 0,
  "missing_documents": [],
  "policy_references": [
    "Section 4.1"
  ],
  "confidence": 0.98,
  "explanation": "All expenses comply with the reimbursement policy."
}
```

---

# 🛠️ Business Validation Tools

## 📘 Policy Lookup Tool

Retrieves relevant reimbursement policy sections using semantic search over the FAISS vector database.

---

## 🧾 Receipt Validation Tool

Checks:

- Receipt exists
- Required attachment is present

---

## 💰 Reimbursement Limit Checker

Validates expenses against reimbursement limits and calculates:

- Approved amount
- Rejected amount

---

## 🔁 Duplicate Receipt Checker

Checks whether a receipt has already been used in another reimbursement claim.

---

## 👤 Approval Threshold Checker

Determines the required approver based on the reimbursement amount.

---

# 🖥️ Streamlit Dashboard

The application provides an interactive dashboard that displays:

- Claim Details
- Expense Items
- Planner Decision
- Tool Execution Results
- Retrieved Policy
- Final Decision
- Approved & Rejected Amounts
- Confidence Score
- Missing Documents
- Policy References
- AI Explanation
- Downloadable JSON Decision Report

---

# 📷 Screenshots

Screenshots demonstrating the application are available in the `screenshots/` directory.

Included screenshots:

- Home Screen
- Sample Claim Selection
- JSON Upload
- Planner Decision
- Tool Execution Results
- Retrieved Policy
- Final Decision
- Download JSON Output

---

# ▶️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd travel-reimbursement-agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=groq model name here
EMBEDDING_MODEL=embedding model name here
```

---

# ▶️ Run the Application

Launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

---

# 🧪 Sample Test Cases

The project includes five sample reimbursement claims demonstrating different reimbursement scenarios.

| Claim | Scenario | Decision |
|--------|----------|----------|
| CLM-001 | Valid reimbursement | Approve |
| CLM-002 | Expense exceeds reimbursement limit | Partially Approve |
| CLM-003 | Late claim submission | Reject |
| CLM-004 | Missing receipt attachment | Manual Review |
| CLM-005 | Duplicate receipt | Reject |

---

# ✅ Assignment Requirements Mapping

| Requirement | Status |
|-------------|--------|
| Claim Intake | ✅ |
| Context Grounding (RAG) | ✅ |
| Tool / Function Usage | ✅ |
| Planner-Based Tool Selection | ✅ |
| Structured JSON Output | ✅ |
| Manual Review Handling | ✅ |
| Streamlit UI | ✅ |

---


