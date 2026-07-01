"""
Policy retriever — semantic search over embedded policy chunks.

Future purpose:
- Accept a query (claim category, expense type, or LLM-generated search string).
- Return top-k relevant policy sections with source metadata for citations.
- Used by the Policy Lookup Tool and displayed in the UI audit trail.

Assignment mapping:
- Context grounding before decision-making (Section 3).
- Policy references in structured output (Section 3).
- Audit trail: show retrieved context (Section 7, optional enhancement).
"""

"""
Retrieves relevant travel policy sections from the FAISS vector store.
"""

from rag.vector_store import load_vector_store


class PolicyRetriever:
    def __init__(self):
        self.vector_store = load_vector_store()

    def retrieve(self, query: str, k: int = 3):
        """
        Retrieve the top-k most relevant policy chunks.
        """
        docs = self.vector_store.similarity_search(query, k=k)
        return docs


if __name__ == "__main__":

    retriever = PolicyRetriever()

    query = "What is the maximum hotel reimbursement per night?"

    results = retriever.retrieve(query)

    print("\nRetrieved Policy Chunks\n")
    print("=" * 60)

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}\n")
        print(doc.page_content[:500])