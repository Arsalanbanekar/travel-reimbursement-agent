"""
Policy retrieval and citation checking.

Two complementary paths:

1. Exact lookup by section id. The rule engine already knows which clauses it
   applied, so their text is fetched directly — no embedding search needed,
   and no chance of missing the clause that actually decided the claim.

2. Semantic search. Used for the parts rules cannot judge — free-text expense
   descriptions, unusual categories, and the non-reimbursable list.

Anything the LLM cites is then checked against what was actually retrieved, so
an invented "Section 9.9" cannot reach the final output.
"""

import re
from functools import lru_cache

from langchain_core.documents import Document

from rag.loader import parse_policy_sections

SECTION_PATTERN = re.compile(r"\d+(?:\.\d+)?")


@lru_cache(maxsize=1)
def _sections_by_id() -> dict[str, Document]:
    return {doc.metadata["section"]: doc for doc in parse_policy_sections()}


def normalise_reference(reference: str) -> list[str]:
    """
    Pull section ids out of a free-form citation string.

    'Section 4.1, Section 4.2' -> ['4.1', '4.2']
    """

    return SECTION_PATTERN.findall(reference or "")


def verify_policy_references(
    references: list[str],
    allowed_sections: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Split citations into supported and unsupported.

    A citation is supported when the section exists in the policy and, if
    allowed_sections is given, was actually part of the retrieved context.
    """

    known = set(_sections_by_id())

    supported: list[str] = []
    unsupported: list[str] = []

    for reference in references:
        ids = normalise_reference(reference)

        if not ids:
            unsupported.append(reference)
            continue

        ok = all(
            section in known
            and (allowed_sections is None or section in allowed_sections)
            for section in ids
        )

        (supported if ok else unsupported).append(reference)

    return supported, unsupported


def merge_references(*groups: list[str]) -> list[str]:
    """
    Combine citation lists into one clean, ordered set.

    The rule engine emits grouped strings ("Section 4.1, Section 4.2") while
    the model emits bare ids ("4.1"). Both are normalised to one entry per
    section, ordered as they appear in the policy.
    """

    known = set(_sections_by_id())

    sections: set[str] = set()
    for group in groups:
        for reference in group or []:
            sections.update(s for s in normalise_reference(reference) if s in known)

    def order(section: str) -> tuple[int, int]:
        parts = section.split(".")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0

    return [f"Section {s}" for s in sorted(sections, key=order)]


def format_sections(documents: list[Document]) -> str:
    """Render retrieved clauses as text for the prompt."""

    if not documents:
        return "No relevant policy found."

    return "\n\n".join(doc.page_content for doc in documents)


class PolicyRetriever:
    """Retrieves policy clauses by id or by meaning."""

    def __init__(self):
        # The vector store is loaded on first semantic search, so exact lookups
        # and the rule engine keep working even if the index has not been built.
        self._vector_store = None

    @property
    def vector_store(self):
        if self._vector_store is None:
            from rag.vector_store import load_vector_store

            self._vector_store = load_vector_store()

        return self._vector_store

    def get_sections(self, section_ids: list[str]) -> list[Document]:
        """Fetch specific clauses by id, in policy order."""

        by_id = _sections_by_id()

        wanted: list[str] = []
        for reference in section_ids:
            for section in normalise_reference(reference):
                if section in by_id and section not in wanted:
                    wanted.append(section)

        return [by_id[section] for section in wanted]

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """Semantic search over the policy."""

        return self.vector_store.similarity_search(query, k=k)

    def retrieve_for_claim(
        self,
        rule_references: list[str],
        queries: list[str],
        k_per_query: int = 2,
    ) -> list[Document]:
        """
        Build the policy context for one claim.

        Starts from the clauses the rule engine actually applied, then adds
        semantic hits for anything the rules could not reason about. Results
        are deduplicated by section id.
        """

        documents = self.get_sections(rule_references)
        seen = {doc.metadata["section"] for doc in documents}

        for query in queries:
            try:
                hits = self.retrieve(query, k=k_per_query)
            except Exception:
                # A missing or unreadable index must not break evaluation;
                # the rule-engine clauses above are already enough to decide.
                break

            for hit in hits:
                section = hit.metadata.get("section")
                if section not in seen:
                    seen.add(section)
                    documents.append(hit)

        return documents


@lru_cache(maxsize=1)
def get_retriever() -> PolicyRetriever:
    """The shared retriever. Use this rather than constructing one per call."""

    return PolicyRetriever()


if __name__ == "__main__":

    retriever = get_retriever()

    print("-- exact lookup --")
    for doc in retriever.get_sections(["Section 4.1, Section 4.2"]):
        print(f"  [{doc.metadata['section']}] {doc.page_content[:70]}...")

    print("\n-- semantic search --")
    for doc in retriever.retrieve("what is the hotel limit per night?"):
        print(f"  [{doc.metadata['section']}] {doc.page_content[:70]}...")

    print("\n-- citation check --")
    supported, unsupported = verify_policy_references(
        ["Section 4.1", "Section 9.9", "made up"]
    )
    print(f"  supported   : {supported}")
    print(f"  unsupported : {unsupported}")
