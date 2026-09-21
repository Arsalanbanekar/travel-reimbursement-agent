"""
Loads the travel policy and splits it into section-aware chunks.

Chunking follows the document's own numbering rather than a fixed character
count, so every chunk is exactly one policy clause (4.1, 11.2, ...) and carries
its section id in metadata. That id is what makes a citation verifiable: the
agent can only cite sections that actually exist and were actually retrieved.
"""

import re

from langchain_core.documents import Document

from agent.config import POLICY_PATH

# "## 4. Hotel Policy"
SECTION_HEADING = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$", re.MULTILINE)

# "4.1. **Domestic hotel limit:** Maximum ..."
CLAUSE_START = re.compile(r"^(\d+\.\d+)\.\s+", re.MULTILINE)

POLICY_DOC_ID = "FIN-TRV-2025-01"


def load_policy() -> str:
    """Load the markdown travel policy."""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Policy not found: {POLICY_PATH}")

    return POLICY_PATH.read_text(encoding="utf-8")


def _split_into_clauses(section_number: str, body: str) -> list[tuple[str, str]]:
    """
    Split a section body into (section_id, text) pairs.

    Sections that use numbered clauses yield one entry per clause. Sections
    built from a table or bullet list (3 and 13) have no clause numbering, so
    they stay whole under their section number.
    """

    starts = list(CLAUSE_START.finditer(body))

    if not starts:
        return [(section_number, body.strip())]

    clauses: list[tuple[str, str]] = []

    # Any preamble before the first numbered clause belongs to the section.
    preamble = body[: starts[0].start()].strip()
    if preamble:
        clauses.append((section_number, preamble))

    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(body)
        clauses.append((match.group(1), body[match.start() : end].strip()))

    return clauses


def parse_policy_sections() -> list[Document]:
    """Parse the policy into one Document per clause, with metadata."""

    text = load_policy()
    headings = list(SECTION_HEADING.finditer(text))

    documents: list[Document] = []

    for i, heading in enumerate(headings):
        number, title = heading.group(1), heading.group(2)
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[heading.end() : end]

        for section_id, content in _split_into_clauses(number, body):
            if not content:
                continue

            # The heading is prepended so each chunk stands on its own once
            # retrieved out of context.
            documents.append(
                Document(
                    page_content=f"Section {section_id} — {title}\n\n{content}",
                    metadata={
                        "section": section_id,
                        "title": title,
                        "policy_id": POLICY_DOC_ID,
                        "source": POLICY_PATH.name,
                    },
                )
            )

    return documents


def chunk_policy() -> list[Document]:
    """Public entry point used when building the vector store."""

    return parse_policy_sections()


def known_sections() -> set[str]:
    """
    Every section id that exists in the policy.

    Used to reject citations the model invented.
    """

    return {doc.metadata["section"] for doc in parse_policy_sections()}


if __name__ == "__main__":

    docs = chunk_policy()

    print(f"{len(docs)} chunks\n")

    for doc in docs:
        preview = doc.page_content.split("\n\n", 1)[-1].replace("\n", " ")[:80]
        print(f"  [{doc.metadata['section']:>4}] {doc.metadata['title'][:22]:<24} {preview}")
