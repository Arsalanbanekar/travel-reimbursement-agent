"""
Tests for policy parsing and citation checking.

These deliberately avoid semantic search so the suite stays fast and runs
without loading the embedding model.
"""

import pytest

from rag.loader import chunk_policy, known_sections
from rag.retriever import (
    PolicyRetriever,
    merge_references,
    normalise_reference,
    verify_policy_references,
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_every_chunk_carries_a_section_id():
    """Without a section id a chunk cannot support a verifiable citation."""

    for doc in chunk_policy():
        assert doc.metadata.get("section")
        assert doc.metadata.get("title")


def test_clauses_are_split_individually():
    """4.1 and 4.2 must be separate chunks, not one 500-character blob."""

    sections = known_sections()

    assert {"4.1", "4.2", "10.1", "11.1", "12.2", "14.7"} <= sections


def test_unnumbered_sections_stay_whole():
    """Section 13 is a bullet list with no clause numbers, so it stays intact."""

    sections = known_sections()

    assert "13" in sections
    assert "3" in sections


def test_chunk_is_self_contained():
    """Each chunk names its own section, so it makes sense out of context."""

    hotel = next(d for d in chunk_policy() if d.metadata["section"] == "4.1")

    assert hotel.page_content.startswith("Section 4.1")
    assert "Hotel Policy" in hotel.page_content
    assert "5,000" in hotel.page_content


# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reference,expected",
    [
        ("Section 4.1", ["4.1"]),
        ("Section 4.1, Section 4.2", ["4.1", "4.2"]),
        ("Section 13", ["13"]),
        ("no numbers here", []),
    ],
)
def test_normalise_reference(reference, expected):
    assert normalise_reference(reference) == expected


# ---------------------------------------------------------------------------
# Citation checking — the guard against invented policy
# ---------------------------------------------------------------------------


def test_real_sections_are_supported():
    supported, unsupported = verify_policy_references(
        ["Section 4.1", "Section 11.1"]
    )

    assert supported == ["Section 4.1", "Section 11.1"]
    assert unsupported == []


def test_invented_sections_are_rejected():
    """A hallucinated 'Section 9.9' must never pass as a citation."""

    supported, unsupported = verify_policy_references(
        ["Section 4.1", "Section 9.9", "as per company policy"]
    )

    assert supported == ["Section 4.1"]
    assert set(unsupported) == {"Section 9.9", "as per company policy"}


def test_citation_must_be_in_retrieved_context():
    """
    A real section that was never retrieved is still unsupported.

    This stops the model citing a clause it did not actually read.
    """

    supported, unsupported = verify_policy_references(
        ["Section 4.1", "Section 12.1"],
        allowed_sections={"4.1"},
    )

    assert supported == ["Section 4.1"]
    assert unsupported == ["Section 12.1"]


def test_multi_section_citation_requires_all_parts_valid():
    supported, unsupported = verify_policy_references(
        ["Section 4.1, Section 99.9"]
    )

    assert supported == []
    assert unsupported == ["Section 4.1, Section 99.9"]


# ---------------------------------------------------------------------------
# Reference merging
# ---------------------------------------------------------------------------


def test_merge_references_deduplicates_across_formats():
    """
    The rule engine emits 'Section 4.1, Section 4.2'; the model emits '4.1'.
    Both must collapse to one entry per section.
    """

    merged = merge_references(["Section 4.1, Section 4.2"], ["4.1", "Section 5.1"])

    assert merged == ["Section 4.1", "Section 4.2", "Section 5.1"]


def test_merge_references_orders_numerically():
    """13 must sort after 4.1, not before it as a string sort would."""

    merged = merge_references(["Section 13", "Section 4.1", "Section 11.2"])

    assert merged == ["Section 4.1", "Section 11.2", "Section 13"]


def test_merge_references_drops_invented_sections():
    merged = merge_references(["Section 4.1", "Section 99.9"])

    assert merged == ["Section 4.1"]


# ---------------------------------------------------------------------------
# Exact lookup (no embeddings involved)
# ---------------------------------------------------------------------------


def test_exact_section_lookup():
    docs = PolicyRetriever().get_sections(["Section 4.1, Section 4.2"])

    assert [d.metadata["section"] for d in docs] == ["4.1", "4.2"]


def test_exact_lookup_ignores_unknown_sections():
    docs = PolicyRetriever().get_sections(["Section 4.1", "Section 77.7"])

    assert [d.metadata["section"] for d in docs] == ["4.1"]


def test_exact_lookup_deduplicates():
    docs = PolicyRetriever().get_sections(["Section 4.1", "Section 4.1"])

    assert len(docs) == 1
