from typing import Set, Dict, List
import re


CHUNK_ID_PATTERN = re.compile(
    r"\(paper_\d+__chunk_\d+\)"
)

def remove_citations_inside_text(answer):
    """
    Remove parenthetical citation-like patterns from text.
    """
    cleaned = []

    for sentence in answer:
        text = sentence['text']
        text_no_citations = CHUNK_ID_PATTERN.sub("", text).strip()
        text_no_spaces_before_punctuation = re.sub(r"\s+([.,])", r"\1", text_no_citations)

        cleaned.append({
            "text": text_no_spaces_before_punctuation,
            "citations": sentence['citations']
        })

    return cleaned



def format_author_year(authors, year):
    """
    Format authors into a human-readable citation label.
    """
    if not authors:
        return f"Unknown ({year})"

    # Normalize separators
    normalized = authors.replace(" and ", ",")
    parts = [a.strip() for a in normalized.split(",") if a.strip()]

    first_author = parts[0]

    if len(parts) > 1:
        return f"{first_author} et al. ({year})"
    else:
        return f"{first_author} ({year})"



def resolve_answer_citations(answer, source_lookup):
    """
    Replace chunk_id citations with human-readable labels,
    deduplicated at the paper level.
    Returns:
        resolved_answer: list of sentences with [Author, Year] citations
    """

    resolved_answer = []

    for sentence in answer:
        seen_papers = set()
        labels = []

        for cid in sentence.get("citations", []):
            source = source_lookup.get(cid)
            if not source:
                continue

            paper_id = source.get("paper_id")
            if not paper_id or paper_id in seen_papers:
                continue

            labels.append(
                format_author_year(
                    source.get("authors"),
                    source.get("year")
                )
            )
            seen_papers.add(paper_id)

        resolved_answer.append({
            "text": sentence["text"],
            "citations": labels
        })

    return resolved_answer



def build_source_entry(paper_id, source_lookup):
    for src in source_lookup.values():
        if src["paper_id"] == paper_id:
            return {
                "paper_id": paper_id,
                "title": src.get("title"),
                "authors": src.get("authors"),
                "year": src.get("year"),
                "journal": src.get("journal")
            }