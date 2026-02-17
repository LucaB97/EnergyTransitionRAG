import json
from typing import Dict, Any
import streamlit as st


def response_to_json(
    response: Dict,
    include_debug: bool = False
) -> str:
    """
    Serialize full query response to JSON.
    """

    payload = dict(response)

    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    else:
        payload = dict(response)

    if not include_debug:
        payload.pop("debug", None)

    return json.dumps(payload, indent=2)



def response_to_markdown(response: Dict[str, Any]) -> str:
    """
    Serialize query response to a human-readable Markdown report.
    """
    lines = []

    # Title
    lines.append("# Synthesized Answer\n")

    for item in response.get("answer", []):
        text = item["text"]
        citations = item.get("citations", [])

        if citations:
            text += " " + " ".join(f"[{c}]" for c in citations)

        lines.append(f"- {text}")

    # Limitations
    if response.get("limitations"):
        lines.append("\n# Limitations\n")
        for lim in response["limitations"]:
            lines.append(f"- {lim}")

    # Sources
    if response.get("sources"):
        lines.append("\n# Sources\n")
        for src in response["sources"]:
            author_year = f"{src['authors']} ({src['year']})"
            journal = f", {src['journal']}" if src.get("journal") else ""
            lines.append(
                f"- **{src['title']}** — {author_year}{journal}"
            )

    # Evidence metrics
    metrics = response.get("evidence_metrics")
    if metrics:
        lines.append("\n## Evidence Metrics\n")
        for k, v in metrics.items():
            lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")

    # Evidence metrics
    confidence = response.get("confidence")
    if confidence:
        lines.append("\n## Confidence\n")
        for k, v in confidence.items():
            lines.append(f"{k.title()}: {v}")

    return "\n".join(lines)



def export_output(data):
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        with st.container():
            export_format = st.radio(
                "Format",
                ["JSON", "Markdown"],
                horizontal=True,
                label_visibility="collapsed"
            )
                
            if export_format == "JSON":
                has_debug = bool(data.get("debug"))
                include_debug = st.checkbox(
                    "Include debug evidence (chunks & papers)",
                    value=False,
                    disabled=not has_debug
                )
                data_export = response_to_json(data, include_debug)
                filename = "query_response.json"
                mime = "application/json"

            else:
                data_export = response_to_markdown(data)
                filename = "query_response.md"
                mime = "text/markdown"

            st.download_button(
                label="Download results",
                data=data_export,
                file_name=filename,
                mime=mime,
                use_container_width=True,
            )