import json
from utils.text_cleaning import clean_llm_output

class ResearchSynthesisEngine:
    """
    Synthesizes research answers using a structured JSON contract.
    """

    def __init__(self, llm_client, max_attempts=3):
        if not hasattr(llm_client, "generate"):
            raise ValueError("llm_client must implement a .generate(prompt) method")
        self.llm = llm_client
        self.max_attempts = max_attempts


    def build_prompt(self, question, chunks, prompt_template):
        sources_text = ""

        for c in chunks:
            raw_tags = [c.get("first_tag"), c.get("second_tag")]
            tags = [
                str(t) for t in raw_tags
                if t is not None and isinstance(t, str) and t.lower() != "nan"
            ]

            sources_text += f"""
SOURCE:
chunk_id: {c['chunk_id']}
tags: {", ".join(tags)}
content:
{c['text']}
"""

        return prompt_template.replace("{{SOURCES}}", sources_text)\
                              .replace("{{QUESTION}}", question)


    def _validate_output(self, raw_output: str, valid_chunk_ids: set):
        if raw_output.count("{") != raw_output.count("}"):
            raise ValueError("LLM output appears truncated")

        parsed = json.loads(raw_output)

        if "answer" not in parsed or not isinstance(parsed["answer"], list):
            raise ValueError("Invalid or missing 'answer'")

        if "limitations" not in parsed or not isinstance(parsed["limitations"], list):
            raise ValueError("Invalid or missing 'limitations'")

        for i, s in enumerate(parsed["answer"]):
            if "text" not in s or "citations" not in s:
                raise ValueError(f"Malformed answer item at index {i}")
            
            if not isinstance(s["citations"], list) or len(s["citations"]) == 0:
                raise ValueError(f"Empty citations at index {i}")
            
            for cid in s["citations"]:
                if not isinstance(cid, str) or not cid.strip():
                    raise ValueError(f"Invalid citation at index {i}: {cid!r}")
                if cid not in valid_chunk_ids:
                    raise ValueError(f"Citation '{cid}' not found in provided sources")

        return parsed


    def synthesize(self, question, chunks, prompt_template):
        
        valid_chunk_ids = set([c["chunk_id"] for c in chunks])
        base_prompt = self.build_prompt(question, chunks, prompt_template)
        last_error = None

        for attempt in range(1, self.max_attempts + 1):

            prompt = base_prompt

            if attempt > 1:
                prompt += (
                    "\n\nREMINDER: Return ONLY valid JSON, according to the provided schema."
                )

                if last_error:
                    prompt += f"\nPrevious error: {str(last_error)}"

            raw_output = self.llm.generate(prompt)
            output = clean_llm_output(raw_output)

            try:
                return self._validate_output(output, valid_chunk_ids)

            except Exception as e:
                last_error = e

        raise ValueError(last_error)