import json
import random
from utils.text_cleaning import clean_llm_output

class EvidenceRelevanceJudge:
    def __init__(self, llm_client):
        if not hasattr(llm_client, "generate"):
            raise ValueError("llm_client must implement a .generate(prompt) method")
        
        if getattr(llm_client, "temperature", 0) > 0:
            raise ValueError("temperature must be zero")
        
        self.llm = llm_client


    def build_prompt(self, query: str, passage: str, template: str) -> str:
        return template.replace("{query}", query).replace("{passage}", passage)


    def clean_llm_output(self, output: str) -> str:
        output = output.strip()

        if output.startswith("```"):
            # remove first ```
            output = output.split("```", 1)[1]
            # remove optional "json"
            output = output.lstrip("json").strip()
            # remove last ```
            output = output.rsplit("```", 1)[0].strip()
        return output

    
    def validate_score(self, score):
        if not isinstance(score, int):
            raise ValueError("Output must be a INT")

        if not score in [0, 1, 2]:
            raise ValueError("Invalid score. Score must be 0, 1 or 2")

        return True


    def judge_relevance(self, query, passage, prompt_template):

        base_prompt = self.build_prompt(query, passage, prompt_template)
        # print(base_prompt)
        last_error = None

        for attempt in range(5):
            prompt = base_prompt

            if attempt > 0:
                prompt += (
                    "\n\nERROR in previous output:\n"
                    f"{last_error}\n\n"
                    "Fix the output.\n"
                    "You MUST:\n"
                    f"- Return EXACTLY 1 integer\n"
                    "- Each value must be 0, 1, or 2\n"
                    "- Do NOT include explanations or markdown\n"
                )

            output = self.llm.generate(prompt)
            output = self.clean_llm_output(output)

            try:
                score = json.loads(output)
                self.validate_score(score)
                return score

            except Exception as e:
                last_error = str(e)

        raise ValueError(f"Judge failed after retries: {last_error}")
    
    
    def build_prompt_unified(self, query: str, passages: list, template: str) -> str: 
        prompt = template.replace("{query}", query) 
        passage_block = "" 
        for i, p in enumerate(passages, start=1): 
            passage_block += f"Passage: {i}\n{p}\n\n" 
        prompt = prompt.replace("{passages}", passage_block.strip()) 
        return prompt
    

    def validate_scores_unified(self, scores, expected_len):
        if not isinstance(scores, list):
            raise ValueError("Output must be a JSON list")

        if len(scores) != expected_len:
            raise ValueError(
                f"Expected {expected_len} scores, got {len(scores)}"
            )

        if not all(s in [0, 1] for s in scores):
            raise ValueError("Invalid score detected in the output list")

        return True


    def judge_relevance_unified(self, query, shuffled_passages, prompt_template):

        # Shuffle passages to avoid bias
        # indexed_passages = list(enumerate(passages))
        # random.shuffle(indexed_passages)
        # shuffled_passages = [p for _, p in indexed_passages]

        base_prompt = self.build_prompt_unified(query, shuffled_passages, prompt_template)
        attempt = 1
        # last_error = None

        # for attempt in range(5):
        while True:
            prompt = base_prompt

            if attempt > 1:
                prompt += (
                    "\n\nERROR in previous output:\n"
                    f"{last_error}\n\n"
                    "Fix the output.\n"
                    "You MUST:\n"
                    f"- Return EXACTLY {len(shuffled_passages)} integers\n"
                    "- Use ONLY the allowed values specified in the scoring rule\n"
                    "- Return ONLY a JSON list\n"
                    "- Do NOT include explanations or markdown\n"
                )

            output = self.llm.generate(prompt)
            output = self.clean_llm_output(output)

            try:
                scores = json.loads(output)
                self.validate_scores_unified(scores, len(shuffled_passages))
                last_error = None
                # unshuffled_scores = [0] * len(scores)
                # for (orig_idx, _), score in zip(indexed_passages, scores):
                #     unshuffled_scores[orig_idx] = score
                # return unshuffled_scores
                return scores

            except Exception as e:
                last_error = str(e)
                attempt += 1

            if last_error is None:
                break

        # raise ValueError(f"Judge failed after retries: {last_error}")