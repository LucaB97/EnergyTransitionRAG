class QueryExpander:
    def __init__(self, llm_client):
        if not hasattr(llm_client, "generate"):
            raise ValueError("llm_client must implement a .generate(prompt) method")
        self.llm = llm_client

    def produce_expansion(self, question: str, prompt_template: str) -> bool:
        prompt = prompt_template.replace("{{QUESTION}}", question)
        output = self.llm.generate(prompt).strip().lower()
        return output