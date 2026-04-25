import re

class Normalizer:
    def __init__(self, mode = None):
        self.mode = mode
        
        if mode == "stemming":
            from nltk.stem import SnowballStemmer
            self.stemmer = SnowballStemmer("english")

        elif mode == "lemmatization":
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if self.mode == "stemming":
            tokens = text.split()
            tokens = [self.stemmer.stem(t) for t in tokens]
            return " ".join(tokens)

        if self.mode == "lemmatization":
            doc = self.nlp(text)
            return " ".join([t.lemma_ for t in doc if not t.is_space])

        return text