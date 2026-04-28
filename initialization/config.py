class InitializationConfig:
    def __init__(
        self,
        chunk_size,
        overlap,
        embedding,
        normalization_mode,
        reranking,
        auto_build
    ):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        if embedding not in ["hf", "openai"]:
            raise ValueError("Embedding must be 'hf' or 'openai'")

        if normalization_mode is not None and normalization_mode not in ["stemming", "lemmatization"]:
            raise ValueError("Embedding must be 'stemming' or 'lemmatization'")
 
        if reranking not in ["flashrank", "cross_encoding"]:
            raise ValueError("Reranking must be 'flashrank' or 'cross_encoding'")
            
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding = embedding
        self.normalization_mode = normalization_mode
        self.reranking = reranking
        self.auto_build = auto_build


DEFAULT_CONFIG = InitializationConfig(
    chunk_size=500,
    overlap=100,
    embedding="openai",
    normalization_mode="stemming",
    reranking="flashrank",
    auto_build=False
)
