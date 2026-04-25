class InitializationConfig:
    def __init__(
        self,
        chunk_size,
        overlap,
        embedding,
        auto_build,
        normalization_mode        
    ):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        if embedding not in ["hf", "openai"]:
            raise ValueError("Embedding must be 'hf' or 'openai'")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding = embedding
        self.auto_build = auto_build
        self.normalization_mode = normalization_mode


DEFAULT_CONFIG = InitializationConfig(
    chunk_size=500,
    overlap=100,
    embedding="openai",
    auto_build=True,
    normalization_mode=None
)