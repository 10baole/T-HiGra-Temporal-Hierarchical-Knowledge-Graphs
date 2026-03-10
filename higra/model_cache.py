import spacy
from fastcoref import spacy_component
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from higra_agent.logger import logger


class ModelCache:
    _spacy_model = None
    _coref_model = None
    _embedding_model = None
    _tokenizer = None
    _sentenizer = None

    @classmethod
    def get_spacy_model(cls):
        if cls._spacy_model is None:
            cls._spacy_model = spacy.load("en_core_web_lg")
            logger.info(">>> Loaded SpaCy Model")
        return cls._spacy_model

    @classmethod
    def get_coref_model(cls):
        if cls._coref_model is None:
            coref_model = spacy.load("en_core_web_lg")
            coref_model.add_pipe("fastcoref")
            cls._coref_model = coref_model
            logger.info(">>> Loaded Coref Model")
        return cls._coref_model

    @classmethod
    def get_embedding_model(cls):
        if cls._embedding_model is None:
            cls._embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
            logger.info(">>> Loaded Embedding Model")
        return cls._embedding_model

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            logger.info(">>> Loaded Tokenizer")
        return cls._tokenizer

    @classmethod
    def get_sentenizer(cls):
        if cls._sentenizer is None:
            sentenizer = spacy.blank("en")
            sentenizer.add_pipe("sentencizer")
            cls._sentenizer = sentenizer
            logger.info(">>> Loaded Sentenizer")
        return cls._sentenizer

    @classmethod
    def preload_all(cls):
        """Optional method to preload all models (e.g., for CLI or notebook environments)."""
        cls.get_spacy_model()
        # cls.get_coref_model()
        cls.get_embedding_model()
        cls.get_tokenizer()
        cls.get_sentenizer()
