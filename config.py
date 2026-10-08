from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, model_validator


class Settings(BaseModel):
    """Configuración validada del sistema RAG híbrido."""

    pinecone_api_key: SecretStr
    index_name: str = "preentrega4-hybrid-rag"
    namespace: str = "technical-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    embedding_provider: Literal["gemini", "openai"] = "gemini"
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    embedding_model: str = "gemini-embedding-2"
    embedding_dimension: int = Field(default=768, ge=128, le=3072)

    gemini_model: str = "gemini-3.8-flash"

    top_k: int = Field(default=5, ge=1, le=10)
    semantic_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.35, ge=0.0, le=1.0)

    chunk_size: int = Field(default=600, ge=500, le=1200)
    chunk_overlap: int = Field(default=80, ge=50, le=300)

    data_dir: Path = Path("./data")
    manifest_path: Path = Path("./.cache/ingestion_manifest.json")

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE.")

        if abs((self.semantic_weight + self.lexical_weight) - 1.0) > 1e-6:
            raise ValueError(
                "SEMANTIC_WEIGHT + LEXICAL_WEIGHT debe sumar 1.0."
            )

        if self.embedding_provider == "gemini":
            if self.gemini_api_key is None:
                raise ValueError(
                    "GEMINI_API_KEY es obligatoria cuando "
                    "EMBEDDING_PROVIDER=gemini."
                )

        if self.embedding_provider == "openai":
            if self.openai_api_key is None:
                raise ValueError(
                    "OPENAI_API_KEY es obligatoria cuando "
                    "EMBEDDING_PROVIDER=openai."
                )

        return self

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError(
                "Falta PINECONE_API_KEY. Copia .env.example como .env "
                "y configura tu credencial."
            )

        return cls(
            pinecone_api_key=pinecone_key,
            index_name=os.getenv(
                "INDEX_NAME",
                "preentrega4-hybrid-rag",
            ),
            namespace=os.getenv(
                "PINECONE_NAMESPACE",
                "technical-docs",
            ),
            pinecone_cloud=os.getenv(
                "PINECONE_CLOUD",
                "aws",
            ),
            pinecone_region=os.getenv(
                "PINECONE_REGION",
                "us-east-1",
            ),
            embedding_provider=os.getenv(
                "EMBEDDING_PROVIDER",
                "gemini",
            ).lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "gemini-embedding-2",
            ),
            embedding_dimension=int(
                os.getenv("EMBEDDING_DIMENSION", "768")
            ),
            gemini_model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash",
            ),
            top_k=int(os.getenv("TOP_K", "5")),
            semantic_weight=float(
                os.getenv("SEMANTIC_WEIGHT", "0.65")
            ),
            lexical_weight=float(
                os.getenv("LEXICAL_WEIGHT", "0.35")
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", "600")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            manifest_path=Path(
                os.getenv(
                    "MANIFEST_PATH",
                    "./.cache/ingestion_manifest.json",
                )
            ),
        )
