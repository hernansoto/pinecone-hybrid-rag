from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    """
    Devuelve el mismo embedding para ingesta y consulta.

    El índice de Pinecone debe usar exactamente la misma dimensión.
    """
    if settings.embedding_provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
            output_dimensionality=settings.embedding_dimension,
        )

    if settings.embedding_provider == "openai":
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            dimensions=settings.embedding_dimension,
        )

    raise ValueError(
        f"Proveedor de embeddings no soportado: "
        f"{settings.embedding_provider}"
    )
