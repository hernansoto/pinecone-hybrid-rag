from __future__ import annotations

import time
from typing import Any

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import Settings
from embeddings import build_embeddings


def _value(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _index_names(pc: Pinecone) -> set[str]:
    indexes = pc.list_indexes()

    if hasattr(indexes, "names"):
        return set(indexes.names())

    names: set[str] = set()
    for item in indexes:
        name = _value(item, "name")
        if name:
            names.add(name)
    return names


def _is_ready(description: Any) -> bool:
    status = _value(description, "status", {})
    return bool(_value(status, "ready", False))


def build_pinecone_client(settings: Settings) -> Pinecone:
    return Pinecone(
        api_key=settings.pinecone_api_key.get_secret_value()
    )


def ensure_index(settings: Settings) -> Pinecone:
    """
    Verifica el índice Serverless y lo crea si no existe.

    Si ya existe, valida dimensión y métrica para evitar un mismatch silencioso.
    """
    pc = build_pinecone_client(settings)

    if settings.index_name not in _index_names(pc):
        print(
            f"Creando índice Serverless '{settings.index_name}' "
            f"({settings.embedding_dimension} dimensiones)..."
        )

        pc.create_index(
            name=settings.index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

        while True:
            description = pc.describe_index(settings.index_name)
            if _is_ready(description):
                break
            time.sleep(2)

        print("Índice creado y listo.")
    else:
        description = pc.describe_index(settings.index_name)

        dimension = int(_value(description, "dimension"))
        metric = str(_value(description, "metric"))

        if dimension != settings.embedding_dimension:
            raise RuntimeError(
                "Mismatch de dimensiones: "
                f"Pinecone={dimension}, configuración="
                f"{settings.embedding_dimension}. "
                "Usa otro INDEX_NAME o recrea el índice con la dimensión correcta."
            )

        if metric.lower() != "cosine":
            raise RuntimeError(
                f"El índice usa metric={metric}; este proyecto espera cosine."
            )

        print(
            f"Índice '{settings.index_name}' existente y compatible."
        )

    return pc


def build_vectorstore(
    settings: Settings,
    pc: Pinecone | None = None,
) -> PineconeVectorStore:
    pc = pc or ensure_index(settings)
    index = pc.Index(settings.index_name)

    return PineconeVectorStore(
        index=index,
        embedding=build_embeddings(settings),
        namespace=settings.namespace,
        text_key="text",
    )


def namespace_vector_count(
    settings: Settings,
    pc: Pinecone | None = None,
) -> int:
    pc = pc or build_pinecone_client(settings)
    index = pc.Index(settings.index_name)
    stats = index.describe_index_stats()

    namespaces = _value(stats, "namespaces", {}) or {}
    namespace_stats = _value(namespaces, settings.namespace)

    if namespace_stats is None:
        return 0

    return int(_value(namespace_stats, "vector_count", 0) or 0)
