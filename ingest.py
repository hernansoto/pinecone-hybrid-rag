from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from config import Settings
from documents import load_and_split
from pinecone_store import (
    build_vectorstore,
    ensure_index,
    namespace_vector_count,
)


def corpus_fingerprint(
    settings: Settings,
    chunks,
) -> str:
    hasher = hashlib.sha256()

    hasher.update(settings.embedding_provider.encode())
    hasher.update(settings.embedding_model.encode())
    hasher.update(str(settings.embedding_dimension).encode())
    hasher.update(str(settings.chunk_size).encode())
    hasher.update(str(settings.chunk_overlap).encode())

    for chunk in chunks:
        hasher.update(chunk.metadata["chunk_id"].encode())
        hasher.update(chunk.page_content.encode("utf-8"))

    return hasher.hexdigest()


def read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(
    settings: Settings,
    fingerprint: str,
    chunk_count: int,
) -> None:
    settings.manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "fingerprint": fingerprint,
        "index_name": settings.index_name,
        "namespace": settings.namespace,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunk_count": chunk_count,
    }

    settings.manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ingest(force: bool = False) -> None:
    settings = Settings.from_env()
    pc = ensure_index(settings)

    chunks = load_and_split(settings)
    fingerprint = corpus_fingerprint(settings, chunks)

    current_count = namespace_vector_count(settings, pc)
    manifest = read_manifest(settings.manifest_path)

    if (
        not force
        and current_count > 0
        and manifest
        and manifest.get("fingerprint") == fingerprint
        and manifest.get("index_name") == settings.index_name
        and manifest.get("namespace") == settings.namespace
    ):
        print(
            f"Namespace '{settings.namespace}' ya contiene "
            f"{current_count} vectores y el corpus no cambió."
        )
        print("Se omite la reingesta.")
        return

    index = pc.Index(settings.index_name)

    if force and current_count > 0:
        print(
            f"Eliminando namespace '{settings.namespace}' "
            "antes de reindexar..."
        )
        index.delete(
            delete_all=True,
            namespace=settings.namespace,
        )

    if (
        not force
        and current_count > 0
        and (
            manifest is None
            or manifest.get("fingerprint") != fingerprint
        )
    ):
        raise RuntimeError(
            "Pinecone ya contiene vectores pero el corpus/configuración "
            "local cambió. Ejecuta `python ingest.py --force` para evitar "
            "mezclar chunks incompatibles."
        )

    print(f"Chunks a ingerir: {len(chunks)}")
    print(
        "Embedding: "
        f"{settings.embedding_provider}/"
        f"{settings.embedding_model} "
        f"({settings.embedding_dimension}d)"
    )
    print(f"Namespace: {settings.namespace}")

    vectorstore = build_vectorstore(settings, pc)

    ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    vectorstore.add_documents(
        documents=chunks,
        ids=ids,
    )

    write_manifest(
        settings,
        fingerprint,
        len(chunks),
    )

    print(
        f"Ingesta completada en Pinecone: "
        f"{settings.index_name}/{settings.namespace}"
    )
    print(
        "Cada vector contiene metadata de source, page, category, "
        "doc_id, title, chunk_id y el texto bajo la clave 'text'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingesta del dataset técnico en Pinecone Serverless."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recrea el contenido del namespace.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(force=args.force)
