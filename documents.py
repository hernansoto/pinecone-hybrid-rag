from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings


def load_documents(data_dir: Path) -> list[Document]:
    """
    Carga el dataset JSON.

    Cada archivo representa un documento técnico y contiene páginas.
    Esto permite persistir metadata avanzada:
    source, page, category, doc_id y title.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio de datos: {data_dir}"
        )

    documents: list[Document] = []

    for path in sorted(data_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))

        required = {"doc_id", "title", "category", "pages"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(
                f"{path.name}: faltan campos requeridos {sorted(missing)}"
            )

        for page in payload["pages"]:
            page_number = int(page["page"])
            content = str(page["content"]).strip()

            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_id": payload["doc_id"],
                        "title": payload["title"],
                        "category": payload["category"],
                        "source": path.name,
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise RuntimeError(
            f"No se encontraron documentos JSON válidos en {data_dir}."
        )

    return documents


def split_documents(
    documents: list[Document],
    settings: Settings,
) -> list[Document]:
    """
    Fragmenta cada página de forma independiente para conservar el número
    de página correcto en cada chunk.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []

    for document in documents:
        page_chunks = splitter.split_documents([document])

        for chunk_index, chunk in enumerate(page_chunks):
            chunk.metadata = {
                **chunk.metadata,
                "chunk_index": chunk_index,
                "chunk_id": (
                    f"{chunk.metadata['doc_id']}:"
                    f"p{chunk.metadata['page']}:"
                    f"c{chunk_index}"
                ),
            }
            chunks.append(chunk)

    return chunks


def load_and_split(settings: Settings) -> list[Document]:
    return split_documents(
        load_documents(settings.data_dir),
        settings,
    )
