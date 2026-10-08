import asyncio

from rag_system import RAGSystem


async def main() -> None:
    rag = RAGSystem()

    question = (
        "¿Qué hace model_dump en Pydantic y en qué se diferencia "
        "de model_dump_json?"
    )

    print(f"Pregunta: {question}\n")

    documents = await rag.retrieve(question)

    print("Top-5 híbrido:")
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata
        print(
            f"{rank}. "
            f"{metadata.get('doc_id')} | "
            f"{metadata.get('source')} | "
            f"página {metadata.get('page')} | "
            f"{metadata.get('category')}"
        )

    print("\nRespuesta RAG:")
    answer = await rag.answer(question)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
