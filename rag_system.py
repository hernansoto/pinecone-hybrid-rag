from __future__ import annotations

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Settings
from documents import load_and_split
from pinecone_store import build_vectorstore, ensure_index


class RAGSystem:
    """
    RAG híbrido:
    - recuperación semántica en Pinecone;
    - recuperación léxica BM25 local;
    - Reciprocal Rank Fusion mediante EnsembleRetriever.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

        pc = ensure_index(self.settings)
        self.vectorstore = build_vectorstore(
            self.settings,
            pc,
        )

        # BM25 necesita el mismo corpus/chunking que se subió a Pinecone.
        local_chunks = load_and_split(self.settings)

        self.semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.top_k},
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            local_chunks
        )
        self.bm25_retriever.k = self.settings.top_k

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                self.semantic_retriever,
                self.bm25_retriever,
            ],
            weights=[
                self.settings.semantic_weight,
                self.settings.lexical_weight,
            ],
            id_key="chunk_id",
        )

        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.gemini_model,
        )

    async def retrieve(
        self,
        query: str,
    ) -> list[Document]:
        if not query.strip():
            raise ValueError("La consulta no puede estar vacía.")

        documents = await self.hybrid_retriever.ainvoke(
            query.strip()
        )

        # EnsembleRetriever puede fusionar hasta k de cada recuperador.
        # La interfaz pública garantiza exactamente un Top-K final.
        return documents[: self.settings.top_k]

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        blocks: list[str] = []

        for rank, document in enumerate(documents, start=1):
            metadata = document.metadata

            blocks.append(
                f"[RESULTADO {rank}]\n"
                f"doc_id: {metadata.get('doc_id')}\n"
                f"source: {metadata.get('source')}\n"
                f"page: {metadata.get('page')}\n"
                f"category: {metadata.get('category')}\n"
                f"chunk_id: {metadata.get('chunk_id')}\n"
                f"texto:\n{document.page_content}"
            )

        return "\n\n---\n\n".join(blocks)

    async def answer(self, query: str) -> str:
        """
        Demostración RAG completa. La evaluación de Precision/Recall
        mide solamente la etapa de recuperación.
        """
        documents = await self.retrieve(query)
        context = self._format_context(documents)

        messages = [
            (
                "system",
                """
Eres un asistente técnico.
Responde únicamente con información del CONTEXTO recuperado.
Si el contexto no alcanza para responder, responde: "No lo sé".
No inventes datos ni fuentes.
""".strip(),
            ),
            (
                "human",
                f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{query}",
            ),
        ]

        response = await self.llm.ainvoke(messages)

        if isinstance(response.content, str):
            return response.content

        return str(response.content)
