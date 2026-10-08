from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rag_system import RAGSystem

GOLDEN_SET_PATH = Path("./golden_set.json")


def load_golden_set() -> list[dict]:
    payload = json.loads(
        GOLDEN_SET_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(payload, list) or len(payload) < 5:
        raise ValueError(
            "golden_set.json debe contener al menos 5 casos."
        )

    return payload


async def evaluate() -> None:
    rag = RAGSystem()
    benchmark = load_golden_set()

    precision_scores: list[float] = []
    recall_scores: list[float] = []

    print(
        f"\nEvaluación híbrida "
        f"(Top-{rag.settings.top_k})\n"
        + "=" * 72
    )

    for index, case in enumerate(benchmark, start=1):
        question = case["pregunta"]
        expected_doc_id = case["documento_id_esperado"]

        docs = await rag.retrieve(question)

        retrieved_doc_ids = [
            str(doc.metadata.get("doc_id"))
            for doc in docs
        ]

        # Con un documento fuente relevante por pregunta:
        # Recall@k = 1 si al menos un chunk de ese documento aparece.
        recall_at_k = (
            1.0
            if expected_doc_id in retrieved_doc_ids
            else 0.0
        )

        # Cada chunk proveniente del documento esperado se considera útil.
        useful_results = sum(
            1
            for doc_id in retrieved_doc_ids
            if doc_id == expected_doc_id
        )

        precision_at_k = (
            useful_results / len(retrieved_doc_ids)
            if retrieved_doc_ids
            else 0.0
        )

        recall_scores.append(recall_at_k)
        precision_scores.append(precision_at_k)

        print(f"\nCaso {index}")
        print(f"Pregunta: {question}")
        print(f"Esperado: {expected_doc_id}")
        print(
            "Recuperados: "
            + ", ".join(retrieved_doc_ids)
        )
        print(
            f"Precision@{rag.settings.top_k}: "
            f"{precision_at_k:.3f}"
        )
        print(
            f"Recall@{rag.settings.top_k}: "
            f"{recall_at_k:.3f}"
        )

    macro_precision = (
        sum(precision_scores) / len(precision_scores)
    )
    macro_recall = (
        sum(recall_scores) / len(recall_scores)
    )

    print("\n" + "=" * 72)
    print("RESUMEN")
    print(
        f"Precision@{rag.settings.top_k} promedio: "
        f"{macro_precision:.3f} "
        f"({macro_precision * 100:.1f}%)"
    )
    print(
        f"Recall@{rag.settings.top_k} promedio: "
        f"{macro_recall:.3f} "
        f"({macro_recall * 100:.1f}%)"
    )


if __name__ == "__main__":
    asyncio.run(evaluate())
