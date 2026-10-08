from config import Settings
from pinecone_store import ensure_index


def main() -> None:
    settings = Settings.from_env()
    ensure_index(settings)

    print("\nConfiguración validada:")
    print(f"- index: {settings.index_name}")
    print(f"- namespace: {settings.namespace}")
    print(f"- cloud: {settings.pinecone_cloud}")
    print(f"- region: {settings.pinecone_region}")
    print(
        f"- embeddings: {settings.embedding_provider}/"
        f"{settings.embedding_model}"
    )
    print(f"- dimension: {settings.embedding_dimension}")


if __name__ == "__main__":
    main()
