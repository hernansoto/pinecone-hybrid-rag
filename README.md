# Pinecone Hybrid RAG — Pre-entrega 4

Sistema RAG escalable en la nube con:

- Pinecone Serverless para recuperación semántica;
- BM25 para recuperación léxica;
- `EnsembleRetriever` para fusión híbrida;
- Gemini/OpenAI como proveedores de embeddings;
- Gemini como modelo generativo de la demo;
- `Precision@5` y `Recall@5` sobre un Golden Set.

## Arquitectura

```text
Dataset JSON
   │
   ▼
RecursiveCharacterTextSplitter
600 tokens / 80 overlap
   │
   ▼
Embeddings
Gemini 768d (default)
o OpenAI 1536d
   │
   ▼
Pinecone Serverless
namespace: technical-docs
   │
   ├──────────────────────────┐
   ▼                          ▼
Vector Retriever            BM25Retriever
Pinecone                    corpus local
   │                          │
   └────────────┬─────────────┘
                ▼
        EnsembleRetriever
     Reciprocal Rank Fusion
                │
                ▼
              Top-5
                │
        ┌───────┴────────┐
        ▼                ▼
   evaluate.py        Gemini LLM
Precision@5          respuesta RAG
Recall@5
```

## Requisitos cumplidos

- índice Pinecone Serverless creado automáticamente si no existe;
- validación de dimensión y métrica del índice;
- namespace configurable;
- ingesta de documentos técnicos JSON;
- chunking con `RecursiveCharacterTextSplitter`;
- chunks de 600 tokens aprox. y overlap de 80;
- metadata avanzada:
  - `source`;
  - `page`;
  - `category`;
  - `doc_id`;
  - `title`;
  - `chunk_id`;
- texto original almacenado en metadata de Pinecone bajo `text`;
- IDs determinísticos para upserts;
- búsqueda semántica en Pinecone;
- búsqueda léxica BM25;
- fusión mediante `EnsembleRetriever`;
- Top-5 final;
- benchmark de 5 preguntas;
- `Precision@5`;
- `Recall@5`;
- demo de generación RAG;
- secretos mediante `.env`.

## Dataset

El proyecto incluye cinco documentos técnicos:

```text
data/
├── 01_pydantic_v2.json
├── 02_asyncio.json
├── 03_fastapi.json
├── 04_sqlalchemy.json
└── 05_pinecone.json
```

Cada documento tiene:

```json
{
  "doc_id": "...",
  "title": "...",
  "category": "...",
  "pages": [
    {
      "page": 1,
      "content": "..."
    }
  ]
}
```

Esto permite conservar fuente, página y categoría durante todo el pipeline.

## 1. Crear entorno virtual

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Verificar:

```bash
python --version
```

## 2. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Crear credenciales

Se necesita:

1. una API key de Pinecone;
2. una API key de Gemini para la configuración predeterminada.

El proyecto también permite usar OpenAI para embeddings.

## 4. Configurar `.env`

Copiar:

### Windows

```powershell
Copy-Item .env.example .env
```

### Linux/macOS

```bash
cp .env.example .env
```

Configuración recomendada:

```env
PINECONE_API_KEY=tu_pinecone_key
INDEX_NAME=preentrega4-hybrid-rag
PINECONE_NAMESPACE=technical-docs
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=tu_gemini_key
OPENAI_API_KEY=

EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIMENSION=768

GEMINI_MODEL=gemini-3.8-flash

TOP_K=5
SEMANTIC_WEIGHT=0.65
LEXICAL_WEIGHT=0.35

CHUNK_SIZE=600
CHUNK_OVERLAP=80
```

No subir `.env` a Git.

## 5. Crear/verificar el índice Serverless

```bash
python setup_pinecone.py
```

Si el índice no existe se crea automáticamente con:

```text
vector type: dense
metric: cosine
dimension: 768
cloud: aws
region: us-east-1
```

Si ya existe, el script valida que la dimensión coincida.

Esto evita el error clásico de intentar subir embeddings de una dimensión
distinta a la configurada en Pinecone.

## 6. Ingestar

```bash
python ingest.py
```

El script:

1. lee los JSON;
2. conserva página, categoría y fuente;
3. genera chunks;
4. produce embeddings;
5. hace upsert en Pinecone;
6. utiliza el namespace configurado.

El texto original también queda almacenado en Pinecone. `PineconeVectorStore`
usa la metadata `text` para poder reconstruir el `Document` durante retrieval.

### Reingesta

El proyecto genera un manifiesto local en:

```text
.cache/ingestion_manifest.json
```

Si el corpus y configuración no cambiaron, evita recalcular embeddings.

Si se modificó el dataset, chunking o modelo:

```bash
python ingest.py --force
```

Esto elimina solamente el namespace configurado y lo vuelve a poblar.

## 7. Recuperador híbrido

`RAGSystem` crea dos retrievers.

### Semántico

```python
semantic_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
```

### Léxico

```python
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5
```

### Ensemble

```python
hybrid_retriever = EnsembleRetriever(
    retrievers=[
        semantic_retriever,
        bm25_retriever,
    ],
    weights=[0.65, 0.35],
    id_key="chunk_id",
)
```

`EnsembleRetriever` aplica Reciprocal Rank Fusion sobre ambos rankings.

La búsqueda híbrida es útil porque:

- Pinecone recupera similitud semántica;
- BM25 favorece coincidencias léxicas exactas;
- nombres técnicos e identificadores pueden beneficiarse de BM25;
- consultas conceptuales suelen beneficiarse del embedding.

## 8. Probar el RAG

```bash
python main.py
```

La demo:

1. ejecuta una búsqueda híbrida;
2. muestra los Top-5;
3. genera una respuesta grounded con Gemini.

La evaluación no llama al LLM porque mide exclusivamente la calidad del
recuperador.

## 9. Ejecutar evaluación

```bash
python evaluate.py
```

`golden_set.json` contiene cinco casos:

```json
{
  "pregunta": "...",
  "documento_id_esperado": "..."
}
```

### Recall@5

En este benchmark existe un documento fuente esperado por consulta.

```text
Recall@5 = 1
```

si al menos un chunk del documento esperado aparece entre los cinco resultados.

En caso contrario:

```text
Recall@5 = 0
```

El resultado final es el promedio de las cinco consultas.

### Precision@5

Cada chunk perteneciente al documento esperado se considera relevante.

```text
Precision@5 =
chunks relevantes recuperados / resultados recuperados
```

Por ejemplo, si tres de los cinco chunks provienen del documento esperado:

```text
Precision@5 = 3 / 5 = 0.60
```

El script imprime las métricas por pregunta y el promedio final.

## Golden Set

Las cinco preguntas evalúan:

1. Pydantic;
2. asyncio;
3. FastAPI;
4. SQLAlchemy;
5. Pinecone.

El objetivo es comprobar que términos exactos y conceptos semánticos puedan
beneficiarse de la combinación BM25 + vectores.

## Namespace

El proyecto usa:

```env
PINECONE_NAMESPACE=technical-docs
```

El namespace se utiliza tanto en la ingesta como en las consultas.

En una arquitectura multi-tenant podría cambiarse, por ejemplo, a:

```text
tenant-a
tenant-b
tenant-c
```

para separar lógicamente los vectores de cada cliente.

## Metadata de Pinecone

Cada chunk contiene metadata similar a:

```json
{
  "doc_id": "python-asyncio",
  "title": "Python asyncio: concurrencia cooperativa",
  "category": "python-async",
  "source": "02_asyncio.json",
  "page": 1,
  "chunk_index": 0,
  "chunk_id": "python-asyncio:p1:c0",
  "text": "contenido original del chunk..."
}
```

Esto satisface el requisito de mantener el texto dentro de Pinecone y evita
necesitar otra base de datos para reconstruir el contexto.

## Alternativa OpenAI

La consigna menciona `text-embedding-3-small`.

El proyecto lo soporta.

Cambiar `.env` a:

```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=tu_openai_key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

Como un índice Pinecone tiene dimensión fija, si el índice original fue creado
a 768 dimensiones se debe usar otro `INDEX_NAME` o recrearlo.

Por ejemplo:

```env
INDEX_NAME=preentrega4-openai-rag
```

Luego:

```bash
python setup_pinecone.py
python ingest.py
```

La configuración predeterminada usa Gemini para permitir realizar la entrega
sin depender de créditos de la API de OpenAI.

## Seguridad

`.gitignore` excluye:

```text
.env
.venv/
.cache/
__pycache__/
```

Antes del push:

```bash
git check-ignore .env
```

debe devolver:

```text
.env
```

Nunca colocar claves reales en `.env.example`.

## Archivos principales

| Archivo | Responsabilidad |
| --- | --- |
| `setup_pinecone.py` | Crear/verificar índice Serverless |
| `ingest.py` | Chunking + embeddings + upsert |
| `pinecone_store.py` | Cliente e integración Pinecone |
| `documents.py` | Carga JSON y chunking |
| `embeddings.py` | Gemini/OpenAI embeddings |
| `rag_system.py` | Retriever híbrido + demo RAG |
| `evaluate.py` | Precision@5 y Recall@5 |
| `golden_set.json` | Benchmark |
| `main.py` | Demo end-to-end |

## Orden de ejecución

```bash
python setup_pinecone.py
python ingest.py
python main.py
python evaluate.py
```

## Criterios de entrega

| Criterio | Implementación |
| --- | --- |
| Pinecone Serverless | Sí |
| Setup automático | `setup_pinecone.py` |
| Validación dimensión | Sí |
| Dataset técnico | 5 documentos JSON |
| RecursiveCharacterTextSplitter | Sí |
| Chunking 500-800 tokens | 600 |
| Overlap | 80 |
| Metadata fuente | Sí |
| Metadata página | Sí |
| Metadata categoría | Sí |
| Texto en metadata | `text` |
| Namespace | Sí |
| Retriever vectorial | Pinecone |
| Retriever léxico | BM25 |
| Retriever híbrido | `EnsembleRetriever` |
| Rank fusion | Reciprocal Rank Fusion |
| Top-5 | Sí |
| Golden Set | 5 preguntas |
| Precision@5 | Sí |
| Recall@5 | Sí |
| Demo RAG | Sí |
| `.env.example` | Sí |
| Sin secrets en Git | Sí |
