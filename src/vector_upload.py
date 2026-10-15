from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings,StorageContext,VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core.schema import TextNode,NodeRelationship,RelatedNodeInfo
from pathlib import Path
import json
import uuid

JSONL = Path("RAG/data/pdf_chunks.jsonl")

# Only the breadcrumb is embedded; everything else is filter/citation metadata.
EMBED_KEYS = {"breadcrumb"}
# Keep the LLM prompt tidy: it sees breadcrumb + text, not bookkeeping fields.
LLM_HIDDEN = {"chunk_id", "tokens", "token_count", "page_pdf", "page_pdf_end", "pdf_page",
              "standard", "edition", "has_shall", "has_note", "has_notes", "notes",
              "mentions_documented_info", "keywords", "theme_no", "theme_id",
              "iso27002_ref", "annex_ref", "parent", "external_references"}
# Fields that are chunk content/ids rather than metadata (flat-schema files)
NON_META = {"id", "text", "embed_text"}


def flatten(meta: dict) -> dict:
    """Vector stores (Chroma, Pinecone, ...) want flat primitives, not lists/None."""
    flat = {}
    for k, v in meta.items():
        if v is None:
            continue
        if k == "breadcrumb":
            flat[k] = " > ".join([meta.get("standard", ""), *v]).strip(" >")
        elif isinstance(v, list):
            flat[k] = ", ".join(map(str, v))
        else:
            flat[k] = v
    return flat


def load_nodes(path: Path = JSONL) -> list[TextNode]:
    nodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        raw_meta = c["metadata"] if "metadata" in c else {k: v for k, v in c.items() if k not in NON_META}
        meta = flatten(raw_meta)
        
        # 1. Get your raw string ID
        raw_id = str(c.get("id") or c.get("chunk_id") or raw_meta["chunk_id"])
        
        # 2. Convert it into a deterministic UUID (v5)
        # NAMESPACE_DNS guarantees that the same raw_id always generates the exact same UUID
        qdrant_safe_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
        
        # 3. Store the original readable ID in metadata so you don't lose it
        meta["original_chunk_id"] = raw_id
        
        nodes.append(TextNode(
            id_=qdrant_safe_id,  # <-- Pass the valid UUID string here
            text=c["text"],
            metadata=meta,
            excluded_embed_metadata_keys=[k for k in meta if k not in EMBED_KEYS],
            excluded_llm_metadata_keys=[k for k in meta if k in LLM_HIDDEN],
            metadata_separator="\n",
            metadata_template="{key}: {value}",
        ))

    for a, b in zip(nodes, nodes[1:]):
        a.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=b.node_id)
        b.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=a.node_id)
    return nodes

nodes = load_nodes(JSONL)
# embedding_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.embed_model = embedding_model

COLLECTION_NAME = "Structured_RAG"
qdrant_client = QdrantClient()

vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME,client=qdrant_client)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    show_progress=True
)