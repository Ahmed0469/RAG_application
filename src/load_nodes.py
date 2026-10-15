"""
Load iso27001_chunks.jsonl into LlamaIndex TextNodes, index them, and query.

    pip install llama-index-core llama-index-retrievers-bm25
    # plus ONE embedding backend:
    pip install llama-index-embeddings-openai            # needs OPENAI_API_KEY
    pip install llama-index-embeddings-huggingface       # fully local

    python load_nodes.py
"""
import json
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

JSONL = Path("iso27001_chunks.jsonl")

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
        # Supports both layouts: {"id","text","metadata":{...}} and flat {"chunk_id","text",...}
        raw_meta = c["metadata"] if "metadata" in c else {k: v for k, v in c.items() if k not in NON_META}
        meta = flatten(raw_meta)
        node_id = c.get("id") or c.get("chunk_id") or raw_meta["chunk_id"]
        nodes.append(TextNode(
            id_=node_id,                       # stable IDs -> safe to re-ingest / upsert
            text=c["text"],                    # clean text, no breadcrumb duplication
            metadata=meta,
            excluded_embed_metadata_keys=[k for k in meta if k not in EMBED_KEYS],
            excluded_llm_metadata_keys=[k for k in meta if k in LLM_HIDDEN],
            metadata_separator="\n",
            metadata_template="{key}: {value}",
        ))

    # Link neighbours (document order) so you can pull in prev/next context later.
    for a, b in zip(nodes, nodes[1:]):
        a.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=b.node_id)
        b.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=a.node_id)
    return nodes


def setup_embeddings():
    """Pick one. Local option needs no API key."""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except ImportError:
        from llama_index.embeddings.openai import OpenAIEmbedding
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def build_retrievers(nodes, index):
    from llama_index.core.llms import MockLLM
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.retrievers.bm25 import BM25Retriever

    vector = index.as_retriever(similarity_top_k=5)
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    # Hybrid: BM25 catches exact terms ("Statement of Applicability", "A.8.28"),
    # vectors catch paraphrases. Reciprocal-rank fusion merges them.
    hybrid = QueryFusionRetriever(
        [vector, bm25], similarity_top_k=5, num_queries=1,   # num_queries=1 -> no query rewriting
        mode="reciprocal_rerank", use_async=False,
        llm=MockLLM(),   # required by the constructor but never called when num_queries=1
    )
    return vector, bm25, hybrid


def get_control(index, control_id: str):
    """Exact lookup by control ID, skipping vector search entirely."""
    retriever = index.as_retriever(
        similarity_top_k=1,
        filters=MetadataFilters(filters=[
            MetadataFilter(key="control_id", value=control_id, operator=FilterOperator.EQ)]),
    )
    return retriever.retrieve(control_id)


if __name__ == "__main__":
    nodes = load_nodes()
    print(f"{len(nodes)} nodes loaded")

    setup_embeddings()
    index = VectorStoreIndex(nodes)            # embeds each node once

    vector, bm25, hybrid = build_retrievers(nodes, index)
    for hit in hybrid.retrieve("What must a Statement of Applicability contain?"):
        m = hit.node.metadata
        print(f"{hit.score:.3f}  {m.get('clause') or m.get('control_id')}  {m['title']}")

    print(get_control(index, "A.8.28")[0].node.get_content()[:200])

    # Optional: persist, then reload later with load_index_from_storage
    # index.storage_context.persist("./storage")
