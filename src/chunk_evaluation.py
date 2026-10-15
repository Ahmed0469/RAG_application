from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex,StorageContext,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from vector_upload import JSONL,load_nodes
import json

questions_dir = "RAG/data/questions.jsonl"

qdrant_client = QdrantClient()
COLLECTION_NAME = "Structured_RAG"
COLLECTION_NAME_2 = "Docling_Chunks"

# Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.embed_model = embedding_model

vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME,client=qdrant_client)
vector_store_2 = QdrantVectorStore(collection_name=COLLECTION_NAME_2,client=qdrant_client)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
storage_context_2 = StorageContext.from_defaults(vector_store=vector_store_2)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store,storage_context=storage_context)
index_2 = VectorStoreIndex.from_vector_store(vector_store=vector_store_2,storage_context=storage_context_2)

retriever = index.as_retriever(similarity_top_k=6)
retriever_2 = index_2.as_retriever(similarity_top_k=10)

rows = []
with open(questions_dir) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print(f"[eval] loaded {len(rows)} questions from {questions_dir}")

result_chunks = []

for row in rows:
    question = row['question']
    ground_truth = row['ground_truth']

    custom_chunk_results = retriever.retrieve(question)
    custom_chunks = [chunk.node.text for chunk in custom_chunk_results]

    docling_chunk_results = retriever_2.retrieve(question)
    docling_chunks = [chunk.node.text for chunk in docling_chunk_results]

    result_chunk = {
        'question' : question,
        'ground_truth' : ground_truth,
        'custom_chunking_chunks' : custom_chunks,
        'docling_chunking_chunks' : docling_chunks
    }
    result_chunks.append(result_chunk)

with open("RAG/data/Result_Chunks.jsonl",'w') as file:
    json.dump(result_chunks,file)