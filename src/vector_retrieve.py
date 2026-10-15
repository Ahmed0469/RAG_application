from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex,StorageContext,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from vector_upload import JSONL,load_nodes

qdrant_client = QdrantClient(':memory:')
COLLECTION_NAME = "Langchain_Rag"

embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

Settings.embed_model = embedding_model

vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME,client=qdrant_client)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store,storage_context=storage_context)

# retriever = index.as_retriever(similarity_top_k=6)

def build_retrievers(nodes, index):
    from llama_index.core.llms import MockLLM
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.retrievers.bm25 import BM25Retriever
 
    vector = index.as_retriever(similarity_top_k=5)
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    hybrid = QueryFusionRetriever(
        [vector, bm25], similarity_top_k=5, num_queries=1,
        mode="reciprocal_rerank", use_async=False,
        llm=MockLLM(),
    )
    return vector, bm25, hybrid

# nodes = load_nodes(JSONL)
# vector, bm25, hybrid = build_retrievers(nodes,index)
retriever = index.as_retriever(similarity_top_k=5)

import json

questions_dir = "RAG/data/questions.jsonl"

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

    result_chunk = {
        'question' : question,
        'ground_truth' : ground_truth,
        'custom_chunking_chunks' : custom_chunks,
    }
    result_chunks.append(result_chunk)

with open("RAG/data/Result_Chunks_docling_markdown.jsonl",'w') as file:
    json.dump(result_chunks,file)

# query = ""
# while query != 'q':
#     query = input("Enter your query: ")

#     for hit in hybrid.retrieve(query):
#         m = hit.node.metadata
#         print("\nChunk Data : ")
#         print(f"{hit.score:.3f}  {m.get('clause') or m.get('control_id')}  {m['title']}")
#         print("text : ",hit.node.text)

