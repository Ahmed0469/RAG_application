from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.node_parser.docling import DoclingNodeParser
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.core import Settings,StorageContext,VectorStoreIndex
from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
import pickle
import uuid

DOCUMENTS_FILE = "RAG/data/documents.pkl"

with open(DOCUMENTS_FILE,'rb') as file:
    documents = pickle.load(file)

# node_parser = DoclingNodeParser()
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('##','Heading')],strip_headers=False)
chunks = text_splitter.split_text(documents[0].text)

# nodes = node_parser.get_nodes_from_documents(documents,True)
nodes = [TextNode(
    id_= str(uuid.uuid4()),  
    text=chunk.page_content,
) for chunk in chunks]

embedding_model = HuggingFaceEmbedding('BAAI/bge-base-en-v1.5')

Settings.embed_model = embedding_model

qdrant_client = QdrantClient()
COLLECTION_NAME = 'Docling_Chunks_Markdown'

vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME,client=qdrant_client)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes=nodes,storage_context=storage_context,show_progress=True)