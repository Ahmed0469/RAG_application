from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
import uuid
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex,Settings,StorageContext

PDF_DIR = "RAG/data/iso.pdf"

pdf_reader = PyPDFLoader(PDF_DIR)
documents = pdf_reader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

nodes = [TextNode(
    id_= str(uuid.uuid4()),  
    text=chunk.page_content,
) for chunk in chunks]

embedding_model = HuggingFaceEmbedding('BAAI/bge-m3')
Settings.embed_model = embedding_model

COLLECTION_NAME = "Langchain_Rag"
qdrant_client = QdrantClient(':memory:')

vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME,client=qdrant_client)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    show_progress=True
)