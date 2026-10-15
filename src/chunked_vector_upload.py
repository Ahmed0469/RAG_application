import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 
from huggingface_hub import constants
constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True

from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.node_parser import SemanticSplitterNodeParser 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings,StorageContext,VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from unstructured.partition.pdf import partition_pdf
import regex as re
from llama_index.core.schema import TextNode
import unstructured_pytesseract



#---------Loads Raw Data and Clean------------
# reader = UnstructuredReader()
# documents = reader.load_data(file="RAG/data/iso.pdf")
# with open("RAG/data/iso_cleaned.txt",'w',encoding='utf-8') as file:
#     text = ""
#     for doc in documents:
#         text += doc.text + "\n"
#     file.write(text)

# with open("RAG/data/iso_cleaned.txt",'r',encoding='utf-8') as file:
#     documents = file.read()
#     documents = re.sub(r"©\s"," ",documents)
#     documents = re.sub(r"ISO/IEC\s\w+\S+","",documents)
#     documents = re.sub(r"--\S+","",documents)
#     documents = re.sub(r"(\.){2,}","",documents)

# with open("RAG/data/iso_cleaned.txt",'w',encoding='utf-8') as file:
#     file.write(documents)


# reader = UnstructuredReader()
# documents = reader.load_data(file="RAG/data/iso.pdf")

# embedding_model = HuggingFaceEmbedding(model_name='all-MiniLM-L6-v2')

# node_parser = SemanticSplitterNodeParser(embed_model=embedding_model)

# unstructured_pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# elements = partition_pdf(
#     filename="RAG/data/iso.pdf",
#     strategy='hi_res',
#     infer_table_structure=True,
#     chunking_strategy='by_title',
# )

# table_elements =[]
# text_elements = []

# for element in elements:
#     if element.category == 'Table':
#         table_elements.append(element.metadata.text_as_html)
#     elif element.category in ["NarrativeText", "Title", "ListItem"]:
#         text_elements.append(str(element))

# nodes = []

# for text in text_elements:
#     nodes.append(TextNode(
#         text=text,
#         metadata={'type':'text'}
#     ))

# for table in table_elements:
#     nodes.append(
#         TextNode(
#             text=table,
#             metadata={'type':'table','format':'html'}
#         )
#     )

def clean_data(text):
    text = re.sub(r"©\s"," ",text)
    text = re.sub(r"ISO/IEC\s\w+\S+","",text)
    # doc.text = re.sub(r"--\S+","",doc.text)
    # doc.text = re.sub(r"(\.){2,}","",doc.text)
    return text

reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
documents = reader.load_data("RAG/data/iso.pdf")

with open("RAG/data/iso_cleaned.txt",'w',encoding='utf-8') as file:
    text = ""
    print("len of Docs : ",len(documents))
    for doc in documents:
        # node.text = clean_data(node.text)
        text += doc.text + "\n"
    file.write(text)

embedding_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

node_parser = DoclingNodeParser()

nodes = node_parser.get_nodes_from_documents(documents,True)

# nodes = [node for node in nodes if node.metadata.get('page_no') not in {1,2,3,4}]

with open("RAG/data/iso_cleaned_nodes.txt",'w',encoding='utf-8') as file:
    text = ""
    for node in nodes:
        # node.text = clean_data(node.text)
        text += node.text + "\n"
    file.write(text)

for node in nodes:
    print('metadata',node.metadata)
    break
    # node.excluded_embed_metadata_keys.append('last_modified')
    # node.excluded_embed_metadata_keys.append('languages')
    # node.excluded_embed_metadata_keys.append('file_directory')
    # node.excluded_embed_metadata_keys.append('filename')
    # node.excluded_embed_metadata_keys.append('filetype')
    # # node.excluded_embed_metadata_keys.append('coordinates')#----------------Remove this----------------------

    # node.excluded_llm_metadata_keys.append('last_modified')
    # node.excluded_llm_metadata_keys.append('languages')
    # node.excluded_llm_metadata_keys.append('file_directory')
    # node.excluded_llm_metadata_keys.append('filetype')
    # node.excluded_llm_metadata_keys.append('coordinates')#----------------Remove this----------------------
    
# Settings.embed_model = embedding_model
# # # # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# qdrant_client = QdrantClient()
# COLLECTION_NAME = "ISO_RAG_test_2"

# vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME,client=qdrant_client)

# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# index = VectorStoreIndex(
#     nodes=nodes,
#     storage_context=storage_context,
#     show_progress=True
# )