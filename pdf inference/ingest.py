# add your pdfs in data folder

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("\n\n\nSTARTING")

model_name="sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )

loader = DirectoryLoader('data/', glob="**/*.pdf",
                         show_progress=True,
                         use_multithreading=True,
                         loader_cls=PyPDFLoader)

print("\n\n\nDOCS LOADING")
documents = loader.load()
print(f"Docs Length:{len(documents)}")

print("\n\nPREPARING TEXT")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)
print("\nTEXT PREPARED")

print("\n\n\nCREATING VECTOR DB")
vector_store = Chroma.from_documents(texts, embeddings,
                                     collection_metadata={"hnsw:space": "cosine"},
                                     persist_directory="stores/rag_data")

print("VECTOR DB Successfully Created!")