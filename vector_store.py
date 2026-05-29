from dotenv import load_dotenv
import os
from src.helper import fetch_text_from_PDFs, keep_source_metadata, create_chunks, fetch_embeddings_HF
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = fetch_text_from_PDFs(data="pdf/")
relevant_docs = keep_source_metadata(extracted_data)
chunks = create_chunks(relevant_docs)

embeddings = fetch_embeddings_HF()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

batch_size = 50

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    PineconeVectorStore.from_documents(
        documents=batch,
        index_name=index_name,
        embedding=embeddings,
    )