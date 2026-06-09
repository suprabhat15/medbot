from flask import Flask, render_template, request, session, jsonify
from dotenv import load_dotenv
import os
import uuid
import tempfile

from werkzeug.utils import secure_filename

from src.helper import (
    fetch_embeddings_HF,
    fetch_text_from_PDF_file,
    keep_source_metadata,
    create_chunks,
)
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableField

app = Flask(__name__)
load_dotenv()
app.secret_key = os.environ["FLASK_SECRET_KEY"]

# Reject uploads larger than 35 MB before they reach the handler.
app.config["MAX_CONTENT_LENGTH"] = 35 * 1024 * 1024

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = fetch_embeddings_HF()

index_name = "medbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# search_kwargs is exposed as a configurable field so each request can scope
# retrieval to a Pinecone namespace (the shared corpus by default, or the
# session's own namespace once the user has uploaded a PDF).
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
).configurable_fields(
    search_kwargs=ConfigurableField(id="retriever_search_kwargs")
)

chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

# 1) Rewrite follow-up questions into standalone questions
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the latest user question into a standalone question using the chat history. "
        "Do not answer the question. If the question is already standalone, return it as-is."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    chat_model,
    retriever,
    contextualize_q_prompt
)

# 2) Answer using retrieved context + chat history
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{input}")
])

question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)

# 3) Store chat history per session
# store = {} keeps memory in RAM only. That is fine for local development, but it resets when the server restarts. For production, use Redis or a database for chat history.
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

rag_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/history")
def history():
    session_id = session.get("session_id")
    if not session_id or session_id not in store:
        return jsonify([])

    messages = []
    for m in store[session_id].messages:
        role = "user" if m.type == "human" else "bot"
        messages.append({"role": role, "text": m.content})
    return jsonify(messages)

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return ""

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    # Once the user has uploaded a PDF, answer from their document's namespace;
    # otherwise fall back to the shared medical corpus (default namespace).
    namespace = session["session_id"] if session.get("has_upload") else ""

    result = rag_with_history.invoke(
        {"input": msg},
        config={"configurable": {
            "session_id": session["session_id"],
            "retriever_search_kwargs": {"k": 3, "namespace": namespace},
        }}
    )

    return str(result["answer"])

@app.route("/upload", methods=["POST"])
def upload():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file provided."}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    namespace = session["session_id"]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        docs = fetch_text_from_PDF_file(tmp_path)
        if not docs:
            return jsonify({"error": "Could not extract any text from this PDF."}), 400

        # Keep the original filename as the source so answers can cite it.
        for doc in docs:
            doc.metadata["source"] = filename

        relevant_docs = keep_source_metadata(docs)
        chunks = create_chunks(relevant_docs)

        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            PineconeVectorStore.from_documents(
                documents=chunks[i:i + batch_size],
                index_name=index_name,
                embedding=embeddings,
                namespace=namespace,
            )

        session["has_upload"] = True

        return jsonify({
            "status": "ok",
            "filename": filename,
            "chunks": len(chunks),
        })
    except Exception:
        app.logger.exception("PDF upload failed")
        return jsonify({"error": "Failed to process the PDF. Please try again."}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File is too large. Maximum size is 35 MB."}), 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)