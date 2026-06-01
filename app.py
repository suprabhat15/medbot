from flask import Flask, render_template, request, session
from dotenv import load_dotenv
import os
import uuid

from src.helper import fetch_embeddings_HF
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

app = Flask(__name__)
load_dotenv()

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

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
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

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return ""

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    result = rag_with_history.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session["session_id"]}}
    )

    return str(result["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)