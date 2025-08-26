import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatOpenAI(
    openai_api_key=os.getenv("NVIDIA_API_KEY"),
    openai_api_base=os.getenv("NVIDIA_API_BASE"),
    model_name=os.getenv("NVIDIA_MODEL")
)

st.set_page_config("NVIDIA LLaMA Chatbot")
st.title("Chat with NVIDIA LLaMA (Nemotron-70B)")
st.markdown("Upload a PDF and ask questions about it. The bot remembers the conversation.")


uploaded_file = st.file_uploader("Upload PDF", type="pdf")


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    full_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

   
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([full_text])

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={"device": "cpu"})
    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success("PDF processed. You can now ask questions.")


for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**LLaMA:** {msg.content}")


def handle_user_input():
    user_input = st.session_state.temp_input  
    if user_input and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        
        history_str = ""
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                history_str += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_str += f"Assistant: {msg.content}\n"

        
        prompt = f"""
You are a helpful assistant. Use the conversation and the context below to answer the current question.
If the context does not contain the answer, say "The answer is not in the document."

### Previous Conversation:
{history_str}

### Context from Document:
{context}

### Current Question:
{user_input}

### Answer:
"""

        st.session_state.chat_history.append(HumanMessage(content=user_input))

        
        response = llm([HumanMessage(content=prompt)])

        
        st.session_state.chat_history.append(AIMessage(content=response.content))


st.text_input("You:", key="temp_input", on_change=handle_user_input)

