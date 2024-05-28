import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from flask import Flask, send_from_directory
from werkzeug.utils import secure_filename
from threading import Thread

from dotenv import load_dotenv
load_dotenv()

# Load groq API Keys.
groq_api_key = "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"

st.title("ChatGroq with LLAMA3 Demo :sparkles:")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on provided context only.
Please provide Accurate response based on question and explain it widely.
<context>
{context}
<context>
Question: {input}
"""
)

# Set up a directory to save the uploaded files
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to run Flask server
def run_flask():
    app = Flask(__name__)

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(UPLOAD_FOLDER, filename)

    app.run(port=8000)

# Start Flask server in a separate thread
if 'flask_thread' not in st.session_state:
    flask_thread = Thread(target=run_flask)
    flask_thread.start()
    st.session_state['flask_thread'] = flask_thread

# Streamlit file uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

file_url = None  # Initialize file_url

if uploaded_file is not None:
    # Secure the file name
    filename = secure_filename(uploaded_file.name)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save the uploaded file to the specified file path
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_url = f"http://localhost:8000/uploads/{filename}"
    st.write(file_url)

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFLoader(file_path)  # Data Injection
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Document splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector Huggingface Embedding

prompt1 = st.text_input("Enter the Question from your Mind:")

if st.button("Document Embeddings"):
    vector_embeddings()
    st.write("Vector store DB is Ready.")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': prompt1})
    st.write(response['answer'])

st.header('', divider='rainbow')
st.markdown('''
    Developed by KARTAVYA MASTER :8ball:
''')
