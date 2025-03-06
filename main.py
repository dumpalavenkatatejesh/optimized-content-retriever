import os
import streamlit as st
import pickle
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

# Define URLs
st.title("Optimized Web Content Retreiver ")
st.sidebar.title("Article URLs")
urls = []

# Define FAISS index file path
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
main_placeholder = st.empty()
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

# Initialize OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, max_tokens=500)
if process_url_clicked:
# Load data from the web
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

# Check if data is loaded
    if not data:
        print("Error: Failed to load data from the URL.")
        exit()

# Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)


# Create embeddings and FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

# Save FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)


# Query for market scenario
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever()

            # Use ConversationalRetrievalChain and pass an empty chat_history
            chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

            try:
                result = chain.invoke({"question": query, "chat_history": []})
                st.header("Answer")
                st.write(result["answer"])
                
            except Exception as e:
                print(f"Error during retrieval: {e}")
