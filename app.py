import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
# Ini adalah jembatan penghubung yang baru
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

st.set_page_config(page_title="Chatbot Bisnis", page_icon="🤖")
st.title("🤖 Asisten AI Bisnis Saya")
st.caption("Tanya apa saja seputar produk, stok, atau harga!")

@st.cache_resource(show_spinner=False)
def inisialisasi_chatbot():
    # Mengambil kunci
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GEMINI_API_KEY"] = api_key # Sistem baru menggunakan nama variabel ini
    
    # 1. Menggunakan kelas GoogleGenAI yang baru dan modern
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
    Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=api_key)

    # 2. Database super baru agar tidak bentrok dengan sisa-sisa error lama
    db = chromadb.PersistentClient(path="./database_modern")
    chroma_collection = db.get_or_create_collection("data_bisnis")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Membaca data
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    return index.as_query_engine()

with st.spinner("Sedang menghubungkan ke server Google yang baru..."):
    query_engine = inisialisasi_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait produk kita hari ini?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ketik pertanyaan Anda..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi..."):
            response = query_engine.query(prompt)
            st.markdown(response.response)
            
    st.session_state.messages.append({"role": "assistant", "content": response.response})
