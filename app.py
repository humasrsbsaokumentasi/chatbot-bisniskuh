import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

st.set_page_config(page_title="Chatbot Bisnis", page_icon="🤖")
st.title("🤖 Asisten AI Bisnis Saya")
st.caption("Tanya apa saja seputar produk, stok, atau harga!")

@st.cache_resource(show_spinner=False)
def inisialisasi_chatbot():
    # Mengambil kunci dari brankas Streamlit
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # 1. PASTIKAN NAMA MODEL PERSIS SEPERTI INI (Jangan diubah ke versi lain)
    Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=api_key)
    Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=api_key)

    # 2. KITA BUAT NAMA FOLDER DATABASE BARU ("database_online") AGAR TIDAK BENTROK
    db = chromadb.PersistentClient(path="./database_online")
    chroma_collection = db.get_or_create_collection("data_bisnis")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Membaca data dari folder "data" yang ada di GitHub
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    return index.as_query_engine()

with st.spinner("Sedang menyiapkan data untuk pertama kalinya (ini butuh waktu beberapa detik)..."):
    query_engine = inisialisasi_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait bisnis kita hari ini?"}]

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
