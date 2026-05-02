import streamlit as st
import os
import re # Untuk mencari pola teks
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

st.set_page_config(page_title="Chatbot Bisnis", page_icon="🤖")
st.title("🤖 Asisten AI Bisnis Saya")
st.caption("Tanya apa saja seputar produk, stok, atau harga!")

@st.cache_resource(show_spinner=False)
def inisialisasi_chatbot():
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GEMINI_API_KEY"] = api_key
    
    # Menggunakan model yang sudah terbukti stabil di environment Anda
    #Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
    Settings.llm = GoogleGenAI(model="gemini-1.5-flash", api_key=api_key)
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

    db = chromadb.PersistentClient(path="./database_mandiri")
    chroma_collection = db.get_or_create_collection("data_bisnis")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    return index.as_query_engine()

# --- FUNGSI MODIFIKASI UNTUK ESTETIKA ---
def proses_jawaban_dan_foto(teks_jawaban):
    """
    Mencari ID Foto Drive, mengambilnya, lalu menghapus kodenya dari teks jawaban.
    """
    pola_id = r"ID_FOTO:\s*([\w-]+)"
    match = re.search(pola_id, teks_jawaban)
    
    file_id = None
    teks_bersih = teks_jawaban
    
    if match:
        file_id = match.group(1)
        # Menghapus ID_FOTO dari tampilan teks agar lebih estetik
        teks_bersih = re.sub(pola_id, "", teks_jawaban).strip()
    
    return teks_bersih, file_id

# ------------------------------

with st.spinner("Menyiapkan data..."):
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
            # Instruksi agar AI memberikan ID tanpa perlu kita tampilkan ke user
            instruksi_tambahan = "\n\nPenting: Jika ada ID_Foto_Drive di data, wajib tulis di akhir jawaban dengan format 'ID_FOTO: [ID]'"
            response = query_engine.query(prompt + instruksi_tambahan)
            
            # Memisahkan teks bersih dengan ID foto
            jawaban_final, id_foto = proses_jawaban_dan_foto(response.response)
            
            # 1. Tampilkan teks yang sudah bersih (tanpa ID_FOTO)
            st.markdown(jawaban_final)
            
            # 2. Tampilkan foto jika ID ditemukan
            if id_foto:
                direct_link = f"https://drive.google.com/thumbnail?id={id_foto}&sz=w1000"
                st.image(direct_link, caption="Dokumentasi Produk", width="stretch")
            
    # Simpan jawaban yang sudah bersih ke dalam history chat
    st.session_state.messages.append({"role": "assistant", "content": jawaban_final})
