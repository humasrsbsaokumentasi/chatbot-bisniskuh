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
    
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

    db = chromadb.PersistentClient(path="./database_mandiri")
    chroma_collection = db.get_or_create_collection("data_bisnis")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    return index.as_query_engine()

# --- FUNGSI BARU UNTUK FOTO ---
def ekstrak_dan_tampilkan_foto(teks_jawaban):
    """
    Mencari ID Foto Drive di dalam jawaban AI dan menampilkannya sebagai gambar.
    """
    # Mencari pola ID Drive (biasanya 33-44 karakter)
    # Kita asumsikan di database Anda, ID foto diawali dengan kata 'FOTO:' 
    # agar AI mudah menuliskannya di jawaban.
    match = re.search(r"ID_FOTO:\s*([\w-]+)", teks_jawaban)
    
    if match:
        file_id = match.group(1)
        direct_link = f"https://drive.google.com/uc?export=view&id={file_id}"
        st.image(direct_link, caption="Dokumentasi Produk", width="stretch")

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
            # Kita instruksikan AI agar selalu menyertakan ID_FOTO jika ada di data
            instruksi_tambahan = f"\n\nPenting: Jika ada kolom ID_Foto_Drive di data, tuliskan di akhir jawaban dengan format 'ID_FOTO: [ID]'"
            response = query_engine.query(prompt + instruksi_tambahan)
            
            st.markdown(response.response)
            # Jalankan fungsi foto tepat setelah jawaban teks muncul
            ekstrak_dan_tampilkan_foto(response.response)
            
    st.session_state.messages.append({"role": "assistant", "content": response.response})
