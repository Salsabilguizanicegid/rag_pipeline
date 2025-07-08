import streamlit as st
from rag_pipeline import RAGPipeline

PDF_FOLDER = r"C:\Users\salsabil.guizani\OneDrive - CEGID\Desktop\Salsabil AI-POD\pdfs"
INDEX_PATH = "faiss_index.idx"

@st.cache_resource
def load_pipeline():
    pipeline = RAGPipeline(pdf_folder=PDF_FOLDER, index_path=INDEX_PATH)
    if not pipeline.prepare_pipeline():
        return None
    return pipeline

pipeline = load_pipeline()

st.set_page_config(page_title="RAG AI Assistant", layout="wide")

st.markdown("""
<style>
.chat-box {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 8px;
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 20px;
}
.user-message {
    background-color: gray;
    color: black;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: right;
}
.bot-message {
    background-color: #212529;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: left;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.title("RAG Assistant")
st.markdown("Posez vos questions à propos des documents PDF chargés.")

if pipeline is None:
    st.error("Échec du chargement des documents. Vérifie le dossier ou le contenu.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("question_form", clear_on_submit=True):
    user_input = st.text_input("Votre question :", placeholder="Ex: Quels sont les taux de TVA ?")
    submitted = st.form_submit_button("Envoyer")

chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for role, msg in st.session_state.chat_history:
        class_name = "user-message" if role == "user" else "bot-message"
        st.markdown(f'<div class="{class_name}">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if submitted and user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("L'assistant réfléchit..."):
        answer = pipeline.search_and_respond(user_input)
    st.session_state.chat_history.append(("bot", answer))
    st.rerun()
