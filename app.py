import streamlit as st
from rag_pipeline import read_pdf, chunk_text, embed_texts, create_faiss_index, search_and_respond

@st.cache_resource
def prepare_rag_pipeline(pdf_path: str):
    text = read_pdf(pdf_path)
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    index = create_faiss_index(vectors)
    return chunks, vectors, index

PDF_PATH = r"C:\Users\salsabil.guizani\OneDrive - CEGID\Desktop\Salsabil AI-POD\guide_fiscal_entreprise.pdf"
chunks, vectors, index = prepare_rag_pipeline(PDF_PATH)


st.set_page_config(page_title="RAG AI Assistant", layout="wide")

st.markdown("""
<style>
.chat-box {
    background-color: gray;
    padding: 2px;
    border-radius: 2px;
    height: 1px;
    overflow-y: auto;
}
.user-message {
    background-color: gray;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: right;
}
.bot-message {
    background-color: black;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: left;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.title("RAG Assistant")
st.markdown("Posez vos questions à propos du guide fiscal chargé.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("question_form", clear_on_submit=True):
    user_input = st.text_input("Votre question :", placeholder="Ex: Quels sont les taux de TVA applicables ?")
    submitted = st.form_submit_button("Envoyer")

chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div class="user-message">{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if submitted and user_input:
    st.session_state.chat_history.append(("user", user_input))
    
    with st.spinner("L'assistant réfléchit..."):
        answer = search_and_respond(user_input, chunks, index, vectors)
    
    st.session_state.chat_history.append(("bot", answer))
    st.rerun()  
