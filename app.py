import streamlit as st
import fitz  # PyMuPDF
from model_utils import load_ner_model, extract_entities

st.set_page_config(page_title="Resume Data Extractor", layout="wide")

st.title("📄 Resume Data Extractor using NLP")
st.write("Upload a PDF resume and extract structured information using NLP.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    # Read PDF text
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page in pdf_doc:
        full_text += page.get_text()

    st.subheader("📃 Extracted Text")
    st.text_area("Resume Content", full_text, height=200)

    # Load and run model
    with st.spinner("Analyzing resume..."):
        ner_model = load_ner_model()
        extracted_data = extract_entities(full_text, ner_model)

    st.subheader("📌 Extracted Entities")
    for entity_type, items in extracted_data.items():
        st.markdown(f"**{entity_type}:** {', '.join(set(items))}")
