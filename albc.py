import streamlit as st
import google.generativeai as genai
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document as LCDocument
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import io
import json
import re
import tempfile
import os
import mimetypes
from datetime import datetime
from typing import Dict, Any, List

# --- Page Config ---
st.set_page_config(page_title="RAG-Powered Universal AI Document Filler", layout="wide", page_icon="🧠")
st.title("🧠 RAG-Powered Universal AI Document Filler")
st.markdown("""
**RAG-Enhanced: Retrieves exact context from sources before filling—no more misses!**

Chunks sources → Embeds → Retrieves relevant pieces → Augments prompts for precise extraction.
Upload template (.docx) and sources (PDFs, images, scans). AI fills perfectly.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Gemini Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Get free key: https://aistudio.google.com/app/apikey")
    model_choice = st.selectbox(
        "Model (Pro for handwriting/complex docs)",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=1
    )
    embedding_model = "models/embedding-001"  # Gemini's embedding model (free tier available)
    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512, help="Smaller = finer retrieval")
    top_k = st.slider("Top-K Chunks per Field", 1, 5, 3, help="More = richer context")
    st.info("**RAG Magic**: Embeds sources with Gemini → Retrieves exact matches → Generates grounded fills.")

# --- Uploads ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("1. Template (.docx with [placeholders])", type="docx")
with col2:
    source_files = st.file_uploader("2. Source Files (PDF, Images, Scans...)", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "heic"])

if st.button("🚀 RAG-Extract & Fill Form", type="primary"):
    if not api_key or not template_file or not source_files:
        st.error("Please provide API key, template, and source files.")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_choice)
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=api_key)

    progress = st.progress(0)
    status = st.empty()

    # Step 1: Extract placeholders
    status.text("🔍 Reading template placeholders...")
    doc = Document(template_file)
    placeholder_pattern = r'\[([^\]]+)\]'
    placeholders = set()

    for paragraph in doc.paragraphs:
        placeholders.update(re.findall(placeholder_pattern, paragraph.text))
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    placeholders.update(re.findall(placeholder_pattern, paragraph.text))

    placeholders = sorted([p.strip() for p in placeholders if p.strip()])
    
    if not placeholders:
        st.warning("No placeholders found! Use format like [Client Name], [Date].")
        st.stop()

    progress.progress(10)
    status.text(f"Found {len(placeholders)} fields: {', '.join(placeholders[:8])}{'...' if len(placeholders)>8 else ''}")

    # Step 2: RAG Indexing - Load, Chunk, Embed Sources
    status.text("📚 RAG Indexing: Extracting & embedding source chunks...")
    documents = []
    for file in source_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        if file.name.lower().endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        else:  # Images: Use Gemini Vision to extract text/description
            g_file = genai.upload_file(path=tmp_path, mime_type=mimetypes.guess_type(tmp_path)[0] or "image/jpeg")
            vision_prompt = "Extract all text, describe images, and summarize key details (names, dates, numbers, descriptions)."
            vision_response = model.generate_content([vision_prompt, g_file])
            docs = [LCDocument(page_content=vision_response.text, metadata={"source": file.name})]
            genai.delete_file(g_file.name)

        documents.extend(docs)
        os.unlink(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embed & Store in Chroma (in-memory for simplicity; use persist_directory for production)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=tempfile.mkdtemp())
    progress.progress(40)

    # Step 3: RAG Retrieval & Generation per Placeholder
    status.text("🔄 RAG Pipeline: Retrieving & extracting per field...")
    extracted_data = {}
    prompt_template = """
    Context from sources: {context}

    Template Field: {field}

    Extract the exact value for [{field}] from the context only. Use semantic matching (e.g., [DATE_LOSS] → "DOL" or "Incident Date").
    If missing: "Not Found".
    Output JSON: {{"value": "extracted_value", "confidence": 0.9, "source_snippet": "brief quote"}}
    """
    chain = (
        ChatPromptTemplate.from_template(prompt_template)
        | model  # Use Gemini as LLM
        | JsonOutputParser()
    )

    for i, field in enumerate(placeholders):
        # Semantic query for retrieval
        query = f"What is the value for '{field}'? Look for related terms like {field.lower().replace('_', ' or ')}."
        relevant_chunks = vectorstore.similarity_search(query, k=top_k)
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        # Generate with RAG-augmented prompt
        try:
            response = chain.invoke({"context": context, "field": field})
            extracted_data[field] = response
        except Exception as e:
            extracted_data[field] = {"value": "Not Found", "confidence": 0.0, "source_snippet": str(e)}

        progress.progress(40 + (50 * (i + 1) / len(placeholders)))
        status.text(f"Processed {i+1}/{len(placeholders)}: {field}")

    progress.progress(90)
    status.text("📝 Filling template...")

    # Preview Extracted Data
    st.subheader("RAG-Extracted Data (with Confidence & Sources)")
    high_conf = {k: v for k, v in extracted_data.items() if v.get("confidence", 0) >= 0.7}
    low_conf = {k: v for k, v in extracted_data.items() if v.get("confidence", 0) < 0.7}

    col_a, col_b = st.columns(2)
    with col_a:
        st.success(f"High Confidence ({len(high_conf)})")
        st.json({k: {"value": v["value"], "source": v["source_snippet"]} for k, v in high_conf.items()})
    with col_b:
        if low_conf:
            st.warning(f"Low/Missing ({len(low_conf)})")
            st.json({k: {"value": v["value"], "source": v["source_snippet"]} for k, v in low_conf.items()})
        else:
            st.balloons()

    # Step 4: Replace in Document (Preserves Formatting)
    def replace_in_paragraph(paragraph, key, value):
        placeholder = f"[{key}]"
        if placeholder in paragraph.text:
            for run in paragraph.runs:
                if placeholder in run.text:
                    run.text = run.text.replace(placeholder, str(value["value"]))

    for p in doc.paragraphs:
        for key in placeholders:
            if key in extracted_data:
                replace_in_paragraph(p, key, extracted_data[key])

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for key in placeholders:
                        if key in extracted_data:
                            replace_in_paragraph(p, key, extracted_data[key])

    # Download
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)

    progress.progress(100)
    status.text("✅ RAG Complete! Form filled with grounded extractions.")

    st.download_button(
        label="📥 Download RAG-Filled Document",
        data=bio,
        file_name=f"RAG_Filled_{template_file.name.replace('.docx', '')}_{datetime.now().strftime('%Y%m%d')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary"
    )

    # Cleanup (optional: persist Chroma if needed)
    # vectorstore.delete_collection()
