import streamlit as st
from docx import Document
import io
import json
import re
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, List

import pdfplumber
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# FAISS for RAG
import faiss
import numpy as np

# Optional Gemini LLM client
try:
    import google.generativeai as genai
except Exception:
    genai = None

# --- Page Config ---
st.set_page_config(page_title="RAG-Powered AI Document Filler", layout="wide", page_icon="🧠")
st.title("🧠 RAG-Powered AI Document Filler")
st.markdown("""
**High-Accuracy RAG Document Filler**

Uses embeddings + LLM + OCR to extract and fill template fields from PDFs/images.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Gemini Configuration")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", type="password", value=env_key)
    model_choice = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512)
    top_k = st.slider("Top-K Chunks per Field", 1, 5, 3)

# --- Uploads ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("Template (.docx)", type="docx")
with col2:
    source_files = st.file_uploader("Source Files (PDF/Images)", accept_multiple_files=True,
                                    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "heic"])

# --- Functions ---
def ocr_image(path: str) -> str:
    img = Image.open(path).convert("L")
    return pytesseract.image_to_string(img)

def extract_text_from_pdf(path: str) -> str:
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
    except Exception:
        pass
    joined = "\n".join([t for t in text if t])
    if len(joined.strip()) < 50:
        try:
            pages = convert_from_path(path, dpi=300)
            ocr_texts = [pytesseract.image_to_string(p) for p in pages]
            joined = "\n".join(ocr_texts)
        except Exception:
            joined = joined
    return joined

def split_text_chunks(text: str, chunk_size: int = 512) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def embed_text(text_list: List[str]) -> np.ndarray:
    # Simple Gemini embedding
    if not genai:
        raise ValueError("Gemini client not installed")
    genai.configure(api_key=api_key)
    embeddings = []
    for txt in text_list:
        emb_resp = genai.generate_embeddings(model="models/embedding-001", text=[txt])
        embeddings.append(np.array(emb_resp.data[0].embedding, dtype="float32"))
    return np.vstack(embeddings)

def call_gemini_for_field(field: str, context: str) -> dict:
    prompt = (
        f"Extract the best value for the template field [{field}] from the context below.\n"
        f"Context:\n{context}\n"
        "Respond ONLY with JSON: {\"value\": \"...\", \"confidence\": 0.0-1.0, \"source_snippet\": \"...\"}"
    )
    try:
        genai.configure(api_key=api_key)
        if hasattr(genai, "generate"):
            resp = genai.generate(model=model_choice, prompt=prompt)
            text = getattr(resp, "text", "") or getattr(resp, "output", "")
        else:
            resp = genai.chat.create(model=model_choice, messages=[{"role": "user", "content": prompt}])
            text = getattr(resp, "last", {}).get("content", str(resp))
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                return json.loads(m.group(0))
            else:
                return {"value": text.strip(), "confidence": 0.0, "source_snippet": text.strip()}
    except Exception as e:
        return {"value": "", "confidence": 0.0, "source_snippet": f"LLM error: {e}"}

def replace_in_paragraph(paragraph, key, value):
    placeholder = f"[{key}]"
    if placeholder in paragraph.text:
        for run in paragraph.runs:
            if placeholder in run.text:
                run.text = run.text.replace(placeholder, str(value))

# --- Main Process ---
if st.button("🚀 Extract & Fill"):

    if not api_key or not template_file or not source_files:
        st.error("Provide API key, template, and source files.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    # Load template
    status.text("🔍 Extracting placeholders...")
    doc = Document(template_file)
    placeholder_pattern = r'\[([^\]]+)\]'
    placeholders = set()
    for p in doc.paragraphs:
        placeholders.update(re.findall(placeholder_pattern, p.text))
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    placeholders.update(re.findall(placeholder_pattern, p.text))
    placeholders = sorted([p.strip() for p in placeholders if p.strip()])
    if not placeholders:
        st.warning("No placeholders found!")
        st.stop()
    progress.progress(10)

    # Extract text from sources
    status.text("📚 Extracting text from sources...")
    all_texts = []
    for file in source_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        try:
            if file.name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(tmp_path)
            else:
                text = ocr_image(tmp_path)
            if text.strip():
                all_texts.append(text)
        finally:
            os.unlink(tmp_path)
    full_text = "\n".join(all_texts)
    progress.progress(30)

    # Chunk + Embed + FAISS
    status.text("🔗 Embedding and building vector store...")
    chunks = split_text_chunks(full_text, chunk_size)
    embeddings = embed_text(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    progress.progress(50)

    # Extract each field
    status.text("🔄 Extracting fields using RAG + LLM...")
    extracted_data = {}
    for i, field in enumerate(placeholders):
        # Retrieve top-k chunks
        query_emb = embed_text([field])[0].reshape(1, -1)
        D, I = index.search(query_emb, top_k)
        top_chunks = [chunks[idx] for idx in I[0] if idx < len(chunks)]
        context = "\n".join(top_chunks)
        result = call_gemini_for_field(field, context)
        extracted_data[field] = result
        progress.progress(50 + int(40 * (i + 1) / len(placeholders)))
        status.text(f"Processed {i+1}/{len(placeholders)}: {field}")

    progress.progress(90)
    status.text("📝 Filling template...")

    # Replace placeholders in DOCX
    for p in doc.paragraphs:
        for k, v in extracted_data.items():
            replace_in_paragraph(p, k, v.get("value", "Not Found"))
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for k, v in extracted_data.items():
                        replace_in_paragraph(p, k, v.get("value", "Not Found"))

    # Preview
    st.subheader("Extracted Data")
    st.json(extracted_data)

    # Download
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    progress.progress(100)
    status.text("✅ Done! Document ready for download.")
    st.download_button(
        label="📥 Download Filled Document",
        data=bio,
        file_name=f"RAG_Filled_{template_file.name.replace('.docx','')}_{datetime.now().strftime('%Y%m%d')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
