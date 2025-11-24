import streamlit as st
from docx import Document
import io
import re
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Any

import pdfplumber
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

try:
    import google.generativeai as genai
except Exception:
    genai = None

# --- Page Config ---
st.set_page_config(page_title="Ultra-Accurate RAG AI Document Filler", layout="wide", page_icon="🤖")
st.title("🤖 Ultra-Accurate RAG AI Document Filler with Chunked Context")
st.markdown("Every placeholder is extracted using context-aware AI. Works well with PDFs, scans, images.")

# --- Sidebar ---
with st.sidebar:
    st.header("Gemini Config")
    api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY",""))
    model_choice = st.selectbox("Model", ["gemini-2.5-flash","gemini-2.5-pro"], index=0)
    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512)
    top_k = st.slider("Top-K Chunks per Field", 1, 5, 3)

# --- Upload ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("Template (.docx)", type="docx")
with col2:
    source_files = st.file_uploader("Source Files (PDF/Images)", accept_multiple_files=True, type=["pdf","png","jpg","jpeg","tiff","bmp","heic"])

if st.button("🚀 Extract & Fill"):
    if not api_key or not template_file or not source_files:
        st.error("Provide API key, template, and sources!")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    # --- Extract placeholders ---
    status.text("🔍 Reading template placeholders...")
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
    progress.progress(5)

    # --- Extract text from sources ---
    status.text("📚 Extracting text from sources...")
    all_text: List[str] = []
    def ocr_image(path): return pytesseract.image_to_string(Image.open(path))
    def extract_pdf(path):
        text=[]
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text.append(page.extract_text() or "")
        except: pass
        joined = "\n".join([t for t in text if t])
        if len(joined.strip())<50:
            try:
                pages = convert_from_path(path)
                joined = "\n".join([pytesseract.image_to_string(p) for p in pages])
            except: pass
        return joined

    for file in source_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        try:
            if file.name.lower().endswith(".pdf"): text = extract_pdf(tmp_path)
            else: text = ocr_image(tmp_path)
            if text.strip(): all_text.append(f"--- {file.name} ---\n{text}")
        finally:
            try: os.unlink(tmp_path)
            except: pass

    full_context = "\n\n".join(all_text)
    progress.progress(15)

    # --- Chunking ---
    def chunk_text(text:str, size:int)->List[str]:
        words = text.split()
        chunks=[]
        for i in range(0,len(words),size):
            chunks.append(" ".join(words[i:i+size]))
        return chunks
    context_chunks = chunk_text(full_context, chunk_size)

    # --- LLM extraction per field using top-K chunks ---
    status.text("🤖 Extracting fields via AI with top-K context...")
    extracted_data: Dict[str, Any] = {}
    for i, field in enumerate(placeholders):
        # Rank chunks by keyword overlap
        tokens = [t.lower() for t in re.findall(r"\w+", field) if len(t)>2]
        ranked_chunks = sorted(context_chunks, key=lambda c: sum(t in c.lower() for t in tokens), reverse=True)
        top_chunks = "\n\n".join(ranked_chunks[:top_k])

        prompt = f"Extract the best value for [{field}] from context below. Respond ONLY in JSON:\n{{\"value\":\"\",\"confidence\":0.0,\"source_snippet\":\"\"}}\n\nContext:\n{top_chunks[:8000]}"
        try:
            genai.configure(api_key=api_key)
            resp = genai.generate(model=model_choice, prompt=prompt) if hasattr(genai,"generate") else genai.chat.create(model=model_choice,messages=[{"role":"user","content":prompt}])
            text = getattr(resp,"text","") or str(resp)
            try: parsed = json.loads(text)
            except: parsed = {"value": text.strip(),"confidence":0.0,"source_snippet": text.strip()}
            extracted_data[field] = parsed
        except Exception as e:
            extracted_data[field] = {"value":"Not Found","confidence":0.0,"source_snippet":f"LLM error: {e}"}

        progress.progress(min(15+int(80*(i+1)/len(placeholders)),100))
        status.text(f"Processed {i+1}/{len(placeholders)}: {field}")

    # --- Replace placeholders in document ---
    def replace_in_paragraph(p,key,val):
        for run in p.runs: run.text = run.text.replace(f"[{key}]", str(val))
    for p in doc.paragraphs:
        for key in placeholders:
            replace_in_paragraph(p,key,extracted_data[key]["value"])
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for key in placeholders:
                        replace_in_paragraph(p,key,extracted_data[key]["value"])

    # --- Display results ---
    st.subheader("Extracted Data")
    high_conf = {k:v for k,v in extracted_data.items() if v.get("confidence",0)>=0.7}
    low_conf = {k:v for k,v in extracted_data.items() if v.get("confidence",0)<0.7}
    col_a,col_b = st.columns(2)
    with col_a:
        st.success(f"High Confidence ({len(high_conf)})")
        st.json({k:{"value":v["value"],"source":v["source_snippet"]} for k,v in high_conf.items()})
    with col_b:
        if low_conf:
            st.warning(f"Low/Missing ({len(low_conf)})")
            st.json({k:{"value":v["value"],"source":v["source_snippet"]} for k,v in low_conf.items()})
        else: st.balloons()

    # --- Download ---
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    progress.progress(100)
    status.text("✅ Extraction & Fill Complete!")
    st.download_button("📥 Download Filled Document", data=bio,
                       file_name=f"Filled_{template_file.name.replace('.docx','')}_{datetime.now().strftime('%Y%m%d')}.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
