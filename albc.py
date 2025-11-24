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

# Fuzzy matching
from rapidfuzz import fuzz

# Optional Gemini LLM client
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

# --- Page Config ---
st.set_page_config(page_title="RAG-Powered AI Document Filler", layout="wide", page_icon="🧠")
st.title("🧠 RAG-Powered AI Document Filler")
st.markdown("""
**Improved Accuracy:** Fuzzy matching + LLM fallback for almost all fields.

Upload template (.docx) with [placeholders] and sources (PDFs, images, scans). AI fills automatically.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Gemini Configuration")
    api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    model_choice = st.selectbox("Model (Pro for handwriting/complex docs)", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512)
    top_k = st.slider("Top-K Chunks per Field", 1, 5, 3)

# --- Uploads ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("1. Template (.docx with [placeholders])", type="docx")
with col2:
    source_files = st.file_uploader("2. Source Files (PDF, Images, Scans...)", accept_multiple_files=True, type=["pdf","png","jpg","jpeg","tiff","bmp","heic"])

if st.button("🚀 Extract & Fill Form"):

    if not api_key or not template_file or not source_files:
        st.error("Provide API key, template, and source files.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    # --- Step 1: Extract placeholders ---
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
        st.warning("No placeholders found!")
        st.stop()
    progress.progress(10)
    status.text(f"Found {len(placeholders)} fields: {', '.join(placeholders[:8])}{'...' if len(placeholders)>8 else ''}")

    # --- Step 2: Extract text from sources ---
    status.text("📚 Extracting text from sources...")
    all_text_parts: List[str] = []

    def ocr_image(path: str) -> str:
        try:
            img = Image.open(path)
            return pytesseract.image_to_string(img)
        except:
            return ""

    def extract_text_from_pdf(path: str) -> str:
        text = []
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text.append(page.extract_text() or "")
        except:
            pass
        joined = "\n".join([t for t in text if t])
        if len(joined.strip()) < 50:
            try:
                pages = convert_from_path(path)
                joined = "\n".join([pytesseract.image_to_string(p) for p in pages])
            except:
                joined = joined
        return joined

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
                all_text_parts.append(f"--- Source: {file.name} ---\n{text}")
        finally:
            try: os.unlink(tmp_path)
            except: pass

    full_context = "\n\n".join(all_text_parts)
    progress.progress(40)

    # --- Step 3: Fuzzy matching + heuristics ---
    status.text("🔎 Extracting fields...")
    extracted_data: Dict[str, Any] = {}
    date_re = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
    number_re = re.compile(r"\b\d[\d,./-]{1,50}\b")

    def find_best_line(field, text):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        best_score = 0
        best_line = ""
        for line in lines:
            score = fuzz.partial_ratio(field.lower(), line.lower())
            if score > best_score:
                best_score = score
                best_line = line
        # Heuristic extraction
        val = "Not Found"
        if best_line:
            if ":" in best_line:
                val = best_line.split(":",1)[1].strip()
            elif "-" in best_line:
                val = best_line.split("-",1)[1].strip()
            else:
                d = date_re.search(best_line)
                n = number_re.search(best_line)
                val = d.group(0) if d else (n.group(0) if n else best_line)
        conf = min(0.95, 0.4 + 0.005*best_score)
        return val, conf, best_line

    for i, field in enumerate(placeholders):
        val, conf, snippet = find_best_line(field, full_context)
        extracted_data[field] = {"value": val, "confidence": conf, "source_snippet": snippet}
        progress.progress(40 + (50*(i+1)/len(placeholders)))
        status.text(f"Processed {i+1}/{len(placeholders)}: {field}")

    # --- Step 4: LLM fallback ---
    def call_gemini(field, context):
        if not genai: return {"value":"", "confidence":0.0, "source_snippet":"Gemini not installed"}
        prompt = f"Extract the best value for [{field}] from the context below. Respond ONLY in JSON {{'value':'...','confidence':0.0-1.0,'source_snippet':'...'}}\n\nContext:\n{context}"
        try:
            genai.configure(api_key=api_key)
            resp = genai.generate(model=model_choice, prompt=prompt) if hasattr(genai, "generate") else genai.chat.create(model=model_choice, messages=[{"role":"user","content":prompt}])
            text = getattr(resp, "text", str(resp))
            m = re.search(r"\{.*\}", text, re.S)
            parsed = json.loads(m.group(0)) if m else {"value": text.strip(), "confidence":0.0, "source_snippet": text.strip()}
            return parsed
        except:
            return {"value":"", "confidence":0.0, "source_snippet":"LLM call failed"}

    llm_calls = 0
    for key, info in list(extracted_data.items()):
        if info["confidence"] < 0.8 or info["value"]=="Not Found":
            ctx = full_context[:8000]
            parsed = call_gemini(key, ctx)
            if parsed.get("value"):
                extracted_data[key] = parsed
            llm_calls += 1
    if llm_calls: status.text(f"Called Gemini for {llm_calls} fields")

    progress.progress(90)
    status.text("📝 Filling template...")

    # --- Step 5: Replace in Document ---
    def replace_in_paragraph(paragraph, key, value):
        placeholder = f"[{key}]"
        for run in paragraph.runs:
            run.text = run.text.replace(placeholder, str(value))
    for p in doc.paragraphs:
        for key in placeholders:
            if key in extracted_data: replace_in_paragraph(p, key, extracted_data[key]["value"])
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for key in placeholders:
                        if key in extracted_data: replace_in_paragraph(p, key, extracted_data[key]["value"])

    # --- Step 6: Show and Download ---
    st.subheader("Extracted Data")
    high = {k:v for k,v in extracted_data.items() if v["confidence"]>=0.8}
    low = {k:v for k,v in extracted_data.items() if v["confidence"]<0.8}
    col_a, col_b = st.columns(2)
    with col_a: st.success(high)
    with col_b: st.warning(low if low else "All fields high confidence!")
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    progress.progress(100)
    status.text("✅ Completed!")
    st.download_button("📥 Download Filled Document", data=bio, file_name=f"Filled_{datetime.now().strftime('%Y%m%d')}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
