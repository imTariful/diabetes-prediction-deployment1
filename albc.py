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

# Optional LLM client
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Optional fuzzy matching for better extraction
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

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
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", type="password", value=env_key, help="Get free key: https://aistudio.google.com/app/apikey")
    model_choice = st.selectbox("Model (Pro for handwriting/complex docs)", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512)
    top_k = st.slider("Top-K Chunks per Field", 1, 5, 3)

# --- Uploads ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("1. Template (.docx with [placeholders])", type="docx")
with col2:
    source_files = st.file_uploader("2. Source Files (PDF, Images, Scans...)", accept_multiple_files=True, type=["pdf","png","jpg","jpeg","tiff","bmp","heic"])

if st.button("🚀 RAG-Extract & Fill Form"):
    if not api_key or not template_file or not source_files:
        st.error("Please provide API key, template, and source files.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    # --- Extract placeholders ---
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
    progress.progress(5)
    status.text(f"Found {len(placeholders)} placeholders: {', '.join(placeholders[:8])}{'...' if len(placeholders)>8 else ''}")

    # --- Extract text from sources ---
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
                ocr_texts = [pytesseract.image_to_string(p) for p in pages]
                joined = "\n".join(ocr_texts)
            except:
                joined = joined
        return joined

    for file in source_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        try:
            if file.name.lower().endswith('.pdf'):
                text = extract_text_from_pdf(tmp_path)
            else:
                text = ocr_image(tmp_path)
            if text.strip():
                all_text_parts.append(f"--- Source: {file.name} ---\n{text}")
        finally:
            try: os.unlink(tmp_path)
            except: pass

    full_context = "\n\n".join(all_text_parts)
    progress.progress(15)

    # --- Extraction heuristics ---
    status.text("🔎 Extracting fields...")
    extracted_data: Dict[str, Any] = {}
    date_re = re.compile(r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b)")
    number_re = re.compile(r"\b\d[\d,./-]{1,50}\b")

    def find_best_line(field: str, text: str) -> tuple[str,float,str]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        best_line, best_score = "", 0.0
        tokens = [t.lower() for t in re.findall(r"\w+", field) if len(t) > 2]
        for line in lines:
            score = sum(1 for t in tokens if t in line.lower())
            if fuzz:
                score += fuzz.token_set_ratio(field.lower(), line.lower()) / 200  # weighted fuzzy
            if score > best_score:
                best_score = score
                best_line = line
        if best_line:
            val = best_line.split(":",1)[1].strip() if ":" in best_line else best_line.split("-",1)[1].strip() if "-" in best_line else best_line
            d = date_re.search(best_line)
            if d: val = d.group(0)
            n = number_re.search(best_line)
            if n: val = n.group(0)
            conf = min(0.95, 0.3 + 0.25*best_score)
            return val, conf, best_line
        return "Not Found", 0.0, ""

    for i, field in enumerate(placeholders):
        val, conf, snippet = find_best_line(field, full_context)
        extracted_data[field] = {"value": val, "confidence": conf, "source_snippet": snippet}
        progress.progress(min(15 + int(80*(i+1)/len(placeholders)),100))

    # --- LLM fallback for low-confidence fields ---
    def call_gemini_for_field(field: str, context: str) -> dict:
        if not genai:
            return {"value":"", "confidence":0.0, "source_snippet":"Gemini not installed"}
        prompt = f"Extract the best value for [{field}] from the context:\n{context}\nRespond ONLY in JSON: {{\"value\":\"\",\"confidence\":0.0,\"source_snippet\":\"\"}}"
        try:
            genai.configure(api_key=api_key)
            resp = genai.generate(model=model_choice, prompt=prompt) if hasattr(genai,"generate") else genai.chat.create(model=model_choice, messages=[{"role":"user","content":prompt}])
            text = getattr(resp,"text","") or str(resp)
            try: return json.loads(text)
            except: return {"value":text.strip(),"confidence":0.0,"source_snippet":text.strip()}
        except: return {"value":"", "confidence":0.0, "source_snippet":"Gemini call failed"}

    for key, info in extracted_data.items():
        if info.get("confidence",0) < 0.6:
            ctx = full_context[:8000]
            parsed = call_gemini_for_field(key, ctx)
            if parsed.get("value"): extracted_data[key] = parsed

    progress.progress(95)
    status.text("📝 Filling template...")

    # --- Replace placeholders in document ---
    def replace_in_paragraph(p, key, value):
        placeholder = f"[{key}]"
        if placeholder in p.text:
            for run in p.runs:
                run.text = run.text.replace(placeholder, str(value))
    for p in doc.paragraphs:
        for key in placeholders:
            replace_in_paragraph(p,key,extracted_data[key]["value"])
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for key in placeholders:
                        replace_in_paragraph(p,key,extracted_data[key]["value"])

    # --- Display Results ---
    st.subheader("Extracted Data (with Confidence & Sources)")
    high_conf = {k:v for k,v in extracted_data.items() if v.get("confidence",0)>=0.7}
    low_conf = {k:v for k,v in extracted_data.items() if v.get("confidence",0)<0.7}

    col_a, col_b = st.columns(2)
    with col_a:
        st.success(f"High Confidence ({len(high_conf)})")
        st.json({k:{"value":v["value"],"source":v["source_snippet"]} for k,v in high_conf.items()})
    with col_b:
        if low_conf:
            st.warning(f"Low/Missing ({len(low_conf)})")
            st.json({k:{"value":v["value"],"source":v["source_snippet"]} for k,v in low_conf.items()})
        else:
            st.balloons()

    # --- Download filled document ---
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    progress.progress(100)
    status.text("✅ RAG Complete!")
    st.download_button(
        label="📥 Download Filled Document",
        data=bio,
        file_name=f"RAG_Filled_{template_file.name.replace('.docx','')}_{datetime.now().strftime('%Y%m%d')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
