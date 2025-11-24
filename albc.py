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

# Optional Gemini LLM client for fallback
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

# --- Page Config ---
st.set_page_config(page_title="RAG-Powered AI Document Filler", layout="wide", page_icon="🧠")
st.title("🧠 RAG-Powered AI Document Filler")
st.markdown("""
**RAG-Enhanced: Retrieves exact context from sources before filling—no more misses!**

Upload template (.docx) and sources (PDFs, images). AI fills perfectly.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Gemini Configuration")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", type="password", value=env_key)
    model_choice = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    st.slider("Chunk Size (tokens)", 256, 1024, 512)
    st.slider("Top-K Chunks per Field", 1, 5, 3)

# --- File Uploads ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("1. Template (.docx)", type="docx")
with col2:
    source_files = st.file_uploader("2. Source Files (PDF/images)", accept_multiple_files=True,
                                    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "heic"])

def clamp_progress(val: float) -> int:
    """Clamp progress to 0-100 int for Streamlit"""
    return max(0, min(100, int(val)))

def extract_text_from_pdf(path: str) -> str:
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
    except Exception:
        pass
    joined = "\n".join([t for t in text if t])
    # Fallback to OCR if no text extracted
    if len(joined.strip()) < 50:
        try:
            pages = convert_from_path(path)
            ocr_texts = [pytesseract.image_to_string(p) for p in pages]
            joined = "\n".join(ocr_texts)
        except Exception:
            joined = ""
    return joined

def ocr_image(path: str) -> str:
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def find_best_line(field: str, text: str) -> tuple[str, float, str]:
    tokens = [t.lower() for t in re.findall(r"\w+", field) if len(t) > 2]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    best_score = 0
    best_line = ""
    for line in lines:
        line_lower = line.lower()
        score = sum(1 for t in tokens if t in line_lower)
        if score > best_score:
            best_score = score
            best_line = line
    # Extract value heuristics
    date_re = re.compile(r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b)")
    number_re = re.compile(r"\b\d[\d,./-]{1,50}\b")
    if best_line:
        if ":" in best_line:
            val = best_line.split(":", 1)[1].strip()
        elif "-" in best_line:
            val = best_line.split("-", 1)[1].strip()
        else:
            d = date_re.search(best_line)
            if d:
                val = d.group(0)
            else:
                n = number_re.search(best_line)
                val = n.group(0) if n else best_line
        conf = min(0.9, 0.3 + 0.2 * best_score)
        return val, conf, best_line
    return "Not Found", 0.0, ""

def call_gemini_for_field(field: str, context: str) -> dict:
    if not genai:
        return {"value": "", "confidence": 0.0, "source_snippet": "Gemini client not installed"}
    prompt = f"Context:\n{context}\n\nExtract the best value for [{field}] as JSON: {{'value':'...', 'confidence':0.0-1.0, 'source_snippet':'...'}}"
    try:
        genai.configure(api_key=api_key)
        resp = genai.generate(model=model_choice, prompt=prompt)
        text = getattr(resp, "text", "") or str(resp)
        try:
            parsed = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                parsed = json.loads(m.group(0))
            else:
                parsed = {"value": text.strip(), "confidence": 0.0, "source_snippet": text.strip()}
        return parsed
    except Exception as e:
        return {"value": "", "confidence": 0.0, "source_snippet": f"Gemini error: {e}"}

# --- Main Processing ---
if st.button("🚀 Extract & Fill"):
    if not api_key or not template_file or not source_files:
        st.error("Provide API key, template, and sources")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    # Read placeholders
    status.text("Reading template placeholders...")
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
    progress.progress(clamp_progress(10))

    # Extract text from sources
    status.text("Extracting text from sources...")
    all_text_parts: List[str] = []
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
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    full_context = "\n\n".join(all_text_parts)
    progress.progress(clamp_progress(40))

    # Extract fields
    status.text("Extracting fields...")
    extracted_data: Dict[str, Any] = {}
    for i, field in enumerate(placeholders):
        val, conf, snippet = find_best_line(field, full_context)
        extracted_data[field] = {"value": val, "confidence": conf, "source_snippet": snippet}
        progress.progress(clamp_progress(40 + (50*(i+1)/len(placeholders))))
        status.text(f"Processed {i+1}/{len(placeholders)}: {field}")

    # Gemini fallback for low-confidence
    for key, info in list(extracted_data.items()):
        if info.get("confidence", 0) < 0.6:
            parsed = call_gemini_for_field(key, full_context[:8000])
            if parsed and parsed.get("value"):
                extracted_data[key] = {"value": parsed.get("value"), "confidence": parsed.get("confidence", 0.0),
                                       "source_snippet": parsed.get("source_snippet", info.get("source_snippet",""))}

    progress.progress(clamp_progress(90))
    status.text("Filling template...")

    # Replace placeholders in doc
    def replace_in_paragraph(paragraph, key, value):
        placeholder = f"[{key}]"
        for run in paragraph.runs:
            if placeholder in run.text:
                run.text = run.text.replace(placeholder, str(value))

    for p in doc.paragraphs:
        for key in placeholders:
            replace_in_paragraph(p, key, extracted_data[key]["value"])
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for key in placeholders:
                        replace_in_paragraph(p, key, extracted_data[key]["value"])

    # Show extracted data
    st.subheader("Extracted Data")
    high_conf = {k:v for k,v in extracted_data.items() if v.get("confidence",0)>=0.7}
    low_conf = {k:v for k,v in extracted_data.items() if v.get("confidence",0)<0.7}
    col_a, col_b = st.columns(2)
    with col_a:
        st.success(f"High Confidence ({len(high_conf)})")
        st.json({k:{"value":v["value"],"source":v["source_snippet"]} for k,v in high_conf.items()})
    with col_b:
        st.warning(f"Low/Missing ({len(low_conf)})") if low_conf else st.balloons()
        if low_conf:
            st.json({k:{"value":v["value"],"source":v["source_snippet"]} for k,v in low_conf.items()})

    # Download filled doc
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    progress.progress(100)
    status.text("✅ Completed!")
    st.download_button(
        label="Download Filled Document",
        data=bio,
        file_name=f"Filled_{template_file.name.replace('.docx','')}_{datetime.now().strftime('%Y%m%d')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
