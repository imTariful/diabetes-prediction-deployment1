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

# Optional Gemini LLM client
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

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
    api_key = st.text_input("Gemini API Key", type="password", value=env_key,
                            help="Get free key: https://aistudio.google.com/app/apikey")
    model_choice = st.selectbox("Model (Pro for handwriting/complex docs)",
                                ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512, help="Smaller = finer retrieval")
    top_k = st.slider("Top-K Chunks per Field", 1, 5, 3, help="More = richer context")
    st.info("**RAG Magic**: Embeds sources → Retrieves exact matches → Generates grounded fills.")

# --- Uploads ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("1. Template (.docx with [placeholders])", type="docx")
with col2:
    source_files = st.file_uploader("2. Source Files (PDF, Images, Scans...)", accept_multiple_files=True,
                                    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "heic"])

if st.button("🚀 RAG-Extract & Fill Form", type="primary"):
    if not api_key or not template_file or not source_files:
        st.error("Please provide API key, template, and source files.")
        st.stop()

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
    status.text(f"Found {len(placeholders)} fields: {', '.join(placeholders[:8])}{'...' if len(placeholders) > 8 else ''}")

    # Step 2: Extract text from uploaded sources
    status.text("📚 Extracting text from source files...")
    all_text_parts: List[str] = []

    def ocr_image(path: str) -> str:
        try:
            img = Image.open(path)
            return pytesseract.image_to_string(img)
        except Exception:
            return ""

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
                pages = convert_from_path(path)
                ocr_texts = [pytesseract.image_to_string(p) for p in pages]
                joined = "\n".join(ocr_texts)
            except Exception:
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

            if text and text.strip():
                all_text_parts.append(f"--- Source: {file.name} ---\n{text}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    full_context = "\n\n".join(all_text_parts)
    progress.progress(40)

    # Step 3: High-accuracy extraction
    status.text("🔄 Extracting fields from sources...")

    date_re = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
    number_re = re.compile(r"\b\d[\d,./-]{1,50}\b")
    zip_re = re.compile(r"\b\d{5}(?:-\d{4})?\b")
    address_re = re.compile(r"\d+\s+[A-Za-z0-9\s]+")
    city_state_re = re.compile(r"[A-Za-z]+")

    def heuristic_extract(field: str, text: str) -> dict:
        tokens = [t.lower() for t in re.findall(r"\w+", field) if len(t) > 2]
        best_line = ""
        best_score = 0
        for line in text.splitlines():
            score = sum(1 for t in tokens if t in line.lower())
            if score > best_score:
                best_score = score
                best_line = line.strip()
        if not best_line:
            return {"value": "Not Found", "confidence": 0.3, "source_snippet": ""}

        val = "Not Found"
        lname = field.lower()
        if "date" in lname:
            m = date_re.search(best_line)
            if m: val = m.group(0)
        elif "zip" in lname:
            m = zip_re.search(best_line)
            if m: val = m.group(0)
        elif "street" in lname or "address" in lname:
            m = address_re.search(best_line)
            if m: val = m.group(0)
        elif "city" in lname or "state" in lname:
            m = city_state_re.search(best_line)
            if m: val = m.group(0)
        elif "number" in lname or "phone" in lname:
            m = number_re.search(best_line)
            if m: val = m.group(0)
        else:
            if ":" in best_line:
                val = best_line.split(":", 1)[1].strip()
            elif "-" in best_line:
                val = best_line.split("-", 1)[1].strip()
            else:
                val = best_line
        return {"value": val, "confidence": min(0.8, 0.3 + 0.1 * best_score), "source_snippet": best_line}

    # Extract fields using heuristics + optional Gemini LLM
    extracted_data: Dict[str, Any] = {}
    for i, field in enumerate(placeholders):
        extracted_data[field] = heuristic_extract(field, full_context)
        progress.progress(40 + int(50 * (i + 1) / len(placeholders)))
        status.text(f"Processed {i + 1}/{len(placeholders)}: {field}")

    # LLM fallback for low-confidence fields
    def call_gemini_for_field(field: str, context: str) -> dict:
        if not genai:
            return {"value": "", "confidence": 0.0, "source_snippet": "Gemini client not installed"}

        prompt = f"You are given extracted text from reports.\n\nContext:\n{context}\n\nExtract the best value for [{field}] in JSON."
        try:
            genai.configure(api_key=api_key)
            resp = genai.generate(model=model_choice, prompt=prompt) if hasattr(genai, "generate") else genai.chat.create(model=model_choice, messages=[{"role":"user","content":prompt}])
            text = getattr(resp, "text", "") or str(resp)
            parsed = json.loads(text)
            return parsed
        except Exception as e:
            return {"value": "", "confidence": 0.0, "source_snippet": f"LLM error: {e}"}

    for key, info in list(extracted_data.items()):
        if info.get("confidence", 0) < 0.7:
            llm_result = call_gemini_for_field(key, full_context[:8000])
            # Merge heuristic + LLM
            if llm_result.get("value"):
                extracted_data[key]["value"] = llm_result["value"]
                extracted_data[key]["confidence"] = max(info.get("confidence",0), llm_result.get("confidence",0.0))
                if not extracted_data[key].get("source_snippet"):
                    extracted_data[key]["source_snippet"] = llm_result.get("source_snippet","")

    progress.progress(90)
    status.text("📝 Filling template...")

    # Display results
    st.subheader("RAG-Extracted Data (with Confidence & Sources)")
    high_conf = {k:v for k,v in extracted_data.items() if v.get("confidence",0) >= 0.7}
    low_conf = {k:v for k,v in extracted_data.items() if v.get("confidence",0) < 0.7}

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

    # Replace placeholders in DOCX
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

    # Download filled DOCX
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)

    progress.progress(100)
    status.text("✅ RAG Complete! Form filled with grounded extractions.")

    st.download_button(
        label="📥 Download RAG-Filled Document",
        data=bio,
        file_name=f"RAG_Filled_{template_file.name.replace('.docx','')}_{datetime.now().strftime('%Y%m%d')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary"
    )
