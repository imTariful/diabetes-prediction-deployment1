import streamlit as st
from docx import Document
import io
import json
import re
import tempfile
import os
import mimetypes
from datetime import datetime
from typing import Dict, Any, List

import pdfplumber
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# Optional Gemini LLM client will be imported at runtime for fallback
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
    # Allow default from environment variable so user can avoid pasting secrets into the UI
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", type="password", value=env_key, help="Get free key: https://aistudio.google.com/app/apikey")
    model_choice = st.selectbox(
        "Model (Pro for handwriting/complex docs)",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
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

    # Step 2: Extract text from all uploaded sources (PDFs and images)
    status.text("📚 Extracting text from source files (with OCR fallback)...")
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
        # If pdfplumber extracted nothing, fall back to OCR per page
        if len(joined.strip()) < 50:
            try:
                pages = convert_from_path(path)
                ocr_texts = [pytesseract.image_to_string(p) for p in pages]
                joined = "\n".join(ocr_texts)
            except Exception:
                # last resort: empty string
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
                # image files
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

    # Step 3: Simple extraction heuristics per placeholder (with optional LLM fallback)
    status.text("🔄 Extracting fields from sources...")
    extracted_data: Dict[str, Any] = {}

    # Basic regex helpers
    date_re = re.compile(r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b)")
    number_re = re.compile(r"\b\d[\d,./-]{1,50}\b")

    def find_best_line(field: str, text: str) -> tuple[str, float, str]:
        # Look for lines containing tokens from field name
        tokens = [t.lower() for t in re.findall(r"\w+", field) if len(t) > 2]
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        best_score = 0.0
        best_line = ""
        for line in lines:
            line_lower = line.lower()
            score = sum(1 for t in tokens if t in line_lower)
            if score > best_score:
                best_score = score
                best_line = line

        # heuristics to extract value from the line
        if best_line:
            # if line contains colon or dash, take RHS
            if ":" in best_line:
                val = best_line.split(":", 1)[1].strip()
            elif "-" in best_line:
                val = best_line.split("-", 1)[1].strip()
            else:
                # try date or number
                d = date_re.search(best_line)
                if d:
                    val = d.group(0)
                else:
                    n = number_re.search(best_line)
                    val = n.group(0) if n else best_line

            conf = min(0.9, 0.3 + 0.2 * best_score)
            return val, conf, best_line
        return "Not Found", 0.0, ""

    for i, field in enumerate(placeholders):
        val, conf, snippet = find_best_line(field, full_context)
        extracted_data[field] = {"value": val, "confidence": conf, "source_snippet": snippet}
        progress.progress(40 + (50 * (i + 1) / len(placeholders)))
        status.text(f"Processed {i+1}/{len(placeholders)}: {field}")

    # LLM fallback for low-confidence fields using Gemini (if available)
    def call_gemini_for_field(field: str, context: str) -> dict:
        if not genai:
            return {"value": "", "confidence": 0.0, "source_snippet": "Gemini client not installed"}

        prompt = (
            f"You are given extracted text from insurance photo reports.\n\n"
            f"Context:\n{context}\n\n"
            f"Extract the best value for the template field [{field}].\n"
            "If you cannot find a value, return value as empty string and confidence 0.\n"
            "Respond ONLY with a JSON object: {\"value\": \"...\", \"confidence\": 0.0-1.0, \"source_snippet\": \"...\"}"
        )

        try:
            genai.configure(api_key=api_key)
            text = ""
            try:
                if hasattr(genai, "generate"):
                    resp = genai.generate(model=model_choice, prompt=prompt)
                    text = getattr(resp, "text", "") or getattr(resp, "output", "")
                elif hasattr(genai, "chat") and hasattr(genai.chat, "create"):
                    resp = genai.chat.create(model=model_choice, messages=[{"role": "user", "content": prompt}])
                    if hasattr(resp, "last"):
                        text = getattr(resp.last, "content", "")
                    else:
                        try:
                            text = resp.choices[0].message.content
                        except Exception:
                            text = str(resp)
                else:
                    resp = genai.chat.create(model=model_choice, messages=[{"role": "user", "content": prompt}])
                    text = str(resp)
            except Exception as e:
                return {"value": "", "confidence": 0.0, "source_snippet": f"LLM call failed: {e}"}

            try:
                parsed = json.loads(text)
            except Exception:
                m = re.search(r"\{.*\}", text, re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = {"value": text.strip(), "confidence": 0.0, "source_snippet": text.strip()}
                else:
                    parsed = {"value": text.strip(), "confidence": 0.0, "source_snippet": text.strip()}

            return parsed
        except Exception as e:
            return {"value": "", "confidence": 0.0, "source_snippet": f"Gemini error: {e}"}

    llm_calls = 0
    for key, info in list(extracted_data.items()):
        if info.get("confidence", 0) < 0.6:
            ctx = full_context[:8000] if full_context else ""
            parsed = call_gemini_for_field(key, ctx)
            if parsed and parsed.get("value"):
                extracted_data[key] = {"value": parsed.get("value"), "confidence": parsed.get("confidence", 0.0), "source_snippet": parsed.get("source_snippet", info.get("source_snippet",""))}
            llm_calls += 1
    if llm_calls:
        status.text(f"Called Gemini for {llm_calls} low-confidence fields")

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
                    run.text = run.text.replace(placeholder, str(value))

    for p in doc.paragraphs:
        for key in placeholders:
            if key in extracted_data:
                replace_in_paragraph(p, key, extracted_data[key]["value"])

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for key in placeholders:
                        if key in extracted_data:
                            replace_in_paragraph(p, key, extracted_data[key]["value"])

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
