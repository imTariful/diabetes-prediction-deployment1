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
st.set_page_config(page_title="RAG-Powered AI Document Filler", layout="wide", page_icon="🧠")
st.title("🧠 RAG-Powered AI Document Filler")
st.markdown("""
**Upload a template (.docx) and source files (PDF/Images).**
RAG retrieves exact context and fills placeholders automatically.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Gemini Configuration")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", type="password", value=env_key)
    model_choice = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512)
    top_k = st.slider("Top-K lines per field", 1, 5, 3)

# --- Uploads ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("Template (.docx)", type="docx")
with col2:
    source_files = st.file_uploader("Source files (PDF/Images)", type=["pdf","png","jpg","jpeg","tiff","bmp","heic"], accept_multiple_files=True)

if st.button("🚀 Extract & Fill Form"):
    if not api_key or not template_file or not source_files:
        st.error("Please provide API key, template, and sources.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    # --- Step 1: Extract placeholders ---
    status.text("🔍 Reading placeholders...")
    doc = Document(template_file)
    placeholder_pattern = r"\[([^\]]+)\]"
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
    progress.progress(5)

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
                    page_text = page.extract_text() or ""
                    text.append(page_text)
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
    progress.progress(20)

    # --- Step 3: Field extraction ---
    status.text("🔄 Extracting fields...")
    extracted_data: Dict[str, Any] = {}

    date_re = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
    number_re = re.compile(r"\b\d[\d,./-]{1,50}\b")

    def find_best_line(field: str, text: str, top_k_lines:int=3) -> tuple[str,float,str]:
        tokens = [t.lower() for t in re.findall(r"\w+", field) if len(t)>2]
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        scored_lines = []
        for line in lines:
            score = sum(1 for t in tokens if t in line.lower())
            scored_lines.append((score, line))
        scored_lines.sort(reverse=True)
        best_score, best_line = scored_lines[0] if scored_lines else (0, "")
        for score, line in scored_lines[:top_k_lines]:
            if score > best_score:
                best_score, best_line = score, line
        # Extract value heuristically
        if ":" in best_line: val = best_line.split(":",1)[1].strip()
        elif "-" in best_line: val = best_line.split("-",1)[1].strip()
        else:
            d = date_re.search(best_line)
            val = d.group(0) if d else (number_re.search(best_line).group(0) if number_re.search(best_line) else best_line)
        conf = min(0.95, 0.3 + 0.2*best_score)
        return val or "Not Found", conf, best_line

    for i, field in enumerate(placeholders):
        val, conf, snippet = find_best_line(field, full_context, top_k)
        extracted_data[field] = {"value": val, "confidence": conf, "source_snippet": snippet}
        progress.progress(20 + int(60*(i+1)/len(placeholders)))

    # --- Step 4: LLM fallback ---
    def call_gemini_for_field(field:str, context:str)->dict:
        if not genai:
            return {"value":"Not Found","confidence":0.0,"source_snippet":"Gemini not installed"}
        prompt = f"Extract [{field}] from context below. Respond ONLY as JSON {{'value':'','confidence':0.0,'source_snippet':''}}.\n\nContext:\n{context[:8000]}"
        try:
            genai.configure(api_key=api_key)
            resp = genai.generate(model=model_choice, prompt=prompt)
            text = getattr(resp,"text","") or str(resp)
            try: parsed = json.loads(text)
            except: parsed = {"value": text.strip(), "confidence":0.0, "source_snippet": text.strip()}
            return parsed
        except Exception as e:
            return {"value":"Not Found","confidence":0.0,"source_snippet": f"LLM error: {e}"}

    llm_calls = 0
    for key, info in extracted_data.items():
        if info.get("confidence",0)<0.6:
            parsed = call_gemini_for_field(key, full_context)
            if parsed.get("value"):
                extracted_data[key] = parsed
            llm_calls += 1
    if llm_calls: status.text(f"Called Gemini for {llm_calls} low-confidence fields")
    progress.progress(85)

    # --- Step 5: Fill template ---
    status.text("📝 Filling template...")
    def replace_in_paragraph(paragraph,key,value):
        placeholder=f"[{key}]"
        if placeholder in paragraph.text:
            for run in paragraph.runs:
                run.text = run.text.replace(placeholder,str(value))

    for p in doc.paragraphs:
        for key in placeholders:
            replace_in_paragraph(p,key,extracted_data[key]["value"])
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for key in placeholders:
                        replace_in_paragraph(p,key,extracted_data[key]["value"])

    progress.progress(95)
    status.text("✅ Completed!")

    # --- Step 6: Show extracted data ---
    st.subheader("Extracted Data (Confidence & Source)")
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

    # --- Step 7: Download filled doc ---
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    st.download_button(
        label="📥 Download Filled Document",
        data=bio,
        file_name=f"Filled_{template_file.name.replace('.docx','')}_{datetime.now().strftime('%Y%m%d')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    progress.progress(100)
