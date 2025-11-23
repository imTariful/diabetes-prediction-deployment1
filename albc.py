import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract
import re
import json
import requests
from docx import Document

st.set_page_config(page_title="GLR - Insurance Auto-fill", layout="wide")


# ============================
# PDF TEXT EXTRACTION
# ============================
def extract_text_from_pdf_bytes(pdf_bytes, ocr_lang="eng"):
    text_parts = []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_parts.append(page.get_text("text"))
        extracted = "\n".join(text_parts).strip()
    except:
        extracted = ""

    # If no text → fallback to OCR
    if len(extracted) < 50:
        images = convert_from_bytes(pdf_bytes)
        ocr_texts = []
        for img in images:
            ocr_texts.append(pytesseract.image_to_string(img, lang=ocr_lang))
        extracted = "\n".join(ocr_texts)

    return extracted


# ============================
# DOCX PLACEHOLDER HANDLING
# ============================
PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z0-9_\- ]+)\s*\}\}")

def find_placeholders(doc):
    placeholders = set()

    for p in doc.paragraphs:
        for m in PLACEHOLDER_PATTERN.findall(p.text):
            placeholders.add(m)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for m in PLACEHOLDER_PATTERN.findall(cell.text):
                    placeholders.add(m)

    return list(placeholders)


def replace_placeholders(doc, mapping):
    for p in doc.paragraphs:
        if PLACEHOLDER_PATTERN.search(p.text):
            full_text = p.text
            for key, val in mapping.items():
                full_text = re.sub(r"\{\{\s*" + re.escape(key) + r"\s*\}\}", val, full_text)
            p.text = full_text

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if PLACEHOLDER_PATTERN.search(cell.text):
                    cell_text = cell.text
                    for key, val in mapping.items():
                        cell_text = re.sub(r"\{\{\s*" + re.escape(key) + r"\s*\}\}", val, cell_text)
                    cell.text = cell_text

    return doc


# ============================
# CALL OPENROUTER LLM
# ============================
def ask_llm(placeholders, text):
    api_key = st.secrets["OPENROUTER_API_KEY"]

    system_prompt = (
        "You extract values for insurance report placeholders from PDF text. "
        "Return STRICT JSON {placeholder: value}. Missing = ''. No explanation."
    )

    user_prompt = (
        "Placeholders:\n" + json.dumps(placeholders) +
        "\n\nExtracted Text:\n" + text[:20000] +
        "\n\nReturn JSON only."
    )

    body = {
        "model": "deepseek-r1:free",  # Free OpenRouter model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 800,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        "https://api.openrouter.ai/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=60,
    )

    result = resp.json()
    raw = result["choices"][0]["message"]["content"]

    # Extract JSON
    try:
        parsed = json.loads(raw)
    except:
        m = re.search(r"\{[\s\S]*\}", raw)
        parsed = json.loads(m.group(0)) if m else {}

    # Guarantee all placeholders exist
    return {p: parsed.get(p, "") for p in placeholders}


# ============================
# STREAMLIT UI
# ============================
st.title("📄 GLR Pipeline — Insurance Auto-Fill (OpenRouter LLM)")
st.markdown("Upload a **DOCX template** and **PDF reports**. This app auto-fills the template using LLM extraction.")

docx_file = st.file_uploader("Upload template (.docx)", type=["docx"])
pdf_files = st.file_uploader("Upload photo reports (.pdf)", type=["pdf"], accept_multiple_files=True)
ocr_lang = st.text_input("OCR language (default: eng)", "eng")

if st.button("Generate Filled Document"):
    if not docx_file:
        st.error("Upload a .docx template.")
        st.stop()
    if not pdf_files:
        st.error("Upload at least one PDF.")
        st.stop()

    with st.spinner("Extracting PDF text..."):
        all_text = ""
        for pdf in pdf_files:
            extracted = extract_text_from_pdf_bytes(pdf.read(), ocr_lang)
            all_text += f"\n---- {pdf.name} ----\n{extracted}\n"

    template = Document(docx_file)
    placeholders = find_placeholders(template)

    st.info(f"Found placeholders: {placeholders}")

    with st.spinner("Asking LLM to fill fields..."):
        mapping = ask_llm(placeholders, all_text)

    st.success("LLM extracted the field values.")
    st.json(mapping)

    filled_doc = replace_placeholders(template, mapping)

    buffer = BytesIO()
    filled_doc.save(buffer)
    buffer.seek(0)

    st.download_button(
        "Download Filled Template",
        buffer,
        "filled_template.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
