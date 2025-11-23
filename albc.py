import streamlit as st
import io
import json
import requests
from docx import Document
from PyPDF2 import PdfReader

# --- CONFIGURATION ---
st.set_page_config(
    page_title="GLR Insurance Pipeline",
    page_icon="📋",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(pdf_file):
    """Extracts text from a single PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
        return ""

def extract_text_from_docx_for_context(docx_file):
    doc = Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text for cell in row.cells if cell.text.strip()]
            if row_text:
                full_text.append(" | ".join(row_text))
    return "\n".join(full_text)

def query_gemini_flash(api_key, system_prompt, user_prompt):
    """Query Google Gemini 1.5 Flash model"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "model": "gemini-1.5-flash",
        "temperature": 0
    }
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta2/models/gemini-2.5-flash:generateMessage",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]
    except Exception as e:
        st.warning("Gemini API failed, trying OpenRouter fallback...")
        return None

def query_openrouter(api_key, system_prompt, user_prompt):
    """Fallback LLM using OpenRouter"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"OpenRouter API failed: {e}")
        return None

def fill_docx_template(original_docx, data_mapping):
    doc = Document(original_docx)
    def replace_text_in_paragraph(paragraph, mapping):
        for key, value in mapping.items():
            if key in paragraph.text:
                if "{{" in key:
                    paragraph.text = paragraph.text.replace(key, str(value))
                else:
                    if str(value) not in paragraph.text:
                        paragraph.text = paragraph.text.replace(key, f"{key} {value}")
    for para in doc.paragraphs:
        replace_text_in_paragraph(para, data_mapping)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    replace_text_in_paragraph(para, data_mapping)
    target_stream = io.BytesIO()
    doc.save(target_stream)
    target_stream.seek(0)
    return target_stream

# --- MAIN UI ---
st.title("📄 GLR Pipeline: Automated Insurance Reporting")
st.markdown("""
Upload **one DOCX template** and **one PDF report**.  
The AI will extract data and fill the template.
""")

# --- API Keys ---
gemini_key = "AIzaSyDKeXrfDtNTkCCPznA1Uru6_c9tJk7Z1_Q"
openrouter_key = "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"

docx_file = st.file_uploader("Upload Insurance Template (.docx)", type=['docx'])
pdf_file = st.file_uploader("Upload PDF Photo Report (.pdf)", type=['pdf'])

if st.button("🚀 Process and Fill Template"):
    if not docx_file or not pdf_file:
        st.warning("Please upload both a DOCX template and a PDF report.")
    else:
        st.spinner("Processing pipeline...")
        pdf_text = extract_text_from_pdf(pdf_file)
        template_text = extract_text_from_docx_for_context(docx_file)

        system_prompt = """
        You are an expert insurance adjuster AI. Extract info from PDF and map to template fields. Return JSON.
        Use exact field names. Missing info -> "N/A".
        """
        user_prompt = f"""
        --- TEMPLATE CONTENT ---
        {template_text[:2000]} ...
        
        --- REPORT CONTENT ---
        {pdf_text[:6000]} ...
        
        Output JSON:
        """

        llm_response = query_gemini_flash(gemini_key, system_prompt, user_prompt)
        if not llm_response:
            llm_response = query_openrouter(openrouter_key, system_prompt, user_prompt)

        if llm_response:
            try:
                clean_json = llm_response.replace("```json", "").replace("```", "").strip()
                mapping_data = json.loads(clean_json)
                st.json(mapping_data, expanded=False)
                filled_docx = fill_docx_template(docx_file, mapping_data)
                st.success("Document generated successfully!")
                st.download_button(
                    label="📥 Download Filled Report (.docx)",
                    data=filled_docx,
                    file_name="Filled_Insurance_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except json.JSONDecodeError:
                st.error("Failed to parse LLM response as JSON.")
                st.text(llm_response)
        else:
            st.error("Failed to get response from both Gemini and OpenRouter.")
