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

def extract_text_from_pdfs(pdf_files):
    all_text = ""
    progress_bar = st.progress(0)
    total_files = len(pdf_files)
    
    for idx, pdf_file in enumerate(pdf_files, start=1):
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            all_text += f"\n--- Start of Report: {pdf_file.name} ---\n{text}\n--- End of Report ---\n"
        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {e}")
        progress_bar.progress(idx / total_files)
    progress_bar.empty()
    return all_text

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

def query_gemini(api_key, system_prompt, user_prompt):
    """
    Query Google Gemini (via PaLM API) and return JSON-compatible string
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "model": "gemini-1.5-turbo",  # change if you have access to a different Gemini model
        "temperature": 0
    }
    
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta2/models/gemini-1.5:generateMessage",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]
    except Exception as e:
        st.error(f"LLM API Error: {str(e)}")
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
This tool automates the filling of insurance templates.  
1. Upload your **DOCX Template**.  
2. Upload **PDF Photo Reports**.  
3. The AI extracts data and fills the document.
""")

# --- Gemini API Key ---
# Replace below with your real Gemini API Key
api_key = "AIzaSyCzhgy8y8dgPVtjsJp8xoTBcciow80uDtA"

# Optional sidebar override
with st.sidebar:
    typed_key = st.text_input("Gemini API Key (optional)", type="password")
    if typed_key:
        api_key = typed_key

if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
    st.warning("Please set your Gemini API key in the code or sidebar.")

# File Uploaders
col1, col2 = st.columns(2)
with col1:
    docx_file = st.file_uploader("Upload Insurance Template (.docx)", type=['docx'])
with col2:
    pdf_files = st.file_uploader("Upload Photo Reports (.pdf)", type=['pdf'], accept_multiple_files=True)

# Processing Logic
if st.button("🚀 Process and Fill Template", type="primary"):
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        st.warning("Gemini API key is missing or invalid.")
    elif not docx_file or not pdf_files:
        st.warning("Please upload both a template and at least one PDF report.")
    else:
        with st.spinner("Processing pipeline..."):
            st.write("🔍 Extracting text from PDF reports...")
            pdf_text = extract_text_from_pdfs(pdf_files)
            
            with st.expander("View Extracted PDF Text"):
                st.text_area("PDF Content", pdf_text, height=300)
            
            st.write("📖 Analyzing template structure...")
            template_text = extract_text_from_docx_for_context(docx_file)
            
            st.write("🤖 Querying Gemini LLM to map data...")
            system_prompt = """
            You are an expert insurance adjuster AI. Extract information from inspection reports (PDFs) and map them to fields in an insurance template.
            Return JSON. Use exact field names from template. Missing info -> "N/A".
            """
            
            user_prompt = f"""
            --- TEMPLATE CONTENT ---
            {template_text[:2000]} ...
            
            --- REPORT CONTENT ---
            {pdf_text[:6000]} ...
            
            Output JSON:
            """
            
            llm_response = query_gemini(api_key, system_prompt, user_prompt)
            
            if llm_response:
                try:
                    clean_json = llm_response.replace("```json", "").replace("```", "").strip()
                    mapping_data = json.loads(clean_json)
                    
                    st.json(mapping_data, expanded=False)
                    st.write("✅ Data extracted successfully.")
                    
                    st.write("✍️ Filling DOCX template...")
                    docx_file.seek(0)
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
                st.error("Failed to get response from LLM.")
