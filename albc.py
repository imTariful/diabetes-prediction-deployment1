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
    """
    Extracts raw text from a list of uploaded PDF files.
    """
    all_text = ""
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            all_text += f"\n--- Start of Report: {pdf_file.name} ---\n{text}\n--- End of Report ---\n"
        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {e}")
    return all_text

def extract_text_from_docx_for_context(docx_file):
    """
    Reads the docx file to create a context string for the LLM 
    so it knows what fields exist in the template.
    """
    doc = Document(docx_file)
    full_text = []
    
    # Extract from paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    
    # Extract from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text for cell in row.cells if cell.text.strip()]
            if row_text:
                full_text.append(" | ".join(row_text))
                
    return "\n".join(full_text)

def query_openrouter(api_key, system_prompt, user_prompt):
    """
    Sends the prompt to OpenRouter (using a free or cheap model like DeepSeek or Llama 3).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8501", # Required by OpenRouter
        "X-Title": "GLR Pipeline App",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek/deepseek-chat", # You can change this to "google/gemini-flash-1.5" or others
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
            data=json.dumps(payload)
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"LLM API Error: {str(e)}")
        if response:
            st.error(f"Response details: {response.text}")
        return None

def fill_docx_template(original_docx, data_mapping):
    """
    Replaces text in the docx based on the JSON mapping.
    Tries to preserve formatting by replacing text inside 'runs'.
    """
    doc = Document(original_docx)
    
    def replace_text_in_paragraph(paragraph, mapping):
        # This is a naive replacement. For complex templates, 
        # key-value placement often requires finding a key and appending to it,
        # but for this assignment, we assume direct placeholder replacement 
        # or appending to the field label.
        for key, value in mapping.items():
            if key in paragraph.text:
                # Basic strategy: If the key is "Client Name:", we append the value
                # Or if the key is "{{client_name}}", we replace it.
                
                # Check if it's a placeholder style (e.g. {{key}})
                if "{{" in key: 
                    if key in paragraph.text:
                        paragraph.text = paragraph.text.replace(key, str(value))
                else:
                    # It's likely a label like "Insured Name:". 
                    # We want to turn "Insured Name:" into "Insured Name: John Doe"
                    # Only do this if the value isn't already there
                    if str(value) not in paragraph.text:
                        paragraph.text = paragraph.text.replace(key, f"{key} {value}")

    # Process Paragraphs
    for para in doc.paragraphs:
        replace_text_in_paragraph(para, data_mapping)

    # Process Tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    replace_text_in_paragraph(para, data_mapping)
                    
    # Save to a byte stream
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

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenRouter API Key", type="password", help="Get a key from openrouter.ai")
    st.markdown("Recommended Model: `deepseek/deepseek-chat` (Hardcoded in logic)")
    
    st.divider()
    st.info("Ensure your DOCX has clear labels (e.g., 'Insured Name:', 'Date of Loss:') or placeholders.")

# File Uploaders
col1, col2 = st.columns(2)
with col1:
    docx_file = st.file_uploader("Upload Insurance Template (.docx)", type=['docx'])
with col2:
    pdf_files = st.file_uploader("Upload Photo Reports (.pdf)", type=['pdf'], accept_multiple_files=True)

# Processing Logic
if st.button("🚀 Process and Fill Template", type="primary"):
    if not api_key:
        st.warning("Please provide an OpenRouter API Key in the sidebar.")
    elif not docx_file or not pdf_files:
        st.warning("Please upload both a template and at least one PDF report.")
    else:
        with st.status("Processing Pipeline...", expanded=True) as status:
            
            # Step 1: Extract PDF Data
            st.write("🔍 Extracting text from PDF reports...")
            pdf_text = extract_text_from_pdfs(pdf_files)
            
            # Step 2: Extract Template Structure (Context)
            st.write("📖 analyzing template structure...")
            template_text = extract_text_from_docx_for_context(docx_file)
            
            # Step 3: LLM Inference
            st.write("🤖 Querying LLM to map data...")
            
            system_prompt = """
            You are an expert insurance adjuster AI. Your task is to extract specific information from inspection reports (PDFs) and map them to fields found in an insurance template.
            
            Return a purely JSON object.
            1. Identify fields in the 'Template Content' provided below (e.g., 'Insured Name:', 'Claim Number:', 'Date of Loss:', 'Cause of Loss:').
            2. Extract the corresponding values from the 'Report Content'.
            3. The JSON keys should be the EXACT text found in the template (e.g., "Insured Name:"), and the value should be the extracted data. 
            4. If the template uses placeholders like {{name}}, use that as the key.
            5. If a piece of information is missing, use "N/A".
            """
            
            user_prompt = f"""
            --- TEMPLATE CONTENT (Structure to fill) ---
            {template_text[:2000]} ... (truncated)
            
            --- REPORT CONTENT (Source Data) ---
            {pdf_text[:6000]} ... (truncated to fit context)
            
            Output JSON:
            """
            
            llm_response = query_openrouter(api_key, system_prompt, user_prompt)
            
            if llm_response:
                try:
                    # Parse JSON (sometimes LLMs include markdown ```json block, clean it)
                    clean_json = llm_response.replace("```json", "").replace("```", "").strip()
                    mapping_data = json.loads(clean_json)
                    
                    st.json(mapping_data, expanded=False)
                    st.write("✅ Data Extracted successfully.")
                    
                    # Step 4: Fill Document
                    st.write("✍️ Filling DOCX template...")
                    # Reset pointer for the docx file before reading again
                    docx_file.seek(0)
                    filled_docx = fill_docx_template(docx_file, mapping_data)
                    
                    status.update(label="Pipeline Complete!", state="complete", expanded=True)
                    
                    # Output
                    st.success("Document generated successfully!")
                    
                    st.download_button(
                        label="📥 Download Filled Report (.docx)",
                        data=filled_docx,
                        file_name="Filled_Insurance_Report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    
                except json.JSONDecodeError:
                    st.error("Failed to parse LLM response as JSON. Please try again.")
                    st.text(llm_response)
            else:
                st.error("Failed to get response from LLM.")
