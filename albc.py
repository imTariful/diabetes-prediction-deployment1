import streamlit as st
import google.generativeai as genai
from docx import Document
import io
import re
import json
import tempfile
import os
import mimetypes

# --- Page Config ---
st.set_page_config(page_title="Universal Document Filler", layout="wide")

st.title("🤖 Universal Document Auto-Filler")
st.markdown("""
**Works for ANY industry (Insurance, Legal, Medical, Real Estate, etc.)**
1. **Upload a Template (.docx):** Use placeholders like `[CLIENT_NAME]`, `[DATE]`, `[TOTAL_COST]`.
2. **Upload Source Files:** PDFs, Images (JPG/PNG).
3. **Generate:** The AI extracts the data and fills your document.
""")

# --- Configuration Sidebar ---
with st.sidebar:
    st.header("🔑 API Setup")
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.info("Get your free key at [Google AI Studio](https://aistudio.google.com/)")
    
    st.divider()
    
    st.header("⚙️ Model Settings")
    model_choice = st.selectbox("Choose Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    st.caption("Flash is faster/cheaper. Pro is better for complex reasoning.")

# --- Main File Uploads ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. The Template")
    template_file = st.file_uploader("Upload Word Doc (.docx)", type="docx")

with col2:
    st.subheader("2. The Source Data")
    source_files = st.file_uploader(
        "Upload Reports/Scans/Photos", 
        type=["pdf", "jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

# --- Helper Functions ---

def get_mime_type(filename):
    """Guess the MIME type for Gemini upload."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/pdf" # Default to PDF if unknown

def upload_to_gemini(uploaded_file):
    """Save Streamlit file to disk, upload to Gemini, return handle."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        gemini_file = genai.upload_file(tmp_path, mime_type=get_mime_type(uploaded_file.name))
        return gemini_file
    finally:
        os.remove(tmp_path) # Clean up local file

# --- Main Logic ---

if st.button("🚀 Process and Fill Document", type="primary"):
    # 1. Input Validation
    if not api_key:
        st.error("⚠️ Please enter your API Key in the sidebar.")
        st.stop()
    if not template_file:
        st.error("⚠️ Please upload a .docx template.")
        st.stop()
    if not source_files:
        st.error("⚠️ Please upload at least one source file (PDF or Image).")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_choice)

    with st.spinner("🔍 Scanning template for placeholders..."):
        # 2. Analyze DOCX for [PLACEHOLDERS]
        try:
            doc = Document(template_file)
            placeholders = set()
            # Regex to find anything inside square brackets, e.g., [FIELD_NAME]
            # \w+ matches letters, numbers, underscores
            regex_pattern = r'\[([A-Z0-9_ -]+)\]' 

            # Scan Paragraphs
            for p in doc.paragraphs:
                placeholders.update(re.findall(regex_pattern, p.text))
            
            # Scan Tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            placeholders.update(re.findall(regex_pattern, p.text))
            
            if not placeholders:
                st.warning("No placeholders found! Make sure your doc uses format like [CLIENT_NAME] or [DATE_OF_LOSS].")
                st.stop()
            
            fields_list = sorted(list(placeholders))
            
            # Display found fields in an expander
            with st.expander(f"✅ Found {len(fields_list)} fields to extract (Click to view)"):
                st.write(fields_list)

        except Exception as e:
            st.error(f"Error reading template: {e}")
            st.stop()

    # 3. Upload Source Files to Gemini
    gemini_file_handles = []
    status_text = st.empty()
    
    try:
        progress_bar = st.progress(0)
        for i, file in enumerate(source_files):
            status_text.text(f"Uploading source file {i+1}/{len(source_files)} to AI...")
            handle = upload_to_gemini(file)
            gemini_file_handles.append(handle)
            progress_bar.progress((i + 1) / len(source_files))
        
        status_text.text("🧠 AI is analyzing documents and extracting data...")

        # 4. Construct the Universal Prompt
        # This prompt is NOT hardcoded for insurance. It simply asks for the keys found in step 2.
        prompt = f"""
        You are an expert document analyst.
        
        I have provided one or more source documents (PDFs/Images).
        I need you to extract specific information to fill a form.
        
        Here is the exact list of fields I need you to find data for:
        {json.dumps(fields_list)}

        **Instructions:**
        1. Analyze the source documents thoroughly.
        2. Extract the most accurate value for each field in the list.
        3. If a field matches a concept in the text (e.g., [CLIENT] might match "Customer Name" or "Insured"), extract it.
        4. If a specific piece of data is NOT found, use the value "Not Specified".
        5. Return ONLY a valid JSON object where keys are the field names (without brackets) and values are the extracted strings.
        6. Do not include markdown formatting (like ```json). Just the raw JSON.
        """

        # 5. Generate Content
        response = model.generate_content(
            [prompt] + gemini_file_handles,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0 # Strict extraction
            )
        )
        
        # 6. Parse JSON
        cleaned_response = response.text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:-3]
            
        data_dict = json.loads(cleaned_response)
        
        status_text.text("✅ Data extracted! Filling document...")
        st.json(data_dict, expanded=False) # Show extracted data for verification

        # 7. Fill the Word Document
        # We perform a "safe" replace that keeps formatting
        def replace_in_text(text, data):
            for key, val in data.items():
                pattern = f"[{key}]"
                if pattern in text:
                    text = text.replace(pattern, str(val))
            return text

        # Fill Body Paragraphs
        for p in doc.paragraphs:
            if "[" in p.text:
                p.text = replace_in_text(p.text, data_dict)

        # Fill Tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        if "[" in p.text:
                            p.text = replace_in_text(p.text, data_dict)

        # 8. Create Download
        output = io.BytesIO()
        doc.save(output)
        output.seek(0)

        st.success("🎉 Document generated successfully!")
        
        st.download_button(
            label="📥 Download Filled Document",
            data=output,
            file_name="Filled_Document.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary"
        )

    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
    
    finally:
        # Cleanup Cloud Files to save money/space
        for f in gemini_file_handles:
            try:
                genai.delete_file(f.name)
            except:
                pass
