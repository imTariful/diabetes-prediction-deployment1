import streamlit as st
import google.generativeai as genai
from docx import Document
import io
import re
import json
import tempfile
import os
import mimetypes

# --- Page Setup ---
st.set_page_config(page_title="Smart Universal Doc Filler", layout="wide")
st.title("🧠 Smart Universal Document Filler")
st.markdown("""
**How this works for ANY file:**
1. The App reads your Template and finds placeholders like `[DATE_LOSS]`.
2. It sends your PDF/Images to Google Gemini.
3. **The "Smart" Step:** It asks Gemini to use reasoning to find the matching data. 
   *(e.g., matching "[DATE_LOSS]" to "DOL" or "Date of Incident" in the source).*
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    model_choice = st.selectbox("Model Strength", ["gemini-2.5-flash", "gemini-2.5-pro"])
    st.caption("Use 'Pro' if your documents have very complex handwriting or tables.")

# --- File Uploads ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("1. Upload Template (.docx)", type="docx")
with col2:
    source_files = st.file_uploader("2. Upload Source (PDF/Images)", accept_multiple_files=True)

# --- Logic ---
if st.button("Generate Document", type="primary"):
    if not api_key or not template_file or not source_files:
        st.error("Please provide API Key, Template, and Source Files.")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_choice)

    with st.spinner("Analyzing Template structure..."):
        # 1. Extract Placeholders from Word Doc
        doc = Document(template_file)
        placeholders = set()
        # Find all text inside brackets: [ANY_TEXT_HERE]
        regex = r'\[([A-Z0-9_ -]+)\]'

        for p in doc.paragraphs:
            placeholders.update(re.findall(regex, p.text))
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        placeholders.update(re.findall(regex, p.text))
        
        fields = list(placeholders)
        if not fields:
            st.warning("No placeholders found! Use format [PLACEHOLDER] in your Word doc.")
            st.stop()
            
        with st.expander(f"Found {len(fields)} Target Fields"):
            st.write(fields)

    with st.spinner("Reading Source Files & Extracting Data..."):
        # 2. Upload Source Files to Gemini
        gemini_files = []
        for file in source_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(tmp_path)
            if mime_type is None: mime_type = "application/pdf"
            
            g_file = genai.upload_file(tmp_path, mime_type=mime_type)
            gemini_files.append(g_file)
            os.remove(tmp_path)

        # 3. The "Context-Aware" Prompt
        # This is the key difference. We tell AI to infer meanings.
        prompt = f"""
        You are an intelligent data extraction assistant.
        
        I have a document template with the following placeholders:
        {json.dumps(fields)}

        Your Job:
        1. Analyze the attached source documents (images/PDFs).
        2. For each placeholder in the list, find the corresponding data in the source documents.
        3. **Use Reasoning:** - If the placeholder is [DATE_LOSS], look for "Date of Loss", "DOL", or "Incident Date".
           - If the placeholder is [INSURED_NAME], look for "Insured", "Client", or "Policyholder".
           - If the placeholder is [ROOF_MAT], look for roof details like "3-tab shingles".
        4. Extract the exact value found.
        5. If data is absolutely missing, return "Not Specified".
        
        Return the result as a raw JSON object: {{ "PLACEHOLDER_NAME": "Extracted Value" }}
        """

        try:
            response = model.generate_content(
                [prompt] + gemini_files,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            
            # Clean and Parse JSON
            text_resp = response.text.strip()
            if text_resp.startswith("```json"): text_resp = text_resp[7:-3]
            data = json.loads(text_resp)
            
            st.success("Data Extracted! Review below:")
            st.json(data)

            # 4. Fill the Document
            def replace_text(text, data_dict):
                for key, val in data_dict.items():
                    # Check for [KEY]
                    placeholder_str = f"[{key}]"
                    if placeholder_str in text:
                        # Replace, ensuring it's a string
                        text = text.replace(placeholder_str, str(val))
                return text

            # Paragraphs
            for p in doc.paragraphs:
                if "[" in p.text:
                    p.text = replace_text(p.text, data)
            
            # Tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for p in cell.paragraphs:
                            if "[" in p.text:
                                p.text = replace_text(p.text, data)

            # 5. Download
            bio = io.BytesIO()
            doc.save(bio)
            bio.seek(0)
            
            st.download_button(
                label="📥 Download Completed Report",
                data=bio,
                file_name="Filled_Universal_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary"
            )

        except Exception as e:
            st.error(f"AI Error: {e}")
        
        finally:
            for f in gemini_files:
                try: genai.delete_file(f.name)
                except: pass
