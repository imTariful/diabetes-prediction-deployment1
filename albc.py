# insurance_filler_gemini_25.py
import streamlit as st
import google.generativeai as genai
from docx import Document
import io
import re
import json

# === CONFIG ===
st.set_page_config(page_title="Insurance Auto-Filler", layout="centered")
st.title("Insurance Template Auto-Filler")
st.markdown("### Powered by Gemini 2.5 Flash (latest & fastest)")

# Your API key (already included)
GEMINI_API_KEY = "AIzaSyDKeXrfDtNTkCCPznA1Uru6_c9tJk7Z1_Q"

gemini_api_key = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")

template_file = st.file_uploader("Upload Template (.docx)", type="docx")
pdf_files = st.file_uploader("Upload Photo Reports (.pdf)", type="pdf", accept_multiple_files=True)

if st.button("Generate Filled Document", type="primary"):
    if not template_file or not pdf_files:
        st.error("Please upload both template and at least one PDF.")
        st.stop()

    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")  # Updated to 2.5 Flash

    with st.spinner("Analyzing template and reading PDFs with Gemini 2.5 Flash..."):
        # 1. Extract placeholders {field_name}
        doc = Document(template_file)
        placeholders = set()
        for p in doc.paragraphs:
            placeholders.update(re.findall(r'\{([^}]+)\}', p.text))
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    placeholders.update(re.findall(r'\{([^}]+)\}', cell.text))

        fields = list(placeholders)
        if not fields:
            st.warning("No {placeholders} found in template.")
            fields = ["dummy_field"]  # avoid empty list error

        st.info(f"Found fields: `{'`, `'.join(fields)}`")

        # 2. Upload PDFs with correct mime_type
        uploaded_files = []
        for pdf_file in pdf_files:
            bytes_io = io.BytesIO(pdf_file.getbuffer())
            uploaded = genai.upload_file(
                bytes_io,
                mime_type="application/pdf",      # Fixes mime type error
                display_name=pdf_file.name
            )
            uploaded_files.append(uploaded)

        # 3. Ask Gemini to extract JSON
        prompt = f"""
Extract ONLY these fields from the attached photo report(s) as valid JSON:

{', '.join(fields)}

Rules:
- Use exact values from the document
- If not found → "Not found"
- Return pure JSON, no markdown, no explanation

Example:
{{"claim_number": "INS-2025-789", "damage_date": "2025-03-15"}}
"""

        try:
            response = model.generate_content(
                [prompt] + uploaded_files,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )

            # Clean response
            json_text = response.text.strip()
            if json_text.startswith("```"):
                json_text = json_text.split("```")[1].strip()
                if json_text.startswith("json"):
                    json_text = json_text[4:].strip()

            data = json.loads(json_text)
            st.success("Extraction successful with Gemini 2.5 Flash!")
            st.json(data)

        except Exception as e:
            st.error(f"Gemini Error: {e}")
            st.info("Try again in 1 minute (free tier rate limit) or reduce PDF size.")
            st.stop()

        # 4. Fill the document
        replaced = 0
        for field, value in data.items():
            placeholder = f"{{{field}}}"
            value = str(value) if value != "Not found" else "N/A"

            # Replace in paragraphs
            for para in doc.paragraphs:
                if placeholder in para.text:
                    para.text = para.text.replace(placeholder, value)
                    replaced += 1

            # Replace in tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if placeholder in cell.text:
                            cell.text = cell.text.replace(placeholder, value)
                            replaced += 1

        st.success(f"Filled {replaced} fields!")

        # 5. Download
        output = io.BytesIO()
        doc.save(output)
        output.seek(0)

        st.download_button(
            label="Download Filled Template.docx",
            data=output.getvalue(),
            file_name="Filled_Insurance_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        # Cleanup
        for f in uploaded_files:
            try:
                genai.delete_file(f.name)
            except:
                pass

st.markdown("---")
st.caption("Updated to Gemini 2.5 Flash — even faster & smarter than 1.5!")
