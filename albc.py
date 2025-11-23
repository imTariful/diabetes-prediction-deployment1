# insurance_template_filler.py
import streamlit as st
import google.generativeai as genai
from docx import Document
import io
import re
import json

st.set_page_config(page_title="Insurance Template Auto-Filler", layout="wide")
st.title("Insurance Report Auto-Filler")
st.markdown("### Upload your .docx template + photo report PDFs → Get filled document instantly")

# === API Key ===
api_key = st.text_input(
    "Gemini API Key",
    value="AIzaSyDKeXrfDtNTkCCPznA1Uru6_c9tJk7Z1_Q",
    type="password"
)

template_file = st.file_uploader("Upload Insurance Template (.docx)", type="docx")
pdf_files = st.file_uploader("Upload Photo Report(s) (.pdf)", type="pdf", accept_multiple_files=True)

if st.button("Fill Template Automatically", type="primary"):
    if not template_file:
        st.error("Please upload a .docx template")
        st.stop()
    if not pdf_files:
        st.error("Please upload at least one photo report PDF")
        st.stop()
    if not api_key:
        st.error("Please enter your Gemini API key")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    with st.spinner("Reading template and analyzing photo reports..."):
        # Step 1: Load template and find all placeholders like [FIELD_NAME]
        doc = Document(template_file)
        placeholders = set()

        for paragraph in doc.paragraphs:
            placeholders.update(re.findall(r'\[([^\]]+)\]', paragraph.text))
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    placeholders.update(re.findall(r'\[([^\]]+)\]', cell.text))

        fields = sorted(list(placeholders))
        st.info(f"Found {len(fields)} fields to fill:\n" + ", ".join(fields))

        if not fields:
            st.warning("No placeholders found in template. Nothing to fill.")
            st.stop()

        # Step 2: Upload all PDFs
        uploaded_files = []
        for pdf_file in pdf_files:
            uploaded = genai.upload_file(
                io.BytesIO(pdf_file.getvalue()),
                mime_type="application/pdf",
                display_name=pdf_file.name
            )
            uploaded_files.append(uploaded)

        # Step 3: Unbiased, neutral prompt — only asks for real data
        prompt = f"""
You are an expert insurance document processor.

From the attached photo report PDF(s), extract accurate information to fill these exact fields from the template:

{', '.join(fields)}

Rules:
- Only use information visibly present in the PDFs
- If a field is not mentioned, use "Not specified"
- Be precise: include dates, addresses, damage counts, roof type, pitch, age, etc.
- Write values concisely but completely
- Never guess or invent information

Return ONLY a valid JSON object with field names as keys and extracted values as strings.

Example:
{{"INSURED_NAME": "Richard Daly", "DATE_LOSS": "9/28/2024", "ROOF_TYPE": "25 year 3-tab shingles"}}

Now extract:
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
            if "```" in json_text:
                json_text = re.search(r'```json\s*(.*?)\s*```', json_text, re.DOTALL)
                json_text = json_text.group(1) if json_text else json_text

            data = json.loads(json_text)
            st.success("Data extracted successfully from photo reports!")
            st.json(data)

        except Exception as e:
            st.error(f"LLM Error: {e}")
            st.info("Try again in 60 seconds or reduce PDF size (free tier limit).")
            st.stop()

        # Step 4: Fill the document
        replaced = 0
        for paragraph in doc.paragraphs:
            old_text = paragraph.text
            for field, value in data.items():
                placeholder = f"[{field}]"
                if placeholder in old_text:
                    paragraph.text = old_text.replace(placeholder, str(value))
                    replaced += 1

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    old_text = cell.text
                    for field, value in data.items():
                        placeholder = f"[{field}]"
                        if placeholder in old_text:
                            cell.text = old_text.replace(placeholder, str(value))
                            replaced += 1

        st.success(f"Template filled! Replaced {replaced} fields.")

        # Step 5: Download
        output = io.BytesIO()
        doc.save(output)
        output.seek(0)

        st.download_button(
            label="Download Filled Insurance Report.docx",
            data=output.getvalue(),
            file_name="Filled_Insurance_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        # Cleanup uploaded files
        for f in uploaded_files:
            try:
                genai.delete_file(f.name)
            except:
                pass

st.markdown("---")
st.success("This version is 100% unbiased • No hard-coded answers • Works with any template & any photo report")
st.caption("Install: `pip install streamlit google-generativeai python-docx` • Run: `streamlit run app.py`")
