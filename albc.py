# insurance_filler_gemini.py
import streamlit as st
import google.generativeai as genai
from docx import Document
import io
import re
import json

# --- Page Config ---
st.set_page_config(page_title="Insurance Auto-Filler", layout="centered")
st.title("Insurance Template Auto-Filler")
st.markdown("### Powered by Gemini 1.5 Flash (free & super fast)")

# --- Hardcoded API key (for your personal use only!) ---
GEMINI_API_KEY = "AIzaSyDKeXrfDtNTkCCPznA1Uru6_c9tJk7Z1_Q"

# Optional: let user override if they want
gemini_api_key = st.text_input(
    "Gemini API Key (already filled for you)",
    value=GEMINI_API_KEY,
    type="password"
)

template_file = st.file_uploader("Upload Insurance Template (.docx)", type="docx")
pdf_files = st.file_uploader("Upload Photo Report(s) (.pdf)", type="pdf", accept_multiple_files=True)

if st.button("Generate Filled Document", type="primary"):
    if not template_file or not pdf_files:
        st.error("Please upload both the .docx template and at least one PDF report.")
        st.stop()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    with st.spinner("Gemini is reading your PDFs and filling the template..."):
        # 1. Find all {placeholders} in the docx
        doc = Document(template_file)
        placeholders = set()
        for p in doc.paragraphs:
            placeholders.update(re.findall(r'\{([^}]+)\}', p.text))
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    placeholders.update(re.findall(r'\{([^}]+)\}', cell.text))

        placeholder_list = list(placeholders)
        st.info(f"Found fields → `{'`, `'.join(placeholder_list)}`")

        # 2. Upload PDFs to Gemini
        uploaded_pdfs = []
        for pdf_file in pdf_files:
            uploaded = genai.upload_file(io.BytesIO(pdf_file.getvalue()), display_name=pdf_file.name)
            uploaded_pdfs.append(uploaded)

        # 3. Ask Gemini to extract exactly these fields
        prompt = f"""
You are an expert insurance processor.
Extract ONLY these fields from the attached photo report(s):

{', '.join(placeholder_list)}

Return pure JSON (nothing else, no markdown):
{{"field_name": "exact value or Not found", ...}}
"""

        try:
            response = model.generate_content(
                [prompt] + uploaded_pdfs,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )

            raw_json = response.text.strip().strip("```json").strip("```")
            data = json.loads(raw_json)
            st.success("Data extracted successfully!")
            st.json(data)

        except Exception as e:
            st.error(f"Gemini error: {e}")
            st.info("Common fix: wait 1–2 minutes (free tier rate limit) or try fewer/lighter PDFs.")
            st.stop()

        # 4. Replace placeholders in the document
        replaced = 0
        for field, value in data.items():
            placeholder = f"{{{field}}}"
            value = str(value) if value != "Not found" else "N/A"

            # Paragraphs
            for para in doc.paragraphs:
                if placeholder in para.text:
                    para.text = para.text.replace(placeholder, value)
                    replaced += 1

            # Tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if placeholder in cell.text:
                            cell.text = cell.text.replace(placeholder, value)
                            replaced += 1

        st.success(f"Done! Replaced {replaced} fields.")

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

        # Cleanup uploaded files
        for f in uploaded_pdfs:
            try:
                genai.delete_file(f.name)
            except:
                pass

st.markdown("---")
st.markdown("**Tip:** Use clear placeholders in your .docx like `{claim_number}`, `{plate_number}`, `{damage_date}`, `{total_cost}`")
