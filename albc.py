import streamlit as st
import google.generativeai as genai
from docx import Document
import io
import json
import re
import tempfile
import os
import mimetypes
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Universal AI Document Filler", layout="wide", page_icon="🧠")
st.title("🧠 Universal AI Document Filler")
st.markdown("""
**No rules. No fixed fields. Just intelligence.**

Upload **any template** (.docx) with placeholders like `[Date]`, `[Client Name]`, `[Damage Description]`, etc.  
Then upload **any source files** — PDFs, photos, scans, handwritten forms, screenshots...  
The AI will **understand everything** and fill your template perfectly.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Gemini Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Get free key: https://aistudio.google.com/app/apikey")
    model_choice = st.selectbox(
        "Model (Pro = better handwriting & reasoning)",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=1
    )
    st.info("Use **Pro** for handwritten, damaged, or complex forms")

# --- Uploads ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("1. Template (.docx with [placeholders])", type="docx")
with col2:
    source_files = st.file_uploader("2. Source Files (PDF, Images, Photos...)", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "heic"])

if st.button("🚀 Fill Document with AI", type="primary"):
    if not api_key or not template_file or not source_files:
        st.error("Please provide API key, template, and source files.")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_choice)

    progress = st.progress(0)
    status = st.empty()

    # Step 1: Extract placeholders
    status.text("Reading template placeholders...")
    doc = Document(template_file)
    placeholder_pattern = r'\[([^\]]+)\]'
    placeholders = set()

    for paragraph in doc.paragraphs:
        placeholders.update(re.findall(placeholder_pattern, paragraph.text))
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    placeholders.update(re.findall(placeholder_pattern, paragraph.text))

    placeholders = [p.strip() for p in placeholders if p.strip()]
    
    if not placeholders:
        st.warning("No placeholders found! Use format like [Client Name], [Date of Birth], etc.")
        st.stop()

    progress.progress(20)
    status.text(f"Found {len(placeholders)} fields: {', '.join(placeholders[:10])}{'...' if len(placeholders)>10 else ''}")

    # Step 2: Upload files to Gemini
    status.text("Uploading source files to AI...")
    gemini_files = []
    for file in source_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        mime_type, _ = mimetypes.guess_type(tmp_path)
        if mime_type is None:
            mime_type = "application/pdf" if tmp_path.lower().endswith(".pdf") else "image/jpeg"

        g_file = genai.upload_file(tmp_path, mime_type=mime_type)
        gemini_files.append(g_file)
        os.unlink(tmp_path)

    progress.progress(50)

    # THE ULTIMATE PROMPT (This is where the magic happens)
    prompt = f"""
You are an expert forensic document analyst and data extraction specialist.

Your mission: Extract **every piece of meaningful information** from the attached source documents (PDFs, photos, scans, forms, handwritten notes, etc.) and map it intelligently to the template fields.

Template fields to fill:
{json.dumps(placeholders, indent=2)}

RULES:
1. **Semantic Matching** — The source may use different wording:
   - [Client Name] → could be "Insured", "Policyholder", "Applicant", "Mr. John Doe"
   - [Date of Loss] → "Incident Date", "DOL", "Loss Occurred", "03/15/2024"
   - [Address] → street, city, ZIP from anywhere
   - [Vehicle Make] → "Toyota", "Ford F-150", etc.

2. **Extract Everything Useful** — Even if not directly asked, include:
   - Names (people, companies, adjusters)
   - Dates (any format)
   - Addresses
   - Phone numbers, emails
   - Claim/Policy/File numbers
   - Damage descriptions
   - Amounts (estimates, deductibles)
   - Vehicle details (make, model, year, VIN, plate)

3. **Be Smart About Context**:
   - If a photo shows a roof with hail damage → fill [Damage Type] = "Hail"
   - If handwritten note says "windstorm 10/5/24" → [Date of Loss] = "10/05/2024", [Cause] = "Windstorm"

4. If a field is truly missing → use "Not Found"
5. If multiple values → pick the most official/relevant one

Return ONLY a clean JSON object with exact placeholder names as keys.

Example:
{{
  "Client Name": "Sarah Johnson",
  "Date of Loss": "10/15/2024",
  "Damage Description": "Hail dents on roof and hood",
  "Claim Number": "CLM-2024-8871",
  "Not Found Example": "Not Found"
}}
"""

    status.text("AI is analyzing all documents deeply... (this may take 20–90 seconds)")
    progress.progress(70)

    try:
        response = model.generate_content(
            [prompt] + gemini_files,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )

        # Clean response
        raw = response.text.strip()
        if "```" in raw:
            raw = re.search(r"\{.*\}", raw, re.DOTALL)
            raw = raw.group(0) if raw else response.text

        data = json.loads(raw)

        progress.progress(90)
        status.text("Filling document...")

        # Smart replacement (preserves formatting)
        def replace_in_paragraph(paragraph, key, value):
            placeholder = f"[{key}]"
            if placeholder in paragraph.text:
                for run in paragraph.runs:
                    if placeholder in run.text:
                        run.text = run.text.replace(placeholder, str(value))

        for p in doc.paragraphs:
            for key, val in data.items():
                replace_in_paragraph(p, key, val)

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for key, val in data.items():
                            replace_in_paragraph(p, key, val)

        # Save and offer download
        bio = io.BytesIO()
        doc.save(bio)
        bio.seek(0)

        progress.progress(100)
        status.text("Done! Download your filled document")

        st.success("Document filled perfectly!")
        st.json(data, expanded=False)

        st.download_button(
            label="📥 Download Filled Document",
            data=bio,
            file_name=f"Filled_{template_file.name.replace('.docx', '')}_{datetime.now().strftime('%Y%m%d')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary"
        )

    except Exception as e:
        st.error(f"Error: {e}")
        if "response.text" in locals():
            st.code(response.text)

    finally:
        # Cleanup
        for f in gemini_files:
            try:
                genai.delete_file(f.name)
            except:
                pass
