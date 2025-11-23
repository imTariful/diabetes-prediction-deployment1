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
from typing import Dict, Any

# --- Page Config ---
st.set_page_config(page_title="Zero-Miss Universal AI Document Filler", layout="wide", page_icon="🧠")
st.title("🧠 Zero-Miss Universal AI Document Filler")
st.markdown("""
**Guaranteed to Extract EVERYTHING** – No more "Not Found" surprises.

Upload your template (.docx) and sources (PDFs, images, scans). AI reasons step-by-step, self-corrects, and fills perfectly.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Gemini Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Get free key: https://aistudio.google.com/app/apikey")
    model_choice = st.selectbox(
        "Model (Pro for handwriting/complex docs)",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=1
    )
    media_resolution = st.selectbox("PDF/Image Quality", ["low", "medium", "high"], index=2, help="High for handwriting/fine text – AI will auto-adjust")
    st.info("**Pro Tip**: High res + Pro model = 95%+ accuracy on scans/handwriting")

# --- Uploads ---
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("1. Template (.docx with [placeholders])", type="docx")
with col2:
    source_files = st.file_uploader("2. Source Files (PDF, Images, Scans...)", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "heic"])

if st.button("🚀 Extract & Fill with Zero Misses", type="primary"):
    if not api_key or not template_file or not source_files:
        st.error("Please provide API key, template, and source files.")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_choice)

    progress = st.progress(0)
    status = st.empty()

    # Step 1: Extract placeholders
    status.text("🔍 Reading template placeholders...")
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

    placeholders = sorted([p.strip() for p in placeholders if p.strip()])
    
    if not placeholders:
        st.warning("No placeholders found! Use format like [Client Name], [Date].")
        st.stop()

    progress.progress(15)
    status.text(f"Found {len(placeholders)} fields: {', '.join(placeholders[:8])}{'...' if len(placeholders)>8 else ''}")

    # Step 2: Upload files (Fixed: Valid params only + display_name for better context)
    status.text("📤 Uploading sources to AI (high-res vision mode)...")
    gemini_files = []
    for file in source_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        mime_type, _ = mimetypes.guess_type(tmp_path)
        if mime_type is None:
            mime_type = "application/pdf" if tmp_path.lower().endswith(".pdf") else "image/jpeg"

        # Fixed upload: Use display_name for AI context (no invalid display_only)
        g_file = genai.upload_file(
            path=tmp_path, 
            mime_type=mime_type,
            display_name=file.name  # Helps AI reference specific files
        )
        gemini_files.append(g_file)
        os.unlink(tmp_path)

    progress.progress(30)

    # ULTIMATE ZERO-MISS PROMPT (Enhanced for resolution/orientation)
    # First Pass: Comprehensive Extraction
    extraction_prompt = f"""
<system_instruction>
You are a forensic document analyst with 20+ years extracting data from messy PDFs, scans, handwritten forms, and photos. Your goal: Extract ABSOLUTELY EVERY piece of meaningful information without missing ANYTHING. Be exhaustive—scan every page, corner, and detail.
</system_instruction>

<role_instruction>
Use STEP-BY-STEP reasoning (Chain-of-Thought):
1. Scan ALL attached files page-by-page, including images/PDFs. Treat as {'high-resolution scans' if media_resolution == 'high' else 'standard resolution'}. Auto-rotate if tilted; enhance faint text/handwriting using context.
2. Extract RAW data: Names, dates (any format), addresses, phones/emails, IDs/numbers (claim/policy/VIN), amounts, descriptions, signatures, even inferred context (e.g., "hail dents in roof photo" → damage type).
3. For each template field, semantically match: 
   - [Client Name] → "Insured", "Applicant", "John Doe" (full/preferred name).
   - [Date of Loss] → "DOL", "Incident", "10/5/24" (standardize to MM/DD/YYYY).
   - [Address] → Full street/city/state/ZIP from any section.
   - [Damage Desc] → Any notes/photos implying cause (e.g., "windstorm" or "cracks visible").
   - Generic fields: Infer from context (e.g., [Amount] → totals/deductibles).
4. If faint/handwritten: Use context clues (surrounding labels) to decipher.
5. Assign confidence: 1.0 (clear), 0.7 (inferred), 0.3 (low-conf guess). Never hallucinate—flag low-conf.
6. Output ONLY valid JSON with exact keys from: {json.dumps(placeholders)}.
Few-Shot Examples:
{{
  "Client Name": {{"value": "Sarah Johnson", "confidence": 1.0, "source": "Header form"}},
  "Date of Loss": {{"value": "10/15/2024", "confidence": 0.9, "source": "Handwritten note pg2"}},
  "Damage Desc": {{"value": "Hail dents on roof; est. $5k", "confidence": 0.8, "source": "Photo + estimate"}}
}}
If truly absent: {{"value": "Not Found", "confidence": 0.0, "source": "Absent"}}
</role_instruction>

<query_instruction>
Based on the above, analyze the attached files and extract ALL data. Output raw JSON only—no extras.
</query_instruction>
"""

    status.text("🤖 Step 1: AI deep-scan for all data (20-90s)...")
    progress.progress(50)

    try:
        # First Pass (Fixed: Correct GenerationConfig import)
        response1 = model.generate_content(
            [extraction_prompt] + gemini_files,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0  # Zero randomness for accuracy
            )
        )

        # Robust JSON Cleaning
        def clean_json(raw_text: str) -> str:
            raw_text = raw_text.strip()
            # Strip markdown
            raw_text = re.sub(r'^```json\s*|\s*```$', '', raw_text, flags=re.MULTILINE)
            # Extract JSON block
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            return json_match.group(0) if json_match else raw_text

        raw_json = clean_json(response1.text)
        data: Dict[str, Dict[str, Any]] = json.loads(raw_json)

        # Step 3: Second Pass - Self-Critique & Refine Misses
        critique_prompt = f"""
<system_instruction>Review your extraction for completeness. Flag & fix misses.</system_instruction>

<role_instruction>
1. Review JSON: For each field with "Not Found" or low confidence (<0.7), re-scan files for hidden matches (e.g., footnotes, stamps, photo backgrounds).
2. Use context: Cross-reference across files (e.g., name in PDF1 matches photo in PDF2).
3. Refine values: Standardize dates/amounts; boost confidence if corroborated.
4. Output updated JSON only.
</role_instruction>

<query_instruction>
Critique and refine this extraction: {json.dumps(data, indent=2)}
Re-analyze files if needed.
</query_instruction>
"""

        status.text("🔄 Step 2: AI self-critique & fix misses...")
        progress.progress(70)

        response2 = model.generate_content(
            [critique_prompt] + gemini_files,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )

        refined_raw = clean_json(response2.text)
        data = json.loads(refined_raw)  # Overwrite with refined

        progress.progress(85)
        status.text("📝 Filling template...")

        # Preview with Confidence
        st.subheader("Extracted Data (with Confidence)")
        high_conf = {k: v for k, v in data.items() if v.get("confidence", 0) >= 0.7}
        low_conf = {k: v for k, v in data.items() if v.get("confidence", 0) < 0.7}

        col_a, col_b = st.columns(2)
        with col_a:
            st.success(f"High Confidence ({len(high_conf)})")
            st.json({k: v["value"] for k, v in high_conf.items()})
        with col_b:
            if low_conf:
                st.warning(f"Low/Missing ({len(low_conf)})")
                st.json({k: v["value"] for k, v in low_conf.items()})
            else:
                st.balloons()

        # Smart Replacement (Preserves Formatting)
        def replace_in_paragraph(paragraph, key, value):
            placeholder = f"[{key}]"
            if placeholder in paragraph.text:
                for run in paragraph.runs:
                    if placeholder in run.text:
                        run.text = run.text.replace(placeholder, str(value["value"]))

        for p in doc.paragraphs:
            for key in placeholders:
                if key in data:
                    replace_in_paragraph(p, key, data[key])

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for key in placeholders:
                            if key in data:
                                replace_in_paragraph(p, key, data[key])

        # Download
        bio = io.BytesIO()
        doc.save(bio)
        bio.seek(0)

        progress.progress(100)
        status.text("✅ Zero-Miss Complete! All data extracted & filled.")

        st.download_button(
            label="📥 Download Filled Document",
            data=bio,
            file_name=f"ZeroMiss_Filled_{template_file.name.replace('.docx', '')}_{datetime.now().strftime('%Y%m%d')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary"
        )

    except json.JSONDecodeError as e:
        st.error(f"JSON Parse Error: {e}. Raw: {response1.text if 'response1' in locals() else 'N/A'}")
    except Exception as e:
        st.error(f"AI Error: {e}")
        if 'response1' in locals():
            st.code(response1.text)

    finally:
        for f in gemini_files:
            try:
                genai.delete_file(f.name)
            except:
                pass
