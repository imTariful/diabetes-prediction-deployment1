import os
import asyncio
import json
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

if not API_KEY:
    st.error("Please set API_KEY in your environment variables or .env file.")
    st.stop()

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Model ---
class NursingNote(BaseModel):
    topic: str
    subject: str
    explanation: str = Field(description="Concise but detailed study note, 2–4 clear paragraphs with examples")
    key_points: List[str] = Field(description="Important bullet points or definitions")
    exam_tip: str = Field(description="Quick exam-oriented tip")
    research_angle: str = Field(description="Insight on how this topic connects with healthcare research or evidence-based practice")
    further_reading: List[str] = Field(description="Suggested research papers, topics, or resources to explore")

# --- Async Generator ---
async def generate_notes(topic: str, subject: str) -> NursingNote:
    prompt = f"""
    You are Samia Islam Sami, an academic and research assistant for BSc Nursing students.  

    Write structured nursing study notes and research insights on the topic: **{topic}**
    Subject area: {subject}

    Format the response as JSON with the following fields:
    - topic
    - subject
    - explanation: 2–4 short paragraphs in simple, clear academic language
    - key_points: 3–6 bullet points of important facts/definitions
    - exam_tip: one quick exam-oriented tip
    - research_angle: 1–2 paragraphs showing how this connects with research or evidence-based nursing
    - further_reading: 2–4 suggestions (topics, papers, or resources)
    """

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200
    )

    raw_output = response.choices[0].message.content

    try:
        if raw_output.strip().startswith("```"):
            raw_output = "\n".join(raw_output.strip().split("\n")[1:-1])
        data = json.loads(raw_output)
        return NursingNote(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        st.error("Failed to parse AI response. Here is the raw output:")
        st.code(raw_output)
        raise e

# --- Streamlit UI ---
st.set_page_config(page_title="Samia's Nursing Assistant", page_icon="🩺", layout="wide")

# --- Stylish Gradient Header ---
st.markdown("""
<div style='
    background: linear-gradient(90deg, #a1c4fd, #c2e9fb);
    padding:30px;
    border-radius:20px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
    text-align:center;
'>
    <h1 style='color:#333; font-family:Verdana;'>🩺 Samia Islam Sami - Nursing Assistant</h1>
    <p style='color:#555; font-size:18px;'>AI-powered academic & research companion for BSc Nursing students</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Input Section ---
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("Enter the topic", placeholder="e.g., Diabetes Management")
        subject = st.text_input("Enter the subject area", placeholder="e.g., Medical-Surgical Nursing")
    with col2:
        st.markdown("### ⚡ Quick Tips")
        st.markdown("- Use clear and specific topic names")
        st.markdown("- Example: 'Cardiac Nursing', 'Pediatric Care', 'Infection Control'")

# --- Generate Notes ---
if st.button("Generate Notes"):
    if topic.strip() == "" or subject.strip() == "":
        st.warning("⚠️ Please enter both topic and subject.")
    else:
        with st.spinner("Generating your study notes... ⏳"):
            nursing_note = asyncio.run(generate_notes(topic, subject))

        # --- Result Cards ---
        st.markdown("### 📖 Personalized Nursing Notes")
        st.success(f"**Topic:** {nursing_note.topic}  |  **Subject:** {nursing_note.subject}")

        # Explanation Card
        st.markdown(f"""
        <div style='
            background: #fff9f0; 
            padding: 20px; 
            border-radius: 15px; 
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-bottom:15px;
        '>
            <h3>📝 Explanation</h3>
            <p style='color:#333'>{nursing_note.explanation}</p>
        </div>
        """, unsafe_allow_html=True)

        # Key Points Card
        st.markdown(f"""
        <div style='
            background: #f0fff4; 
            padding: 20px; 
            border-radius: 15px; 
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-bottom:15px;
        '>
            <h3>🔑 Key Points</h3>
            <ul style='color:#333'>
        """ + "".join([f"<li>{kp}</li>" for kp in nursing_note.key_points]) + "</ul></div>", unsafe_allow_html=True)

        # Exam Tip Card
        st.markdown(f"""
        <div style='
            background: #fff0f6; 
            padding: 20px; 
            border-radius: 15px; 
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-bottom:15px;
        '>
            <h3>💡 Exam Tip</h3>
            <p style='color:#333'>{nursing_note.exam_tip}</p>
        </div>
        """, unsafe_allow_html=True)

        # Research Angle Card
        st.markdown(f"""
        <div style='
            background: #f9f0ff; 
            padding: 20px; 
            border-radius: 15px; 
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-bottom:15px;
        '>
            <h3>🔬 Research Angle</h3>
            <p style='color:#333'>{nursing_note.research_angle}</p>
        </div>
        """, unsafe_allow_html=True)

        # Further Reading Card
        st.markdown(f"""
        <div style='
            background: #f0f9ff; 
            padding: 20px; 
            border-radius: 15px; 
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-bottom:15px;
        '>
            <h3>📚 Further Reading</h3>
            <ul style='color:#333'>
        """ + "".join([f"<li>{ref}</li>" for ref in nursing_note.further_reading]) + "</ul></div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray'>Made with ❤️ by Samia Islam Sami</p>", unsafe_allow_html=True)
