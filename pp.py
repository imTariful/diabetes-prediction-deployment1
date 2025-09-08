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

# --- Model for Nursing Study Output ---
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

    # Attempt to parse JSON safely
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

# Header
st.markdown("""
    <div style='background-color:#4CAF50; padding:20px; border-radius:10px'>
        <h1 style='color:white; text-align:center;'>🩺 Samia Islam Sami - Your Nursing Assistant</h1>
        <p style='color:white; text-align:center; font-size:18px;'>AI-powered academic & research companion for BSc Nursing students</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Input Section
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("Enter the topic you want to study")
        subject = st.text_input("Enter the subject area")
    with col2:
        st.write("⚡ Quick Tips")
        st.markdown("- Use clear topic names")
        st.markdown("- Examples: 'Diabetes Management', 'Cardiac Nursing', 'Infection Control'")

if st.button("Generate Notes"):
    if topic.strip() == "" or subject.strip() == "":
        st.warning("⚠️ Please enter both topic and subject.")
    else:
        with st.spinner("Generating study notes... ⏳"):
            nursing_note = asyncio.run(generate_notes(topic, subject))

        # Result Section
        st.markdown("### 📖 Your Personalized Nursing Notes")
        st.success(f"**Topic:** {nursing_note.topic}  |  **Subject:** {nursing_note.subject}")

        # Explanation
        st.markdown("#### 📝 Explanation")
        st.info(nursing_note.explanation)

        # Key Points
        st.markdown("#### 🔑 Key Points")
        for kp in nursing_note.key_points:
            st.markdown(f"- {kp}")

        # Exam Tip
        st.markdown("#### 💡 Exam Tip")
        st.warning(nursing_note.exam_tip)

        # Research Angle
        with st.expander("🔬 Research Angle (Click to Expand)"):
            st.write(nursing_note.research_angle)

        # Further Reading
        with st.expander("📚 Further Reading Suggestions (Click to Expand)"):
            for ref in nursing_note.further_reading:
                st.markdown(f"- {ref}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray'>Made with ❤️ by Samia Islam Sami</p>", unsafe_allow_html=True)
