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
        # Sometimes model wraps JSON in ```json ... ```
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
st.title("🩺 Samia Islam Sami - Nursing Study & Research Assistant")
st.write("Your academic & research companion for **Nursing and Health Sciences**.")

topic = st.text_input("Enter your topic (any topic you want to learn)")
subject = st.text_input("Enter the subject area (any subject)")

if st.button("Generate Notes"):
    if topic.strip() == "" or subject.strip() == "":
        st.warning("⚠️ Please enter both topic and subject.")
    else:
        with st.spinner("Generating study notes... ⏳"):
            nursing_note = asyncio.run(generate_notes(topic, subject))

        # Display results
        st.subheader("📖 Explanation")
        st.write(nursing_note.explanation)

        st.subheader("🔑 Key Points")
        for kp in nursing_note.key_points:
            st.markdown(f"- {kp}")

        st.subheader("💡 Exam Tip")
        st.info(nursing_note.exam_tip)

        st.subheader("🔬 Research Angle")
        st.write(nursing_note.research_angle)

        st.subheader("📚 Further Reading Suggestions")
        for ref in nursing_note.further_reading:
            st.markdown(f"- {ref}")
