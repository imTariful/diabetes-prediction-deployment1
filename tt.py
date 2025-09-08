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

# --- Model for Veterinary Study Output ---
class VetNote(BaseModel):
    topic: str
    subject: str
    explanation: str = Field(description="Concise but detailed study note, 2–4 clear paragraphs with examples")
    key_points: List[str] = Field(description="Important bullet points or definitions")
    exam_tip: str = Field(description="Quick exam-oriented tip")
    research_angle: str = Field(description="Insight on how this topic connects with veterinary research or evidence-based practice")
    further_reading: List[str] = Field(description="Suggested research papers, topics, or resources to explore")

# --- Async Generator ---
async def generate_notes(topic: str, subject: str) -> VetNote:
    prompt = f"""
    You are Tahia Tamanna's academic and research assistant for ASVM (Animal Husbandry & Veterinary Medicine) students.

    Write structured veterinary study notes and research insights on the topic: **{topic}**
    Subject area: {subject}

    Format the response as JSON with the following fields:
    - topic
    - subject
    - explanation: 2–4 short paragraphs in simple, clear academic language
    - key_points: 3–6 bullet points of important facts/definitions
    - exam_tip: one quick exam-oriented tip
    - research_angle: 1–2 paragraphs showing how this connects with veterinary research or evidence-based practice
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
        return VetNote(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        st.error("Failed to parse AI response. Here is the raw output:")
        st.code(raw_output)
        raise e

# --- Streamlit Dark-Themed UI ---
st.set_page_config(page_title="Tahia's Veterinary Study Assistant", page_icon="🐾", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #1F2937;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    .stTextInput>div>div>input {
        background-color: #1F2937;
        color: #FFFFFF;
        border-radius: 8px;
        padding: 0.5em;
    }
    .stMarkdown {
        color: #E5E7EB;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🐾 Tahia Tamanna - Veterinary Study Partner")
st.write("Your AI companion for **Animal Husbandry & Veterinary Medicine** studies.")

# Input fields
topic = st.text_input("Enter your topic")
subject = st.text_input("Enter the subject area")

if st.button("Generate Notes"):
    if topic.strip() == "" or subject.strip() == "":
        st.warning("⚠️ Please enter both topic and subject.")
    else:
        with st.spinner("Generating study notes... ⏳"):
            veterinary_note = asyncio.run(generate_notes(topic, subject))

        # Display in tabs
        tabs = st.tabs(["📖 Explanation", "🔑 Key Points", "💡 Exam Tip", "🔬 Research Angle", "📚 Further Reading"])

        with tabs[0]:
            st.markdown(f"### Topic: {veterinary_note.topic}")
            st.markdown(f"### Subject: {veterinary_note.subject}")
            st.write(veterinary_note.explanation)

        with tabs[1]:
            for kp in veterinary_note.key_points:
                st.markdown(f"- {kp}")

        with tabs[2]:
            st.info(veterinary_note.exam_tip)

        with tabs[3]:
            st.write(veterinary_note.research_angle)

        with tabs[4]:
            for ref in veterinary_note.further_reading:
                st.markdown(f"- {ref}")
