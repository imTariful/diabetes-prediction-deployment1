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

# --- Streamlit Light-Themed Stylish UI ---
st.set_page_config(page_title="Tahia Tamanna - Vet Study Assistant", page_icon="🐾", layout="wide")
st.markdown(
    """
    <style>
    body {
        background: #FDFDFD;
        color: #1F2937;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4ADE80, #22D3EE);
        color: #FFFFFF;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        background-color: #F3F4F6;
        color: #111827;
        border-radius: 10px;
        padding: 0.5em;
        border: 1px solid #D1D5DB;
    }
    .stMarkdown {
        color: #111827;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.2em;
        margin-bottom: 1em;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("## 🐾 Tahia Tamanna - Veterinary Study Partner")
st.markdown("Your AI companion for **Animal Husbandry & Veterinary Medicine** studies. Explore topics, research angles, and exam tips in a visually rich layout.")

# Input fields
topic = st.text_input("Enter the topic you want to study")
subject = st.text_input("Enter the subject area")

if st.button("Generate Notes"):
    if topic.strip() == "" or subject.strip() == "":
        st.warning("⚠️ Please enter both topic and subject.")
    else:
        with st.spinner("Generating advanced study notes... ⏳"):
            veterinary_note = asyncio.run(generate_notes(topic, subject))

        # Display in stylish cards
        st.markdown(f"### 📖 Explanation")
        st.markdown(f"<div class='card'>{veterinary_note.explanation}</div>", unsafe_allow_html=True)

        st.markdown(f"### 🔑 Key Points")
        for kp in veterinary_note.key_points:
            st.markdown(f"<div class='card'>- {kp}</div>", unsafe_allow_html=True)

        st.markdown(f"### 💡 Exam Tip")
        st.markdown(f"<div class='card'><b>{veterinary_note.exam_tip}</b></div>", unsafe_allow_html=True)

        st.markdown(f"### 🔬 Research Angle")
        st.markdown(f"<div class='card'>{veterinary_note.research_angle}</div>", unsafe_allow_html=True)

        st.markdown(f"### 📚 Further Reading")
        for ref in veterinary_note.further_reading:
            st.markdown(f"<div class='card'>- {ref}</div>", unsafe_allow_html=True)

# --- Creator Credit ---
st.markdown(
    """
    <div style='text-align:center; margin-top:2em; padding:1em; color:#6B7280; font-size:0.9em;'>
        Created with ❤️ by <b>Tarif</b>
    </div>
    """,
    unsafe_allow_html=True
)
