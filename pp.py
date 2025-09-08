import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

if not API_KEY:
    st.error("Please set API_KEY in your environment variables or .env file.")
    st.stop()

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)

# --- Model for Nursing Study Output ---
class NursingNote(BaseModel):
    topic: str
    subject: str
    explanation: str = Field(description="Concise but detailed study note, 2–4 clear paragraphs with examples")
    key_points: List[str] = Field(description="Important bullet points or definitions")
    exam_tip: str = Field(description="Quick exam-oriented tip")
    research_angle: str = Field(description="Insight on how this topic connects with healthcare research or evidence-based practice")
    further_reading: List[str] = Field(description="Suggested research papers, topics, or resources to explore")

# --- Nursing Study & Research Assistant Agent ---
nursing_assistant_agent = Agent(
    name="Nursing Study Assistant",
    instructions="""
    You are an academic and research assistant for a BSc Nursing student.  
    Your role is to generate clear, concise, and well-structured study notes and research insights.  

    Requirements:
    - Write in simple, clear academic language
    - Provide structured explanations (2–4 short paragraphs)
    - Highlight key definitions, nursing procedures, or examples
    - Provide bullet-point style 'key points' for revision
    - Give 1 quick exam-oriented tip
    - Show how this topic connects with research or evidence-based nursing
    - Suggest 2–4 further reading references (topics, papers, or areas to explore)
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    output_type=NursingNote,
)

# --- Async Runner ---
async def generate_notes(topic: str, subject: str):
    result = await Runner.run(
        nursing_assistant_agent,
        f"Write nursing study notes and research insights about '{topic}' in the subject {subject}."
    )
    return result.final_output

# --- Streamlit UI ---
st.set_page_config(page_title="Nursing Study & Research Assistant", page_icon="🩺", layout="wide")
st.title("🩺 BSc Nursing Study & Research Assistant")
st.write("Your academic & research companion for **Nursing and Health Sciences**.")

topic = st.text_input("Enter your topic (e.g., Hypertension Management, Infection Control, Mental Health Nursing)")
subject = st.selectbox(
    "Choose Subject", 
    ["Fundamentals of Nursing", "Medical-Surgical Nursing", "Community Health Nursing", "Obstetrics & Gynecology", "Pediatrics", "Psychiatric Nursing", "Research in Nursing"]
)

if st.button("Generate Notes"):
    if topic.strip() == "":
        st.warning("⚠️ Please enter a topic.")
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
