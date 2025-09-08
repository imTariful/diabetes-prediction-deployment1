import os
import asyncio
import json
import re
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

# --- Model for AI Coding Agent Output ---
class CodeAgentResponse(BaseModel):
    task_description: str
    language: str
    code: str = Field(description="Complete code solution with proper formatting and comments")
    explanation: str = Field(description="Step-by-step explanation of the code, logic, and why it works")
    debug_notes: str = Field(description="Any bugs found or potential errors and their fixes")
    optimization_suggestions: str = Field(description="Tips to make the code faster, cleaner, or more efficient")
    best_practices: List[str] = Field(description="Tips for writing similar code in the future")
    references: List[str] = Field(description="Helpful links or documentation")

# --- Safe JSON Parsing Function ---
def safe_parse_json(raw_output: str):
    raw_output = raw_output.strip()

    # Remove ```json or ``` if present
    if raw_output.startswith("```"):
        raw_output = "\n".join(raw_output.split("\n")[1:-1]).strip()

    if not raw_output:
        return None  # empty response

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # Attempt to extract JSON object using regex
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None

# --- Async Coding Agent Function ---
async def coding_agent(task: str, language: str, mode: str = "write") -> CodeAgentResponse:
    prompt = f"""
    You are Tarif's ultimate coding assistant and tutor.
    Tasks:
    - Write full, correct, and efficient code
    - Explain code step by step
    - Debug code and fix errors
    - Optimize code for performance
    - Answer follow-up questions interactively

    Task: {task}
    Programming Language: {language}
    Mode: {mode}

    IMPORTANT: Output must be valid JSON only. Do not include extra text or markdown.
    Format JSON fields:
    - task_description
    - language
    - code
    - explanation
    - debug_notes
    - optimization_suggestions
    - best_practices (3–5 points)
    - references (2–4 links)
    """

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500
    )

    raw_output = response.choices[0].message.content
    data = safe_parse_json(raw_output)

    if not data:
        st.error("Failed to parse AI response. Raw output:")
        st.code(raw_output)
        return CodeAgentResponse(
            task_description="Error: Could not parse AI output",
            language=language,
            code="",
            explanation="",
            debug_notes="",
            optimization_suggestions="",
            best_practices=[],
            references=[]
        )

    return CodeAgentResponse(**data)

# --- Streamlit UI ---
st.set_page_config(page_title="ByteBot - AI Coding Agent", page_icon="🤖", layout="wide")
st.markdown("""
<style>
body { background: #FDFDFD; color: #1F2937; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.stButton>button { background: linear-gradient(90deg, #4ADE80, #22D3EE); color: #FFFFFF; font-weight: bold; border-radius: 12px; padding: 0.6em 1.2em; box-shadow: 2px 2px 10px rgba(0,0,0,0.15); transition: all 0.2s ease; }
.stButton>button:hover { transform: scale(1.05); }
.stTextInput>div>div>input { background-color: #F3F4F6; color: #111827; border-radius: 10px; padding: 0.5em; border: 1px solid #D1D5DB; }
.stMarkdown { color: #111827; }
.card { background-color: #FFFFFF; padding: 1.2em; margin-bottom: 1em; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); transition: transform 0.2s ease; }
.card:hover { transform: translateY(-5px); }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🤖 ByteBot - AI Coding Agent")
st.markdown("Write, debug, explain, and optimize code in any programming language. Ask complex coding tasks and get interactive guidance.")

# Input fields
task = st.text_area("Describe your coding task or problem")
language = st.text_input("Programming Language (Python, C, C++, Java, JavaScript, etc.)")
mode = st.selectbox("Select Mode", ["write", "debug", "explain", "optimize"])

if st.button("Execute Task"):
    if task.strip() == "" or language.strip() == "":
        st.warning("⚠️ Please enter both task description and programming language.")
    else:
        with st.spinner("Processing your coding task... ⏳"):
            agent_response = asyncio.run(coding_agent(task, language, mode))

        # Display results
        st.markdown(f"### 📝 Task Description")
        st.markdown(f"<div class='card'>{agent_response.task_description}</div>", unsafe_allow_html=True)

        st.markdown(f"### 💻 Code ({agent_response.language})")
        st.markdown(f"<div class='card'><pre>{agent_response.code}</pre></div>", unsafe_allow_html=True)

        st.markdown(f"### 📖 Explanation")
        st.markdown(f"<div class='card'>{agent_response.explanation}</div>", unsafe_allow_html=True)

        st.markdown(f"### 🐞 Debug Notes")
        st.markdown(f"<div class='card'>{agent_response.debug_notes}</div>", unsafe_allow_html=True)

        st.markdown(f"### ⚡ Optimization Suggestions")
        st.markdown(f"<div class='card'>{agent_response.optimization_suggestions}</div>", unsafe_allow_html=True)

        st.markdown(f"### ✅ Best Practices")
        for tip in agent_response.best_practices:
            st.markdown(f"<div class='card'>- {tip}</div>", unsafe_allow_html=True)

        st.markdown(f"### 🌐 References")
        for ref in agent_response.references:
            st.markdown(f"<div class='card'>- {ref}</div>", unsafe_allow_html=True)

# --- Creator Credit ---
st.markdown("""
<div style='text-align:center; margin-top:2em; padding:1em; color:#6B7280; font-size:0.9em;'>
Created by <b>Tarif</b>
</div>
""", unsafe_allow_html=True)
