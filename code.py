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

# --- Model for Coding Assistant Output ---
class CodeResponse(BaseModel):
    problem_statement: str
    language: str
    solution: str = Field(description="Complete code solution with proper formatting and comments")
    explanation: str = Field(description="Step-by-step explanation of the code, logic, and why it works")
    complexity_notes: str = Field(description="Notes on time/space complexity if relevant")
    best_practices: List[str] = Field(description="Tips or best practices while writing similar code")
    references: List[str] = Field(description="Helpful links, documentation, or tutorials")

# --- Async Generator ---
async def generate_code(problem: str, language: str) -> CodeResponse:
    prompt = f"""
    You are Tarif's personal coding assistant and tutor. Your role is to:
    1. Write full, correct, and efficient code in {language}.
    2. Explain the code step-by-step in a simple way.
    3. Provide notes on time/space complexity and best practices.
    4. Suggest references if someone wants to explore more.

    Problem Statement:
    {problem}

    Format the response as JSON with the following fields:
    - problem_statement
    - language
    - solution
    - explanation
    - complexity_notes
    - best_practices (list of 3–5 points)
    - references (list of 2–4 links or tutorials)
    """

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )

    raw_output = response.choices[0].message.content

    try:
        if raw_output.strip().startswith("```"):
            raw_output = "\n".join(raw_output.strip().split("\n")[1:-1])
        data = json.loads(raw_output)
        return CodeResponse(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        st.error("Failed to parse AI response. Here is the raw output:")
        st.code(raw_output)
        raise e

# --- Streamlit Stylish UI ---
st.set_page_config(page_title="Tarif - Coding Assistant", page_icon="💻", layout="wide")
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

st.markdown("## 💻 Tarif - Coding Assistant")
st.markdown("Your AI companion for coding. Write, debug, and understand complex programs with step-by-step explanations.")

# Input fields
problem = st.text_input("Enter the programming problem or request")
language = st.text_input("Programming language (Python, C, C++, Java, JavaScript, etc.)")

if st.button("Generate Code"):
    if problem.strip() == "" or language.strip() == "":
        st.warning("⚠️ Please enter both the problem statement and programming language.")
    else:
        with st.spinner("Generating code and explanation... ⏳"):
            code_response = asyncio.run(generate_code(problem, language))

        # Display in stylish cards
        st.markdown(f"### 📝 Problem Statement")
        st.markdown(f"<div class='card'>{code_response.problem_statement}</div>", unsafe_allow_html=True)

        st.markdown(f"### 💡 Solution ({code_response.language})")
        st.markdown(f"<div class='card'><pre>{code_response.solution}</pre></div>", unsafe_allow_html=True)

        st.markdown(f"### 📖 Explanation")
        st.markdown(f"<div class='card'>{code_response.explanation}</div>", unsafe_allow_html=True)

        st.markdown(f"### ⏱ Complexity Notes")
        st.markdown(f"<div class='card'>{code_response.complexity_notes}</div>", unsafe_allow_html=True)

        st.markdown(f"### ✅ Best Practices")
        for tip in code_response.best_practices:
            st.markdown(f"<div class='card'>- {tip}</div>", unsafe_allow_html=True)

        st.markdown(f"### 🌐 References")
        for ref in code_response.references:
            st.markdown(f"<div class='card'>- {ref}</div>", unsafe_allow_html=True)

# --- Creator Credit ---
st.markdown(
    """
    <div style='text-align:center; margin-top:2em; padding:1em; color:#6B7280; font-size:0.9em;'>
        Created by <b>Tarif</b>
    </div>
    """,
    unsafe_allow_html=True
)
