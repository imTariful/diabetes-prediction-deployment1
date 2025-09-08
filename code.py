import os
import asyncio
import json
import re
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=API_KEY)

# Prompt template
PROMPT_TEMPLATE = """
You are a professional coding assistant. 
Perform the following tasks on the given code:

1. Analyze the code for correctness.
2. Find and fix all errors (if any).
3. Explain the code line by line in simple terms.
4. Suggest optimizations or improvements.

OUTPUT FORMAT (JSON ONLY, no extra text):
{{
    "fixed_code": "<the fully corrected code as a string>",
    "explanation": "<line-by-line or section explanation in bullets or numbering>",
    "optimizations": "<suggested improvements, readability, performance, etc.>"
}}

CODE TO ANALYZE:
\"\"\"
{code}
\"\"\"
"""

# Function to parse AI response safely
def parse_ai_response(raw_output: str):
    raw_output = raw_output.strip()
    if raw_output.startswith("```"):
        raw_output = "\n".join(raw_output.split("\n")[1:-1]).strip()
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        # Backup: extract JSON-looking text
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Cannot parse AI response as JSON:\n{raw_output}")
    return data

# Async function to get AI response
async def analyze_code(code: str):
    prompt = PROMPT_TEMPLATE.format(code=code)
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500
    )
    raw_output = response.choices[0].message.content
    data = parse_ai_response(raw_output)
    return data

# Streamlit UI
st.set_page_config(page_title="AI Coding Assistant", layout="wide")
st.title("🛠️ AI Coding Assistant")
st.write("Enter your code below to debug, explain, and optimize it.")

user_code = st.text_area("Paste your code here", height=300)

if st.button("Analyze Code"):
    if not user_code.strip():
        st.warning("Please enter some code to analyze.")
    else:
        with st.spinner("Analyzing your code..."):
            try:
                result = asyncio.run(analyze_code(user_code))
                
                st.subheader("✅ Fixed Code")
                st.code(result["fixed_code"], language="python")
                
                st.subheader("📖 Explanation")
                st.markdown(result["explanation"].replace("\n", "  \n"))
                
                st.subheader("⚡ Optimizations / Suggestions")
                st.markdown(result["optimizations"].replace("\n", "  \n"))
                
            except Exception as e:
                st.error(f"Error: {e}")
