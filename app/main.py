import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from resume_parser.parser import parse_pdf_to_text
from crew.crew import run_resume_crew
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("🚀 AI-Powered Resume + Job Matching")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

target_company = st.selectbox("Which company are you targeting?", [
    'AB InBev', 'Accenture Japan', 'Amazon', 'American Express', 'Analog Devices',
    'Apple India', 'Bajaj Auto', 'BlackRock', 'Boston Consulting Group', 'Citrix Systems',
    'Cohesity', 'Dream11', 'Eaton', 'EXL Service', 'Exxonmobil', 'Flipkart',
    'Goldman Sachs', 'Google', 'Groww', 'ICICI Bank', 'Intel', 'ITC',
    'Jaguar Land Rover Ltd', 'Jio Platforms', 'JP Morgan Chase & Co', 'MaxLinear',
    'Microsoft', 'Morgan Stanley', 'Paytm', 'Plutus Research Private Limited',
    'Praxis Global Alliance', 'PricewaterhouseCoopers Pvt. Ltd. (PwC)', 'Quadeye',
    'SAP Labs', 'Sprinklr', 'Tata Project Ltd', 'Traceable.ai', 'Uber India',
    'Wells Fargo', 'Zomato'
])

if uploaded_file is not None:
    if st.button("Analyze Resume"):
        with st.spinner("Parsing resume and running CrewAI agents..."):
            try:
                resume_text = parse_pdf_to_text(uploaded_file)
                suggestions = run_resume_crew(resume_text, target_company)

                print("\n🖨️ Suggestions type:", type(suggestions))
                print("📄 Suggestions content:\n", suggestions)
                print(type(suggestions))

                st.success("✅ Suggestions ready!")
                st.markdown("### ✨ Resume Improvement Suggestions")
                st.markdown(suggestions, unsafe_allow_html=True)
                print("\n--- DEBUG ---")
                print("Type of suggestions:", type(suggestions))
                print("Content preview:\n", suggestions[:500])

            except Exception as e:
                st.error(f"❌ Error: {e}")

