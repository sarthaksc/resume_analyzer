import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from resume_parser.parser import parse_pdf_to_text
from crew.crew import run_resume_crew, extract_first_agent_output
from dotenv import load_dotenv
from recommendation.job_faiss_recommender import JobFaissRecommender
load_dotenv()
os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"
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
                suggestions= run_resume_crew(resume_text, target_company)
                structured_resume=extract_first_agent_output(suggestions)

                print("\n🖨️ Suggestions type:", type(suggestions))
                print("📄 Suggestions content:\n", suggestions)
                print(type(suggestions))

                st.success("✅ Suggestions ready!")
                st.markdown("### ✨ Resume Improvement Suggestions")
                st.markdown(suggestions, unsafe_allow_html=True)


                st.markdown("---")
                st.markdown("### 💼 Recommended Jobs Based on Your Resume")

                recommender = JobFaissRecommender()  # Initialize once
                print(structured_resume)
                jobs = recommender.recommend_jobs(structured_resume, top_k=5)

                print("\n🎯 Top Job Recommendations:\n")
                for i, job in enumerate(jobs, 1):
                    print(f"\n🔹 Job {i}:")
                    print(f"🏢 {job['company']} - {job['title']}")
                    print(f"📍 {job['location']} | 💰 {job['salary']} | 🧠 Score: {job['score']}")
                    print(f"📝 Description: {job['description'][:500]}...")  # trimmed for display

                if jobs:
                    st.markdown("## 💼 Recommended Jobs Based on Your Resume")

                    for i, job in enumerate(jobs, 1):
                        if job is None:
                            st.warning(f"⚠️ Skipped Job {i}: Job is None.")
                            continue

                        try:
                            print(f"\n🔍 [DEBUG] Job {i} raw object: {job}")  # Debug print

                            # Ensure it's a dictionary
                            if not isinstance(job, dict):
                                st.warning(f"⚠️ Skipped Job {i}: Job is not a dictionary.")
                                continue

                            title = job.get("title") or "Unknown Title"
                            company = job.get("company") or "Unknown Company"
                            location = job.get("location") or "Unknown Location"
                            salary = job.get("salary") or "Not Provided"
                            score = job.get("score") or "N/A"

                            description = job.get("description")
                            if not isinstance(description, str):
                                description = "No description available"

                            # st.markdown(f"""
                            # ### 🔹 Job {i}
                            # **🏢 {company} - {title}**
                            # 📍 **Location**: {location}
                            # 💰 **Salary**: {salary}
                            # 🧠 **Similarity Score**: {score}
                            # 📝 **Description**: {description[:500]}...
                            # """)

                            with st.container():
                                st.markdown(f"""
                                <div style="padding: 15px; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 25px; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);">
                                    <h4 style="color: #3366cc;">🔹 Job {i}: {title}</h4>
                                    <p><strong>🏢 Company:</strong> {company}</p>
                                    <p><strong>📍 Location:</strong> {location}</p>
                                    <p><strong>💰 Salary:</strong> {salary}</p>
                                    <p><strong>🧠 Similarity Score:</strong> <code>{score}</code></p>
                                    <p><strong>📝 Description:</strong> {description[:500]}{'...' if len(description) > 500 else ''}</p>
                                </div>
                                """, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"❌ Error rendering Job {i}: {e}")
                else:
                    st.warning("⚠️ No jobs to display.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

