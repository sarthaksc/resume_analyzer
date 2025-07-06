
# 📄 Resume Analyzer + 💼 Job Recommender System

An AI-powered web application that analyzes resumes, suggests improvements based on successful examples, and recommends relevant job listings using semantic similarity and FAISS vector search. The system uses multi-agent collaboration, LLMs, and real-time Streamlit frontend.

---

## 🚀 Features

- 🤖 **Multi-Agent Resume Analysis Pipeline**  
  Built with [CrewAI](https://github.com/joaomdmoura/crewai) using GPT-4o agents:
  - **Resume Extractor** – Parses key information like skills, projects, experience, and domains.
  - **Resume Matcher** – Finds semantically similar resumes from successful candidates.
  - **Resume Coach** – Suggests improvements using matched resume patterns.

- 📊 **Job Recommender Engine (FAISS + Embeddings)**  
  Retrieves relevant job postings by comparing semantic similarity between the resume and a job database of 19,000+ entries.

- 🧠 **Semantic Search with SentenceTransformers**  
  Uses `all-MiniLM-L6-v2` model for fast and accurate embedding generation.

- 🌐 **Streamlit Frontend**  
  Interactive GUI with:
  - Resume upload and analysis
  - Similar resume suggestions
  - Personalized job recommendations

---

## 🧰 Tech Stack

| Component         | Technology                     |
|------------------|--------------------------------|
| LLMs & Agents     | GPT-4o via CrewAI              |
| Embeddings        | SentenceTransformers           |
| Similarity Search | FAISS                          |
| Backend           | Python + SQLite                |
| Frontend          | Streamlit                      |
| Vector DB         | FAISS index (.faiss) + metadata (.npy) |

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/resume-analyzer-job-recommender.git
   cd resume-analyzer-job-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Fix tokenizers conflict (if using transformers)**
   ```bash
   pip install tokenizers==0.21.0
   ```

4. **Run the application**
   ```bash
   streamlit run app/main.py
   ```

---

## 📂 Project Structure

```
├── agents.py                  # CrewAI agent definitions
├── tasks.py                   # Task definitions assigned to agents
├── crew.py                    # Crew pipeline orchestration
├── main.py                    # Streamlit frontend
├── job_recommender.py         # SQL + embedding-based recommender
├── job_faiss_recommender.py   # FAISS-based job recommender class
├── build_job_index.py         # Index builder for FAISS
├── jobs.db                    # SQLite job database
├── job_index.faiss            # FAISS index of job embeddings
├── job_meta.npy               # Metadata for quick lookup
```

---
