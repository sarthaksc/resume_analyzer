# crew/agents.py

from crewai import Agent
from utils.llm_config import get_llm
from crew.tools.resume_match_tool import MatchResumesTool
from langchain.tools import Tool

extractor_llm = get_llm(model="gpt-3.5-turbo")  # lightweight, fast
matcher_llm = get_llm(model="gpt-4o")           # for similarity + matching
coach_llm = get_llm(model="gpt-4o")      # suggestions can be lighter too
match_resumes_tool = MatchResumesTool()

doc_extractor_agent = Agent(
    role="Resume Extractor",
    goal="Extract skills, experience, and target companies from user resume",
    backstory=(
        "You're skilled at analyzing resumes and identifying candidate strengths, job titles, and their target companies."
    ),
    verbose=True,
    llm=extractor_llm,
)

# Agent 2: Matcher - finds similar resumes from company folders
resume_matcher_agent = Agent(
    role="Resume Matcher",
    goal="Find similar resumes from companies aligned with the user’s interests and target domain",
    backstory=(
        "You specialize in comparing the user's resume with other resumes submitted to top companies, using semantic similarity to find the best match."
    ),
    verbose=True,
    llm=matcher_llm,
    tools=[match_resumes_tool]  # tool will be passed resume_text and company as args
)

# Agent 3: Coach - suggests improvements
resume_coach_agent = Agent(
    role="Resume Coach",
    goal="Generate actionable suggestions to improve the user's resume to increase selection chances",
    backstory=(
        "You are a resume coach with deep knowledge of what top companies look for in successful resumes."
    ),
    verbose=True,
    llm=coach_llm,
)
