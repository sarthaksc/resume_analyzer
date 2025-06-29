# crew/agents.py

from crewai import Agent
from utils.llm_config import get_llm
llm=get_llm()

# Agent 1: Extractor - turns resumes into structured data
doc_extractor_agent = Agent(
    role="Resume Extractor",
    goal="Extract skills, experience, and target companies from user resume",
    backstory=(
        "You're skilled at analyzing resumes and identifying candidate strengths, job titles, and their target companies."
    ),
    verbose=True,
    llm=llm,
)

# Agent 2: Matcher - finds similar resumes from company folders
resume_matcher_agent = Agent(
    role="Resume Matcher",
    goal="Find resumes from the dataset that best match the user's resume and targeted companies",
    backstory=(
        "You specialize in comparing resumes using semantic similarity and helping candidates align with company standards."
    ),
    verbose=True,
    llm=llm,
)

# Agent 3: Coach - suggests improvements
resume_coach_agent = Agent(
    role="Resume Coach",
    goal="Generate actionable suggestions to improve the user's resume to increase selection chances",
    backstory=(
        "You are a resume coach with deep knowledge of what top companies look for in successful resumes."
    ),
    verbose=True,
    llm=llm,
)
