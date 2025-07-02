# crew/tasks.py

from crewai import Task

# Task 1: Extract structured data from the uploaded resume
extract_resume_task = Task(
    description=(
        "Analyze the uploaded resume text: {resume_text}\n"
        "Extract relevant information including the user's name, experience, skills, and potential target companies."
    ),
    expected_output="A structured summary containing the user's name, experience, skills, and a list of target companies.",
    agent=None  # <-- Assigned later to avoid circular import
)

# Task 2: Match the resume against company folders to find similar successful resumes
match_resume_task = Task(
    description="Use the collaborative filtering tool to find resumes most similar to the user's resume.",
    expected_output="A list of real resumes similar to the user's resume and a short justification for each match.",
    agent=None
    # input_keys=["resume_text", "target_companies"]
)


# Task 3: Provide improvement suggestions based on matched resumes
coach_resume_task = Task(
    description=(
        "Based on this resume: {resume_text} and the user's target companies: {target_companies}, "
        "suggest actionable improvements to increase their selection chances."
    ),
    expected_output="Detailed suggestions to improve the user's resume for better chances at their target companies.",
    agent=None,
    inputs=["resume_text", "matched_resumes"]
    )
