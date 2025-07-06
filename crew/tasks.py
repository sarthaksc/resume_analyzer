# crew/tasks.py

from crewai import Task

# Task 1: Extract structured data from the uploaded resume
extract_resume_task = Task(
    description=(
        "Analyze the uploaded resume text: {resume_text}\n"
        "Extract relevant information including the user's name, experience, skills, and potential target companies."
    ),
    expected_output="A structured summary containing the user's name, experience, skills, and a list of target companies.",
    agent=None,  # <-- Assigned later to avoid circular import
    output_key="structured_resume"
)

# Task 2: Match the resume against company folders to find similar successful resumes
# match_resume_task = Task(
#     description="Use the collaborative filtering tool to find resumes most similar to the user's resume.",
#     expected_output="A list of real resumes similar to the user's resume and a short justification for each match.",
#     agent=None,
#     output_key="matched_resumes",
#     input_keys=["resume_text", "target_companies"]
# )

match_resume_task = Task(
    description=(
        "Given the following resume:\n\n{resume_text}\n\n"
        "And the user's target companies:\n\n{target_companies}\n\n"
        "Your job is to use the `match_resumes` tool to find resumes from the dataset "
        "that are most similar to this resume using collaborative filtering. "
        "Call the tool using the user's resume text as input. "
        "You must return the top 3–5 matched resumes with a similarity score and a brief justification for each."
    ),
    expected_output="A list of real resumes similar to the user's resume and a short justification for each match.",
    agent=None,
    output_key="matched_resumes",
    input_keys=["resume_text", "target_companies"]
)


# Task 3: Provide improvement suggestions based on matched resumes
# coach_resume_task = Task(
#     description=(
#         "Based on the structured resume below:\n{structured_resume}\n\n"
#         "and these matched resumes from similar successful candidates:\n{matched_resumes}\n\n"
#         "provide detailed and actionable improvement suggestions to improve the candidate's chances of getting selected."
#     ),
#     expected_output="Detailed suggestions to improve the user's resume for better chances at their target companies.",
#     agent=None,
#     input_keys=["structured_resume", "matched_resumes"]
#     )

coach_resume_task = Task(
    description=("Using the provided structured resume and matched resumes (accessible via context), "
        "generate detailed, specific suggestions to improve the candidate's resume. "
        "Base your suggestions strictly on the actual content of the matched resumes."),
    expected_output="Detailed suggestions to improve the user's resume for better chances at their target companies.",
    agent=None,
    input_keys=["structured_resume", "matched_resumes"]  # Inputs from previous tasks
)

