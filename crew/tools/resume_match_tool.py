# # crew/tools/resume_match_tool.py
# # from langchain.tools import Tool
# # from crewai_tools import tool
# from crew.matcher import find_similar_resumes
# from crewai.tools import BaseTool
# import json
# from typing import Type
# from pydantic import BaseModel
# from resume_parser.parser import infer_domains_from_resume
#
# with open("data/company_domains.json", "r") as f:
#     COMPANY_DOMAINS = json.load(f)
#
# class MatchResumesInput(BaseModel):
#     resume_text: str
#     company: str
#
# class MatchResumesTool(BaseTool):
#     name: str = "match_resumes"
#     description: str = "Match user resume with resumes from similar domains and companies"
#     args_schema: type = MatchResumesInput
#
#     def _run(self, resume_text: str, company: str) -> str:
#             if not resume_text or not company:
#                 return "❌ Missing resume text or company. Cannot proceed with matching."
#             resume_domains = infer_domains_from_resume(resume_text)
#             company_domains = COMPANY_DOMAINS.get(company, [])
#
#             if not resume_domains:
#                 return "❌ Could not infer any domain from the resume."
#
#             if not set(resume_domains) & set(company_domains):
#                 match_domains = set(resume_domains)
#                 matching_companies = [
#                     comp for comp, domains in COMPANY_DOMAINS.items()
#                     if set(domains) & match_domains
#                 ]
#
#                 if not matching_companies:
#                     return f"❌ Your resume doesn't align with {company}'s domain and no similar companies were found."
#
#                 suggestions = ""
#                 for alt_company in matching_companies:
#                     matches = find_similar_resumes(resume_text, alt_company)
#                     if not matches:
#                         continue
#                     for fname, score, snippet in matches:
#                         suggestions += f"\n📄 {fname} (Score: {score:.2f})\nSnippet:\n{snippet}\n\n"
#
#                 return (
#                     f"⚠️ Your resume domain doesn't align with {company}'s (domains: {company_domains}).\n"
#                     f"🧠 However, here are better matches from your domain ({', '.join(resume_domains)}):\n\n{suggestions}"
#                 )
#
#             # Normal case: domains overlap
#             matches = find_similar_resumes(resume_text, company)
#             if not matches:
#                 return f"No similar resumes found for {company} in domain(s): {', '.join(resume_domains)}."
#
#             output = f"✅ Matched resumes from {company}:\n"
#             for fname, score, snippet in matches:
#                 output += f"\n📄 {fname} (Score: {score:.2f})\nSnippet:\n{snippet}\n\n"
#
#             return output
#
#
#
#     def _arun(self, *args, **kwargs):
#         raise NotImplementedError("Async version not implemented.")
#
# # match_resumes_tool = Tool(
# #     name="match_resumes",
# #     description="Find resumes from the dataset that best match the provided resume text and company name",
# #     func=match_resumes
# # )


import os
from crewai.tools import BaseTool
from matcher.matcher import ResumeMatcher

# Instantiate and build the index only once
matcher = ResumeMatcher()
matcher.build_index("data/company resumes")  # Make sure this path is correct

class MatchResumesTool(BaseTool):
    name: str = "match_resumes"
    description: str = "Find similar resumes from the dataset using collaborative filtering"

    def _run(self, resume_text: str) -> str:
        print("[DEBUG] Entered _run method of MatchResumesTool")
        try:
            results = matcher.find_matches(resume_text)
            print(f"[DEBUG] Retrieved {len(results)} matches")
        except Exception as e:
            print(f"[ERROR] Failed to find matches: {e}")
            return f"[ERROR] Failed to find matches: {e}"
        if len(results)==0:
            print("[DEBUG] No similar resumes found.")
            return "No similar resumes found."
        output = ""
        for i, result in enumerate(results):
            print(f"[DEBUG] Processing result {i}: {result}")
            try:
                company, fname, score = result
                print(f"[DEBUG] Raw score: {score} (type: {type(score)})")

                if hasattr(score, "item"):
                    print("[DEBUG] score is a tensor, converting with .item()")
                    score = score.item()  # Converts single-value tensor to float

                print(f"[DEBUG] Final score as float: {score}")
                output += f"\n📄 **{company} - {fname}** (Similarity: {score:.2f})\n"

            except Exception as e:
                print(f"[ERROR] Failed processing result {i}: {e}")
                return f"[ERROR] Failed processing result {i}: {e}"

        print("[DEBUG] Finished processing all matches.")
        return output
