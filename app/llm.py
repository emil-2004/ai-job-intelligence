"""LLM integration for generating job-match analysis."""

import json
import os

from google import genai
from dotenv import load_dotenv


# Load values from the local .env file without putting secrets in source code.
load_dotenv()


def generate_job_analysis(
	resume_text,
	job_description,
	match_result,
	retrieved_context,
):
	"""Generate a structured analysis from the resume and job-match data."""
	if not resume_text or not job_description:
		return "Error: resume text and job description are required."

	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		return "Error: GEMINI_API_KEY is not configured in the environment."

	match_result = match_result or {}
	retrieved_context = retrieved_context or []

	# Keep every source of information visible to the model in one structured prompt.
	prompt_data = {
		"resume_text": resume_text,
		"job_description": job_description,
		"overall_match_percentage": match_result.get("overall_match_percentage", 0),
		"semantic_similarity": match_result.get("semantic_similarity", 0),
		"skill_match_percentage": match_result.get("skill_match_percentage", 0),
		"matching_skills": match_result.get("matching_skills", []),
		"missing_skills": match_result.get("missing_skills", []),
		"retrieved_context": retrieved_context,
	}

	prompt = (
		"Analyze the candidate for the role using only the supplied information. "
		"Do not invent or assume experience, qualifications, projects, or skills "
		"that are not explicitly present in the resume. Return a structured analysis "
		"with these sections: Overall assessment, Strengths, Missing skills, "
		"Recommended learning areas, and Role-specific interview questions.\n\n"
		f"Supplied information:\n{json.dumps(prompt_data, indent=2)}"
	)

	try:
		# The client uses the key from the environment for authenticated requests.
		client = genai.Client(api_key=api_key)
		model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
		# Use a normal chat message; no tools or function-calling configuration is needed.
		chat = client.chats.create(model=model_name)
		response = chat.send_message(message=prompt)

		return response.text
	except Exception as e:
		# Include diagnostic details while ensuring the API key is never returned.
		details = str(e).replace(api_key, "[REDACTED]")
		return (
			"Error: Gemini request failed.\n"
			f"Type: {type(e).__name__}\n"
			f"Details: {details}"
		)
