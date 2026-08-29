"""Orchestration for resume matching and retrieval-augmented analysis."""

from app.llm import generate_job_analysis
from app.matcher import match_resume_to_job
from app.utils import extract_skills
from app.vector_store import build_vector_store, search_knowledge_base


def analyze_resume_for_job(resume_text, job_description):
	"""Analyze a resume against a job and retrieve context for missing skills."""
	resume_text = resume_text or ""
	job_description = job_description or ""

	# Extract the controlled skill lists before calculating the match.
	resume_skills = extract_skills(resume_text)
	job_skills = extract_skills(job_description)
	match_result = match_resume_to_job(
		resume_text,
		job_description,
		resume_skills,
		job_skills,
	)

	missing_skills = match_result.get("missing_skills", [])
	retrieved_context = []

	# Build the knowledge base once, then search it for each missing skill.
	if missing_skills:
		index, documents = build_vector_store()
		seen_documents = set()

		for skill in missing_skills:
			results = search_knowledge_base(skill, index, documents)
			for document in results:
				if document not in seen_documents:
					seen_documents.add(document)
					retrieved_context.append(document)

	# The existing LLM integration receives the match data and retrieved context.
	ai_analysis = generate_job_analysis(
		resume_text,
		job_description,
		match_result,
		retrieved_context,
	)

	return {
		"resume_skills": resume_skills,
		"job_skills": job_skills,
		"match_result": match_result,
		"retrieved_context": retrieved_context,
		"ai_analysis": ai_analysis,
	}
