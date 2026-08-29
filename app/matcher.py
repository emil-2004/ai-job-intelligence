"""Text embedding and similarity utilities for job matching."""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load the embedding model once so it can be reused for multiple texts.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embedding(text):
	"""Convert text into a Sentence Transformers embedding."""
	if not text or not text.strip():
		return None

	return embedding_model.encode(text)


def calculate_similarity(text1, text2):
	"""Return the cosine similarity between two text strings."""
	embedding1 = create_embedding(text1)
	embedding2 = create_embedding(text2)

	if embedding1 is None or embedding2 is None:
		return 0.0

	# cosine_similarity expects each embedding to be a row in a 2D array.
	score = cosine_similarity([embedding1], [embedding2])[0][0]

	# Keep the public score within the requested 0-to-1 range.
	return float(max(0.0, min(1.0, score)))


def match_resume_to_job(resume_text, job_description, resume_skills, job_skills):
	"""Compare a resume with a job description and its required skills."""
	resume_skills = resume_skills or []
	job_skills = job_skills or []

	# Normalize skills for comparison while preserving the job-skill order in the result.
	resume_skill_names = {skill.strip().lower() for skill in resume_skills}
	matching_skills = []
	missing_skills = []
	seen_job_skills = set()

	for skill in job_skills:
		normalized_skill = skill.strip().lower()
		if normalized_skill in seen_job_skills:
			continue

		seen_job_skills.add(normalized_skill)
		if normalized_skill in resume_skill_names:
			matching_skills.append(skill)
		else:
			missing_skills.append(skill)

	# Skill coverage is measured against the skills required by the job.
	if job_skills:
		skill_match_percentage = len(matching_skills) / len(seen_job_skills)
	else:
		skill_match_percentage = 0.0

	semantic_similarity = calculate_similarity(resume_text, job_description)

	# Both components are ratios, so multiply the weighted result by 100 for display.
	overall_match_percentage = (
		0.60 * semantic_similarity + 0.40 * skill_match_percentage
	) * 100

	return {
		"semantic_similarity": semantic_similarity,
		"skill_match_percentage": skill_match_percentage,
		"overall_match_percentage": overall_match_percentage,
		"matching_skills": matching_skills,
		"missing_skills": missing_skills,
	}
