"""Streamlit interface for the AI Job Intelligence matching pipeline."""

import os
import re
import tempfile

import streamlit as st

from app.parser import extract_text_from_pdf
from app.rag import analyze_resume_for_job


st.set_page_config(
	page_title="AI Job Intelligence & Resume Matcher",
	page_icon=":mag:",
	layout="wide",
)


def display_ai_analysis(analysis):
	"""Display Gemini's response under the requested analysis headings."""
	section_names = [
		"Overall Assessment",
		"Strengths",
		"Missing Skills",
		"Recommended Learning Areas",
		"Role-specific Interview Questions",
	]
	section_pattern = "|".join(re.escape(name) for name in section_names)
	pattern = re.compile(
		r"(?:^|\n)\s*#+\s*("
		+ section_pattern
		+ r")\s*:?[ \t]*\n?(.*?)(?=\n\s*#+\s*(?:"
		+ section_pattern
		+ r")\s*:?[ \t]*\n?|$)",
		re.IGNORECASE | re.DOTALL,
	)
	matches = pattern.findall(analysis)

	if not matches:
		st.markdown(analysis)
		return

	for heading, content in matches:
		st.markdown(f"#### {heading.title()}")
		st.markdown(content.strip())


with st.sidebar:
	st.title("AI Job Intelligence")
	st.write(
		"Upload a resume and paste a job description to compare skills, "
		"retrieve relevant knowledge, and generate an evidence-based analysis."
	)
	st.caption("Your resume is processed for the current analysis session.")


st.title("AI Job Intelligence & Resume Matcher")
st.write("Evaluate how closely a candidate profile aligns with a target role.")

resume_file = st.file_uploader("Upload Resume", type=["pdf"])
job_description = st.text_area(
	"Paste Job Description",
	height=240,
	placeholder="Paste the complete job description here...",
)

if st.button("Analyze Resume", type="primary", use_container_width=True):
	if resume_file is None:
		st.error("Please upload a PDF resume before starting the analysis.")
	elif not job_description.strip():
		st.error("Please paste a job description before starting the analysis.")
	else:
		temporary_path = None
		try:
			# Save the upload briefly because the parser accepts a PDF path.
			with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
				temporary_file.write(resume_file.getvalue())
				temporary_path = temporary_file.name

			with st.spinner("Reading the resume and generating your analysis..."):
				resume_text = extract_text_from_pdf(temporary_path)
				if not resume_text.strip():
					st.error("The uploaded PDF did not contain readable text.")
				else:
					result = analyze_resume_for_job(resume_text, job_description)

					st.divider()
					st.header("Match Overview")
					match_result = result["match_result"]
					metric_columns = st.columns(3)
					metric_columns[0].metric(
						"Overall Match Percentage",
						f"{match_result.get('overall_match_percentage', 0):.1f}%",
					)
					metric_columns[1].metric(
						"Semantic Similarity",
						f"{match_result.get('semantic_similarity', 0) * 100:.1f}%",
					)
					metric_columns[2].metric(
						"Skill Match Percentage",
						f"{match_result.get('skill_match_percentage', 0) * 100:.1f}%",
					)

					st.subheader("Matching Skills")
					matching_skills = match_result.get("matching_skills", [])
					st.success(", ".join(matching_skills) if matching_skills else "None identified")

					st.subheader("Missing Skills")
					missing_skills = match_result.get("missing_skills", [])
					st.warning(", ".join(missing_skills) if missing_skills else "None identified")

					with st.expander("Retrieved Knowledge / RAG Context"):
						context = result.get("retrieved_context", [])
						if context:
							for document in context:
								st.markdown(f"- {document}")
						else:
							st.write("No additional knowledge context was retrieved.")

					st.header("AI Job Analysis")
					display_ai_analysis(result.get("ai_analysis", "No analysis was returned."))
		except Exception:
			st.error("The analysis could not be completed. Please try another PDF or try again.")
		finally:
			if temporary_path and os.path.exists(temporary_path):
				os.remove(temporary_path)
