# Controlled list of skills used for consistent resume matching.
import re


SKILL_DICTIONARY = [
	"Python",
	"SQL",
	"C",
	"C++",
	"Java",
	"JavaScript",
	"Machine Learning",
	"Deep Learning",
	"Scikit-learn",
	"Pandas",
	"NumPy",
	"OpenCV",
	"TensorFlow",
	"PyTorch",
	"NLP",
	"Computer Vision",
	"LLM",
	"RAG",
	"Prompt Engineering",
	"Transformers",
	"FAISS",
	"FastAPI",
	"Streamlit",
	"Docker",
	"Kubernetes",
	"Git",
	"GitHub",
	"Linux",
	"AWS",
	"Azure",
	"PostgreSQL",
]


def extract_skills(text):
	"""Return known skills found in text, preserving dictionary order."""
	if not text:
		return []

	# Lowercase once so matching works regardless of the input's capitalization.
	lowercase_text = text.lower()
	return [
		skill
		for skill in SKILL_DICTIONARY
		if re.search(
			r"(?<![a-z0-9+#])" + re.escape(skill.lower()) + r"(?![a-z0-9+#])",
			lowercase_text,
		)
	]
