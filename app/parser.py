"""Utilities for extracting text from resume and job-description files."""

import pymupdf


def extract_text_from_pdf(pdf_path):
	"""Extract and combine the text from every page in a PDF file.

	Returns an empty string when the file cannot be opened or read.
	"""
	try:
		# The context manager closes the document even if page extraction fails.
		with pymupdf.open(pdf_path) as pdf_document:
			page_text = [page.get_text() for page in pdf_document]

		return "\n".join(page_text)
	except (OSError, RuntimeError, ValueError):
		# Invalid paths and malformed or unreadable PDFs are handled gracefully.
		return ""
