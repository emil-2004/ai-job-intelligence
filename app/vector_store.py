"""Simple FAISS vector search over AI and software engineering concepts."""

import faiss
import numpy as np

from app.matcher import embedding_model


KNOWLEDGE_BASE = [
	"Machine Learning uses algorithms that learn patterns from data to make predictions or decisions.",
	"Deep Learning is a type of machine learning that uses multi-layer neural networks to learn complex patterns.",
	"Natural Language Processing (NLP) helps computers understand, process, and generate human language.",
	"Computer Vision enables computers to understand images and videos using techniques such as image classification and object detection.",
	"Large Language Models (LLMs) are neural networks trained on large amounts of text to understand and generate language.",
	"Retrieval Augmented Generation (RAG) retrieves relevant information from a knowledge base before an LLM generates an answer.",
	"Embeddings represent text as numeric vectors so that semantically similar content can be compared mathematically.",
	"Vector databases store and search embedding vectors to find information that is similar in meaning to a query.",
	"FastAPI is a Python web framework for building fast APIs with automatic validation and documentation.",
	"Docker packages an application and its dependencies into a portable container that can run consistently across environments.",
	"Kubernetes manages containerized applications by automating deployment, scaling, networking, and recovery.",
]


def build_vector_store():
	"""Create a FAISS index from the built-in knowledge base."""
	# Embeddings turn each document into a numeric vector representing its meaning.
	embeddings = embedding_model.encode(KNOWLEDGE_BASE)
	embeddings = np.asarray(embeddings, dtype="float32")

	# FAISS stores vectors and quickly finds the vectors closest to a query vector.
	index = faiss.IndexFlatL2(embeddings.shape[1])
	index.add(embeddings)

	# Returning the original documents lets search results remain readable.
	return index, KNOWLEDGE_BASE


def search_knowledge_base(query, index, documents, top_k=3):
	"""Return the most relevant knowledge-base documents for a query."""
	if not query or not query.strip():
		return []

	if index is None or not documents:
		return []

	# Vector search is useful because it finds related meaning, not only exact words.
	query_embedding = embedding_model.encode([query])
	query_embedding = np.asarray(query_embedding, dtype="float32")
	result_count = min(top_k, len(documents))
	_, nearest_indexes = index.search(query_embedding, result_count)

	return [documents[index] for index in nearest_indexes[0]]
