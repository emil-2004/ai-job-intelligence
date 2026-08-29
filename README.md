# AI Job Intelligence

AI Job Intelligence is a planned platform for analyzing job descriptions and candidate resumes, then surfacing relevant matches and actionable insights.

## Project Structure

```text
ai-job-intelligence/
├── app/
│   ├── main.py
│   ├── parser.py
│   ├── matcher.py
│   ├── llm.py
│   └── utils.py
├── frontend/
│   └── streamlit_app.py
├── data/
│   └── knowledge_base/
├── models/
├── tests/
├── .env
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

## Planned Features

- Parse resumes and job descriptions from supported document formats
- Extract skills, experience, education, and other career signals
- Match candidates with relevant job opportunities
- Generate semantic search embeddings and maintain a knowledge base
- Use large language models for summaries and recommendations
- Provide an interactive Streamlit interface
- Expose application services through a FastAPI backend

## Technology Stack

- Python 3.11
- FastAPI and Uvicorn
- Streamlit
- pandas and NumPy
- scikit-learn
- sentence-transformers and FAISS
- PyMuPDF
- python-dotenv

This project is currently scaffolded only. AI functionality will be implemented in a later phase.
