# Injustice

**Injustice** is an AI-powered legal assistant that provides guidance based on the **1999 Constitution of the Federal Republic of Nigeria**. It helps people who cannot afford legal consultation understand their rights and access accurate legal information.  

---

## Features

- AI bot with full knowledge of the 1999 Nigerian Constitution.  
- Provides answers to legal questions in simple, understandable language.  
- Ideal for individuals seeking legal information without expensive consultations.  
- RAG (Retrieval-Augmented Generation) system integrated with a vector database for fast and accurate responses.  

---

## Technologies Used

- **Python**  
- **LangChain** (for RAG and embeddings)  
- **ChromaDB** (local vector database)  
- **Google Gemini / HuggingFace Embeddings** (AI models)  
- **PostgreSQL** (backend database)  
- **FastAPI** (for backend API)  

---

## Getting Started

1. Clone the repository:  
   ```bash
   git clone https://github.com/Tchris-Hub/Injustice.git
   cd Injustice/backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your `GOOGLE_API_KEY`
4. Run the ingestion script:
   ```bash
   python -m scripts.ingest_data
   ```
5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```