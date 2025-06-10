from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

# Load dataset (PCR candidates contain previous cases)
ds = load_dataset("Exploration-Lab/IL-TUR", "pcr")

# Initialize embedding model & ChromaDB
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="legal_cases")

# Store embeddings
for case in ds["train_candidates"]:
    case_id = case["id"]
    case_text = " ".join(case["text"])  # Join sentences into full text
    embedding = model.encode(case_text).tolist()  # Convert to embedding
    
    collection.add(
        ids=[case_id],  # Store case ID
        embeddings=[embedding],  # Store embedding
        metadatas=[{"text": case_text}]  # Store full text for retrieval
    )

print("Stored all candidate cases in ChromaDB!")

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# User uploads a case PDF
pdf_text = extract_text_from_pdf("/Users/jibines/Desktop/Legalbot.ai/dataset/Final SLP Mukesh kumar.pdf")

# Convert the extracted text into an embedding
query_embedding = model.encode(pdf_text).tolist()

print("User uploaded pdf converted to embeddings")

# Search for similar cases in ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3  # Get top 3 similar cases
)

# Print retrieved case summaries
for i, res in enumerate(results["metadatas"][0]):
    print(f"\n🔹 Similar Case {i+1}:")
    print(res["text"][:1000])  # Print first 1000 characters


import ollama

def generate_summary(case_text):
    prompt = f"""
    Summarize the following legal case in a structured format:
    
    {case_text}
    
    Provide:
    1. Case Name
    2. Jurisdiction
    3. Key Facts
    4. Relevant Legal Provisions
    5. Chronology of Events
    """
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Generate summaries for the retrieved cases
for res in results["metadatas"][0]:
    print("\n🔹 Summary:")
    print(generate_summary(res["text"]))
