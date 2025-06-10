from datasets import load_dataset
import ollama
import chromadb

# Load dataset (PCR candidates contain previous cases)
ds = load_dataset("Exploration-Lab/IL-TUR", "pcr")

# Debug: Print first few cases to inspect structure
print("Sample case structure:", ds["train_candidates"][:3])  

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_embedding(text):
    """Fetches embedding using Ollama's Python API"""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]  # Extract the vector

# Get embedding dimension dynamically
sample_text = "Test case for dimension detection"
embedding_dim = len(get_embedding(sample_text))  # Should be 768 for nomic-embed-text

# Create collection with correct dimension
collection = chroma_client.get_or_create_collection(
    name="legal_cases",
    metadata={"hnsw:space": "cosine"},
    embedding_function=None
)

# Process only 100 cases
for i, case in enumerate(ds["train_candidates"][:100]):  
    if not isinstance(case, dict):  
        print(f"⚠️ Skipping invalid case at index {i}: {case} (Not a dictionary)")
        continue

    case_id = str(case.get("id", f"missing_id_{i}"))  # Use ID if available
    case_text = " ".join(case.get("text", []))  # Handle missing text safely

    if not case_text.strip():  # Skip empty cases
        print(f"⚠️ Skipping case {case_id} (empty text)")
        continue

    embedding = get_embedding(case_text)  # Convert text to embedding
    
    collection.add(
        ids=[case_id],  # Store dataset ID
        embeddings=[embedding],  # Store embedding
        metadatas=[{"text": case_text}]  # Store full text for retrieval
    )
    
    if (i + 1) % 10 == 0:
        print(f"✅ Processed {i + 1}/100 cases...")

print(f"✅ Stored 100 candidate cases in ChromaDB!")

