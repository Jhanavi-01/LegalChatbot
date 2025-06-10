import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import chromadb
import ollama

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="legal_cases")

# Function to extract text from PDFs (including OCR for scanned images)
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")  # Handle file streams
    text = ""

    for page in doc:
        page_text = page.get_text("text")
        text += page_text.strip() + "\n"

        # If no text extracted, apply OCR to the whole page
        if not page_text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img) + "\n"

        # Extract and OCR images inside the page
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes))
            text += pytesseract.image_to_string(pil_img) + "\n"

    return text.strip()

# Function to retrieve similar cases from ChromaDB
def retrieve_similar_cases(query_text, n_results=3):
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query_text)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results["metadatas"][0] if results else []

# Function to generate structured summaries
def generate_summary(case_text):
    prompt = f"""
    Summarize the following legal case in a structured format:

    {case_text}

    **Return the summary in this format:**
    - **Case Name:** [Name]
    - **Jurisdiction:** [Jurisdiction]
    - **Key Facts:** [Key facts of the case]
    - **Relevant Legal Provisions:** [Laws, IPC sections, legal references]
    - **Chronology of Events:** [Step-by-step timeline]
    """

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Initialize Streamlit app
st.title("📜 Legal Chatbot")

# Conversation history
if "history" not in st.session_state:
    st.session_state["history"] = []
if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = None

# File Upload
uploaded_file = st.file_uploader("Upload a case PDF", type=["pdf"])
if uploaded_file:
    st.write("📄 Processing PDF...")
    st.session_state["pdf_text"] = extract_text_from_pdf(uploaded_file)
    st.session_state["history"].append({"role": "system", "content": "📄 PDF uploaded and processed."})
    st.success("✅ PDF processed successfully!")

# Buttons for actions
col1, col2 = st.columns(2)
with col1:
    retrieve_cases_btn = st.button("🔍 Retrieve Cases Similar to This")
with col2:
    generate_summary_btn = st.button("📑 Generate Summary of Processed PDF")

# Retrieve similar cases and return their summaries
if retrieve_cases_btn:
    if not st.session_state["pdf_text"]:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        st.session_state["history"].append({"role": "user", "content": "Retrieve cases similar to this."})
        similar_cases = retrieve_similar_cases(st.session_state["pdf_text"])
        
        if similar_cases:
            for i, case in enumerate(similar_cases):
                summary = generate_summary(case["text"])
                st.session_state["history"].append({"role": "assistant", "content": f"📑 **Case {i+1} Summary:**\n{summary}"})
        else:
            st.session_state["history"].append({"role": "assistant", "content": "❌ No similar cases found."})

# Generate summary of the processed PDF
if generate_summary_btn:
    if not st.session_state["pdf_text"]:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        st.session_state["history"].append({"role": "user", "content": "Generate summary of the processed PDF."})
        summary = generate_summary(st.session_state["pdf_text"])
        st.session_state["history"].append({"role": "assistant", "content": f"📑 **Case Summary:**\n{summary}"})

# User query
user_input = st.text_area("💬 Ask a question about the uploaded case or related legal matters:")
if st.button("Submit Query"):
    if not user_input:
        st.warning("⚠️ Please enter a question.")
    else:
        st.session_state["history"].append({"role": "user", "content": user_input})

        # Context-aware response
        prompt = "The following is a conversation with a legal assistant AI. Use the context below to answer the user's question:\n\n"
        for msg in st.session_state["history"]:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        answer = response["message"]["content"]

        st.session_state["history"].append({"role": "assistant", "content": answer})

# Display conversation history
st.subheader("📜 Conversation History")
for msg in st.session_state["history"]:
    if msg["role"] == "user":
        st.markdown(f"**🧑 User:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**🤖 Chatbot:** {msg['content']}")
