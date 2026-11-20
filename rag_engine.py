from pathlib import Path

import faiss
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def load_pdf(pdf_path):
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {path}. Put the PDF in the app folder "
            "or pass its correct relative path."
        )

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages).strip()
    if not full_text:
        raise ValueError("The PDF contains no readable text.")
    return full_text


def create_chunks(full_text, chunk_size=500, overlap=100):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1.")

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):
    if not chunks:
        raise ValueError("Cannot create embeddings without text chunks.")

    embed_model = SentenceTransformer(model_name)
    vectors = embed_model.encode(chunks, convert_to_numpy=True)
    vectors = vectors.astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, embed_model


def load_base(pdf_path="01_College_Information.pdf"):
    full_text = load_pdf(pdf_path)
    chunks = create_chunks(full_text)
    index, embed_model = build_vectors(chunks)
    return index, embed_model, chunks


def retrieve_content(question, embed_model, chunks, index, num_chunks=3):
    if not question.strip():
        raise ValueError("Question cannot be blank.")
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least one.")

    query_vector = embed_model.encode([question], convert_to_numpy=True)
    query_vector = query_vector.astype("float32")
    faiss.normalize_L2(query_vector)
    _, vector_indices = index.search(query_vector, min(num_chunks, len(chunks)))

    return "\n\n".join(
        chunks[i] for i in vector_indices[0] if i != -1
    )


def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=3,
    model_name="openai/gpt-oss-120b",
):
    if not api_key:
        raise ValueError("Groq API key is missing.")

    background = retrieve_content(
        question, embed_model, chunks, index, num_chunks=num_chunks
    )
    prompt = f"""You are a helpful assistant for Shridevi Institute of Engineering and Technology.
Answer using only the document background below. Do not invent or assume facts.
If the answer is not in the background, say: Information is not available in the college document.

Document background:
{background}

User question:
{question}

Answer:"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content or "No answer was returned."
