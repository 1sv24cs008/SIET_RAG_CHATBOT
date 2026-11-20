import faiss
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError("No readable text found in the PDF.")

    return full_text


def create_chunks(full_text, chunk_size=500, overlap=100):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

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
        raise ValueError("Cannot create embeddings without chunks.")

    embed_model = SentenceTransformer(model_name)
    embed_vectors = embed_model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(embed_vectors.shape[1])
    index.add(embed_vectors)

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
        raise ValueError("Request at least one chunk.")

    query_embed = embed_model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_embed)

    _, vec_indices = index.search(
        query_embed,
        min(num_chunks, len(chunks)),
    )

    background = ""
    for i in vec_indices[0]:
        if i != -1:
            background += chunks[i] + "\n\n"

    return background


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
        raise ValueError("API key is missing.")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks=num_chunks,
    )

    prompt = f"""You are a professional AI assistant.
Answer the user's question using only the document background below.
Do not invent or assume facts. If the information is unavailable, say
"Information missing."

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

    return response.choices[0].message.content
