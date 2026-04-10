"""
build_rag.py — Build FAISS Vector Index from Medical PDFs
==========================================================
Uses LangChain + sentence-transformers + FAISS-CPU.

Run this ONCE (or auto-runs on backend startup if index is missing):
    python build_rag.py

Reads:  backend/medical_texts/*.pdf
Writes: backend/rag_index/  (FAISS index + metadata)
"""

import os, glob, time, sys
from pathlib import Path

TEXTS_DIR = Path(__file__).parent / "medical_texts"
INDEX_DIR = Path(__file__).parent / "rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 80MB, fast, good quality
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def build_index():
    """Process all PDFs → chunk → embed → FAISS index."""
    print("=" * 60)
    print("🏗️  VisionCare RAG Index Builder")
    print("=" * 60)

    # ── 1. Find PDFs ──────────────────────────────────────────────
    pdfs = sorted(glob.glob(str(TEXTS_DIR / "*.pdf")))
    if not pdfs:
        print(f"⚠️  No PDFs found in {TEXTS_DIR}")
        print("   Place medical reference PDFs there and re-run.")
        return 0

    print(f"\n📚 Found {len(pdfs)} PDFs:")
    for p in pdfs:
        sz = os.path.getsize(p) / 1024 / 1024
        print(f"   • {os.path.basename(p)} ({sz:.1f} MB)")

    # ── 2. Load PDFs ──────────────────────────────────────────────
    from langchain_community.document_loaders import PyPDFLoader

    all_docs = []
    for pdf_path in pdfs:
        print(f"\n📖 Loading: {os.path.basename(pdf_path)}...")
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            # Add source metadata
            for page in pages:
                page.metadata["source_book"] = os.path.basename(pdf_path)
            all_docs.extend(pages)
            print(f"   ✅ {len(pages)} pages extracted")
        except Exception as e:
            print(f"   ❌ Error loading: {e}")

    if not all_docs:
        print("❌ No documents loaded. Check PDF files.")
        return 0

    print(f"\n📄 Total pages loaded: {len(all_docs)}")

    # ── 3. Chunk documents ────────────────────────────────────────
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(all_docs)
    print(f"✂️  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Filter out very short chunks (noise)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    print(f"🧹 After filtering short chunks: {len(chunks)}")

    # ── 4. Create embeddings + FAISS index ────────────────────────
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    print(f"\n🧠 Loading embedding model: {EMBED_MODEL}...")
    t0 = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    print(f"📊 Building FAISS index from {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ── 5. Save index ─────────────────────────────────────────────
    INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    elapsed = time.time() - t0
    print(f"\n✅ FAISS index saved to {INDEX_DIR}/")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Time:   {elapsed:.1f}s")
    print(f"   Files:  index.faiss + index.pkl")
    print("=" * 60)

    return len(chunks)


if __name__ == "__main__":
    n = build_index()
    if n:
        print(f"\n🎉 RAG index ready with {n} chunks!")
    else:
        print("\n❌ Failed to build index.")
        sys.exit(1)
