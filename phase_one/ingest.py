import sys
import os
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent.parent))

from phase_one.chunker import chunk_document
from phase_one.embedder import Embedder
from phase_one.vector_store import VectorStore

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
VECTOR_DB_PATH = "data/lancedb"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
 
def extract_text_from_files(file_path:Path)->str | None:
    suffix=file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [!] Could not read {file_path.name}: {e}")
            return None
    elif suffix == ".pdf":
        try:
            import fitz
            doc = fitz.open(str(file_path))
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            full_text = "\n\n".join(pages)
 
            if not full_text.strip():
                print(f"  [!] {file_path.name}: extracted empty text (scanned PDF?)")
                return None
            return full_text

        except ImportError:
            print("  [!] PyMuPDF not installed. Run: pip install pymupdf")
            return None
        except Exception as e:
            print(f"  [!] Could not read PDF {file_path.name}: {e}")
            return None
 
    return None

 
def ingest_all(knowledge_base_dir: Path, db_path: str):
    documents = [
        f for f in knowledge_base_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
    ]
 
    if not documents:
        print(f"\n[!] No documents found in {knowledge_base_dir}/")
        print("    Put .txt, .md, or .pdf files there and run again.")
        return
 
    print(f"\nFound {len(documents)} document(s) in {knowledge_base_dir}/")
    for d in documents:
        print(f"  - {d.name}")
 
    
    store = VectorStore(db_path)
    store.init()
    embedder = Embedder()
    total_chunks = 0
 
    already_ingested = store.list_sources()
 
    for doc_path in documents:
        source_name = doc_path.stem #.stem gives us the filename without the extension
 
      
        if source_name in already_ingested:
            print(f"\n[Skip] '{source_name}' already in database.")
            print("       Delete it with: store.delete_source('{source_name}') to re-ingest.")
            continue
 
        print(f"\n[Processing] {doc_path.name}")

        text = extract_text_from_files(doc_path)
        if not text:
            continue
 
        print(f"  Extracted {len(text):,} characters")
 
       
        chunks = chunk_document(text, source_name)
        print(f"  Split into {len(chunks)} chunks")
 
        if not chunks:
            continue
 
        texts_to_embed = [c["text"] for c in chunks]
        vectors = embedder.embed_batch(texts_to_embed)
 
     
        for i, chunk in enumerate(chunks):
            chunk["vector"] = vectors[i]

        store.add_chunks(chunks)
        total_chunks += len(chunks)
 
    print(f"\n{'='*50}")
    print(f"Ingestion complete.")
    print(f"Total chunks in database: {store.count()}")
    print(f"Sources: {store.list_sources()}")
    print(f"Database location: {db_path}/")
 
 
def run_test_queries(db_path: str):

    test_queries = [
        "someone is unconscious and not breathing",
        "how to purify water in emergency",
        "earthquake safety what to do",
        "flood evacuation route",
        "treating a deep wound with no medical supplies",
    ]
 
    print("\n" + "="*50)
    print("TESTING: running example queries")
    print("="*50)
 
    embedder = Embedder()
    store = VectorStore(db_path)
    store.init()
 
    if store.count() == 0:
        print("[!] Database is empty. Run ingestion first.")
        return
 
    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 40)
 
        query_vector = embedder.embed(query)
 
        results = store.search(query_vector, top_k=3)
 
        for i, result in enumerate(results, 1):
            print(f"  [{i}] Source: {result['source']}  |  Score: {result['score']}")
            preview = result["text"][:200].replace("\n", " ")
            print(f"      {preview}...")
 
 
if __name__ == "__main__":
    KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
 
    if "--test" in sys.argv:
        run_test_queries(VECTOR_DB_PATH)
    else:
        ingest_all(KNOWLEDGE_BASE_DIR, VECTOR_DB_PATH)