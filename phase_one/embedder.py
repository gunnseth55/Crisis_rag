#this same document will be used for the embedding of the documents and the embedding of queries at search time

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            print(f"[Embedder] Loading {MODEL_NAME}...")
            print("  (uses local cache; downloads ~90MB only if missing)")
            try:
                self._model = SentenceTransformer(MODEL_NAME, local_files_only=True)
            except OSError:
                # Setup may download missing weights. Once cached, queries
                # must not make Hub metadata requests or wait for the network.
                self._model = SentenceTransformer(MODEL_NAME)
            print("[Embedder] Ready.")

    def embed(self, text: str) -> list[float]:
        self._load()
        vector = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        self._load()
        all_vectors = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
            batch = texts[i:i+batch_size]
            vectors = self._model.encode(
                batch, normalize_embeddings=True, show_progress_bar=False,
            )
            all_vectors.extend(vectors.tolist())
        return all_vectors




