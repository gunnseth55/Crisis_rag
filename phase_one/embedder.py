#this same document will be used for the embedding of the documents and the embedding of queries at search time

from sentence_transformers import SentenceTransformer
from tqm from tqdm

MODEL_NAME= "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    """
     Wraps the sentence-transformers model.
    Loads the model once on first use (lazy loading).
 
    After the first run, the model is cached at:
        ~/.cache/huggingface/hub/
    Subsequent runs load from cache — no internet needed.   
    """
    def __init__(self):
        self.model=None

    def _load(self):
        if self.model is None:
            print(f"[Embedder] Loading {MODEL_NAME}...")
            print("  (this downloads ~90MB on first run, then uses cache)")
            self._model = SentenceTransformer(MODEL_NAME)
            print("[Embedder] Ready.")
    
    def embed(self, text:str)->list[float]:
        self._load()
        vector=self._model.encode(text,normalize_embeddings=True, show_progress_bar=False,)
        return vector.tolist()

    def embed_batcj(self,texts:list[str],batch_size:int=64)->list[list[float]]:
        self.load()
        all_vectors=[]
        for i in tqdm(range(0,len(texts),batch_size),desc="Embedding chunks"):
            batch= texts[i:i+batch_size]
            vectors=self._model.encode(
                batch, normalize_embeddings=True, show_progress_bar=False,
            )
            all_vectors.extend(vectors.tolist())
        return all_vectors


    