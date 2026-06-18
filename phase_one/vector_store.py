import lancedb
import pyarrow as pa 
from pathlib import Path

TABLE_NAME="crisis_chunks"
EMBEDDING_DIM=384   # must match the embedding model's output size

class VectoreStore:
    def __init__ (self, db_path:str):
        self.db_path=db_path
        self._db=None
        self._table= None

    def init(self):
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self._db=lancedb.connect(self.db_path)

        if TABLE_NAME in self._db.table_names():
            self._tab;e=self._db.open_table(TABLE_NAME)
            count = self._table.count_rows()
            print(f"[VectorStore] Opened existing table with {count} chunks.")

        else:
            schema=pa.schema([
                pa.field("text",        pa.string()),
                pa.field("source",      pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("token_count", pa.int32()),
                pa.field("vector",      pa.list_(pa.float32(), EMBEDDING_DIM)), 
            ])
            self._table=self._db.create_table(TABLE_NAME,schema=schema)
            print(f"[VectoreStore] Created new table '{TABLE_NAME}'. ")
    
    def add_chunks(self,chunks:list[dict]):
        if not self._table:
            raise RuntimeError("Call init() before add_chunks()")
        valid = [c for c in chunks if c.get("vector") and len(c["vector"]) == EMBEDDING_DIM]
 
        if not valid:
            print("[VectorStore] Warning: no valid chunks to add.")
            return
 
        self._table.add(valid)
        print(f"[VectorStore] Added {len(valid)} chunks.")

    def search(self, query_vector:list[float], top_k:int=5)->list[dict]:
        if not self._table:
            raise RuntimeError("Call init() before search()")
 
        results = (
            self._table.search(query_vector)
            .limit(top_k)
            .to_list()
        )
        output= [] 
        for row in results:
            l2_distance=row.get("_distance", 0.0)
            similarity_score=1.0/(1.0+ l2_distance)
            output.append({
                "text":        row["text"],
                "source":      row["source"],
                "chunk_index": row["chunk_index"],
                "token_count": row["token_count"],
                "score":       round(similarity_score, 4),
            })
        return output

    def count(self)->int:
        if not self._table:
            return 0
        return self._table.count_rows()

    def delete_source(self,source_name:str):
        if not self._table:
            raise RuntimeError("Call init() before delete_source()")
        self._table.delete(f"source = '{source_name}'")
        print(f"[VectorStore] Deleted chunks from '{source_name}'.")

    def list_sources(self)->list[str]:
        if not self._table:
            return []
        rows = self._table.to_pandas()["source"].unique().tolist()
        return sorted(rows)
