"""
FAISS-based vector store with simple SQLite metadata.
"""
import os, json, sqlite3, threading, logging
from typing import List, Dict, Optional, Any
import numpy as np
import faiss

logger = logging.getLogger(__name__)
DEFAULT_INDEX_PATH = os.environ.get("VECTOR_INDEX_PATH", "app/model_data/faiss.index")
DEFAULT_META_DB = os.environ.get("VECTOR_META_DB", "app/model_data/faiss_meta.db")
os.makedirs(os.path.dirname(DEFAULT_INDEX_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DEFAULT_META_DB), exist_ok=True)

class FaissVectorStore:
    def __init__(self, index_path: str = DEFAULT_INDEX_PATH, meta_db_path: str = DEFAULT_META_DB, dim: int = 768):
        self.index_path = index_path
        self.meta_db_path = meta_db_path
        self.dim = dim
        self._lock = threading.Lock()
        self._index = None
        self._id_map = []
        self._connect_meta_db()
        self._load_index_if_exists()

    def _connect_meta_db(self):
        self._conn = sqlite3.connect(self.meta_db_path, check_same_thread=False)
        cur = self._conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, version TEXT, metadata TEXT)""")
        self._conn.commit()

    def _load_index_if_exists(self):
        if os.path.exists(self.index_path):
            try:
                self._index = faiss.read_index(self.index_path)
                cur = self._conn.cursor()
                rows = cur.execute("SELECT id FROM vectors ORDER BY rowid ASC").fetchall()
                self._id_map = [r[0] for r in rows]
                logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
                return
            except Exception:
                logger.exception("Failed to load existing FAISS index; creating new one")
        self._index = faiss.IndexFlatIP(self.dim)

    def add_vector(self, uid: str, embedding: np.ndarray, version: str, metadata: Dict[str, Any]):
        with self._lock:
            vec = embedding.reshape(1, -1).astype('float32') if embedding.ndim == 1 else embedding.astype('float32')
            if vec.shape[1] != self.dim:
                raise ValueError(f"Embedding dim {vec.shape[1]} != store dim {self.dim}")
            self._index.add(vec)
            self._id_map.append(uid)
            cur = self._conn.cursor()
            cur.execute("INSERT OR REPLACE INTO vectors (id, version, metadata) VALUES (?, ?, ?)", (uid, version, json.dumps(metadata)))
            self._conn.commit()

    def search(self, query_vec: np.ndarray, top_k: int = 10, version: Optional[str] = None) -> List[Dict[str, Any]]:
        q = query_vec.reshape(1, -1).astype('float32') if query_vec.ndim == 1 else query_vec.astype('float32')
        D, I = self._index.search(q, top_k)
        results = []
        cur = self._conn.cursor()
        for dist_list, idx_list in zip(D, I):
            for dist, idx in zip(dist_list, idx_list):
                if idx < 0 or idx >= len(self._id_map):
                    continue
                uid = self._id_map[idx]
                row = cur.execute("SELECT version, metadata FROM vectors WHERE id=?", (uid,)).fetchone()
                if not row:
                    continue
                if version and row[0] != version:
                    continue
                meta = json.loads(row[1]) if row[1] else {}
                results.append({"id": uid, "score": float(dist), "metadata": meta})
        return results

    def get_by_id(self, uid: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        row = cur.execute("SELECT version, metadata FROM vectors WHERE id=?", (uid,)).fetchone()
        if not row:
            return None
        return {"id": uid, "version": row[0], "metadata": json.loads(row[1]) if row[1] else {}}

    def persist(self):
        with self._lock:
            faiss.write_index(self._index, self.index_path)
            self._conn.commit()
            logger.info("Persisted FAISS index and metadata DB")

