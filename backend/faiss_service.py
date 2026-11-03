import os
import sys
import threading
import numpy as np
import faiss

# Ensure backend directory is in python search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database import get_pooled_connection


class FAISSIndexManager:
    """
    Manages an in-memory FAISS IndexFlatIP(512) for exact cosine similarity search
    over L2-normalized FaceNet biometric embeddings.

    Key Architectural Guarantees:
    1. Exact Search: Uses IndexFlatIP (exhaustive inner-product search with zero quantization loss).
       Because FaceNet vectors are L2-normalized, inner-product mathematically equals cosine similarity.
    2. Single Source of Truth: PostgreSQL is the permanent store of record.
       FAISS is initialized from PostgreSQL on startup and updated post-commit.
    3. Thread-Safe: All index modifications and queries are guarded by a threading.RLock.
    4. Two-Way Mapping: Maintains bidirectional mapping between FAISS sequential integer IDs
       and application string `user_id`s.
    5. No Score Leakage: While internal methods return mathematical similarity scores for backend
       decision-making, these scores must never be serialized to client-facing HTTP responses.
    """

    DIMENSION = 512

    def __init__(self):
        self._lock = threading.RLock()
        self._dimension = self.DIMENSION
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index_to_user_id: dict[int, str] = {}
        self._user_id_to_index: dict[str, int] = {}
        self._initialized = False

    @property
    def dimension(self) -> int:
        return self._dimension

    def is_initialized(self) -> bool:
        with self._lock:
            return self._initialized

    def _ensure_initialized(self) -> None:
        """
        Internal guard: ensures FAISS index is initialized from PostgreSQL
        prior to any search or query. Must be called while holding self._lock.
        """
        if not self._initialized:
            self.initialize_from_database()

    def get_user_count(self) -> int:
        with self._lock:
            self._ensure_initialized()
            return self._index.ntotal

    def contains_user(self, user_id: str) -> bool:
        if not user_id:
            return False
        with self._lock:
            self._ensure_initialized()
            return str(user_id).strip() in self._user_id_to_index

    def get_index_for_user(self, user_id: str) -> int | None:
        if not user_id:
            return None
        with self._lock:
            self._ensure_initialized()
            return self._user_id_to_index.get(str(user_id).strip())

    def get_user_for_index(self, index: int) -> str | None:
        with self._lock:
            self._ensure_initialized()
            return self._index_to_user_id.get(index)

    def _normalize_vector(self, vector: np.ndarray | list[float]) -> np.ndarray:
        """
        Validates 512 dimensions, finite values, and applies L2 normalization.
        Returns a float32 array of shape (512,).
        """
        arr = np.asarray(vector, dtype=np.float32).flatten()
        if arr.shape != (self._dimension,):
            raise ValueError(
                f"Embedding dimension mismatch: Expected {self._dimension}, got {arr.shape[0]}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Embedding contains invalid non-finite (NaN or Inf) values.")

        norm = float(np.linalg.norm(arr))
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError(f"Invalid zero or non-finite vector norm: {norm}.")

        return arr / norm

    def initialize_from_database(self) -> int:
        """
        Reconstructs the FAISS index by querying all enrolled users and their 512-D
        embeddings from PostgreSQL.

        Returns:
            int: Number of vectors loaded into the index.
        """
        select_sql = """
        SELECT user_id, embedding
        FROM users
        WHERE embedding IS NOT NULL
        ORDER BY created_at ASC;
        """

        records: list[tuple[str, list[float]]] = []
        with get_pooled_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(select_sql)
                rows = cur.fetchall()
                for uid, emb in rows:
                    if uid and emb:
                        records.append((str(uid).strip(), emb))

        with self._lock:
            # Recreate clean FAISS index and clear mappings
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index_to_user_id.clear()
            self._user_id_to_index.clear()

            if records:
                normalized_list: list[np.ndarray] = []
                for idx, (uid, emb) in enumerate(records):
                    norm_vec = self._normalize_vector(emb)
                    normalized_list.append(norm_vec)
                    self._index_to_user_id[idx] = uid
                    self._user_id_to_index[uid] = idx

                batch_matrix = np.vstack(normalized_list).astype(np.float32)
                self._index.add(batch_matrix)

            self._initialized = True
            count = self._index.ntotal
            print(f"[FAISS INIT] Successfully loaded {count} embeddings from PostgreSQL into FAISS.")
            return count

    def add_user(self, user_id: str, embedding: np.ndarray | list[float]) -> int:
        """
        Adds a single user's 512-D L2-normalized embedding to the FAISS index.
        Must be called ONLY AFTER successful database commit to preserve PostgreSQL
        as the system of record.

        Parameters:
            user_id (str): Unique user identifier.
            embedding: 512-dimensional vector.

        Returns:
            int: The assigned sequential FAISS integer index.

        Raises:
            ValueError: If user_id is empty, already exists in index, or vector is invalid.
        """
        if not user_id or not str(user_id).strip():
            raise ValueError("User ID cannot be empty.")
        clean_user_id = str(user_id).strip()

        norm_vec = self._normalize_vector(embedding).reshape(1, self._dimension)

        with self._lock:
            if clean_user_id in self._user_id_to_index:
                raise ValueError(
                    f"User ID '{clean_user_id}' is already indexed in FAISS."
                )

            current_count = self._index.ntotal
            self._index.add(norm_vec)
            new_idx = self._index.ntotal - 1

            self._index_to_user_id[new_idx] = clean_user_id
            self._user_id_to_index[clean_user_id] = new_idx
            self._initialized = True

            return new_idx

    def remove_user(self, user_id: str) -> bool:
        """
        Removes a user vector from the FAISS index and updates the internal ID mappings.
        Used primarily for test cleanup.

        Parameters:
            user_id (str): Unique user identifier.

        Returns:
            bool: True if user was found and removed, False otherwise.
        """
        if not user_id:
            return False
        clean_user_id = str(user_id).strip()

        with self._lock:
            if clean_user_id not in self._user_id_to_index:
                return False

            idx = self._user_id_to_index[clean_user_id]

            # IndexFlatIP supports remove_ids with an int64 numpy array
            self._index.remove_ids(np.array([idx], dtype=np.int64))

            # FAISS IndexFlatIP shifts all vectors after the removed index down by 1
            new_index_to_user_id: dict[int, str] = {}
            new_user_id_to_index: dict[str, int] = {}

            for old_idx, uid in self._index_to_user_id.items():
                if uid == clean_user_id:
                    continue
                adjusted_idx = old_idx if old_idx < idx else old_idx - 1
                new_index_to_user_id[adjusted_idx] = uid
                new_user_id_to_index[uid] = adjusted_idx

            self._index_to_user_id = new_index_to_user_id
            self._user_id_to_index = new_user_id_to_index
            return True

    def search_top_k(
        self,
        query_vector: np.ndarray | list[float],
        k: int = 1
    ) -> list[tuple[str, float]]:
        """
        Performs exact 1:N inner-product (cosine similarity) search across all indexed users.
        NOTE: This is strictly an internal backend method. Returned similarity scores
        must not be sent directly to the client/frontend.

        Parameters:
            query_vector: 512-dimensional query vector.
            k (int): Number of top matches to retrieve.

        Returns:
            list[tuple[str, float]]: Ordered list of (user_id, cosine_similarity).
        """
        norm_query = self._normalize_vector(query_vector).reshape(1, self._dimension)

        with self._lock:
            self._ensure_initialized()
            total = self._index.ntotal
            if total == 0:
                return []

            k_search = min(max(1, k), total)
            distances, indices = self._index.search(norm_query, k_search)

            results: list[tuple[str, float]] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx in self._index_to_user_id:
                    uid = self._index_to_user_id[idx]
                    results.append((uid, float(dist)))

            return results

    def compute_user_similarity(
        self,
        user_id: str,
        query_vector: np.ndarray | list[float]
    ) -> float | None:
        """
        Performs exact 1:1 cosine similarity verification for a specific claimed user_id.
        Reconstructs the enrolled vector from FAISS and computes the inner product.
        NOTE: This is strictly an internal backend method.

        Parameters:
            user_id (str): The claimed user identifier.
            query_vector: 512-dimensional probe vector.

        Returns:
            float | None: Cosine similarity in range [-1.0, 1.0], or None if user not enrolled.
        """
        if not user_id or not str(user_id).strip():
            return None
        clean_user_id = str(user_id).strip()

        norm_query = self._normalize_vector(query_vector)

        with self._lock:
            self._ensure_initialized()
            if clean_user_id not in self._user_id_to_index:
                return None

            idx = self._user_id_to_index[clean_user_id]
            stored_vector = self._index.reconstruct(idx)
            similarity = float(np.dot(norm_query, stored_vector))
            return similarity

    def get_user_vector(self, user_id: str) -> np.ndarray | None:
        """
        Retrieves the reconstructed L2-normalized 512-D float32 vector for a given user.
        """
        if not user_id:
            return None
        clean_user_id = str(user_id).strip()
        with self._lock:
            self._ensure_initialized()
            if clean_user_id not in self._user_id_to_index:
                return None
            idx = self._user_id_to_index[clean_user_id]
            return self._index.reconstruct(idx)

    def reset(self) -> None:
        """
        Resets the in-memory index and mappings to empty state.
        """
        with self._lock:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index_to_user_id.clear()
            self._user_id_to_index.clear()
            self._initialized = False


# ==============================================================================
# Global Singleton Instance Accessor
# ==============================================================================

_faiss_manager_instance: FAISSIndexManager | None = None
_instance_lock = threading.Lock()


def get_faiss_manager() -> FAISSIndexManager:
    """
    Returns the thread-safe global singleton instance of FAISSIndexManager.
    """
    global _faiss_manager_instance
    if _faiss_manager_instance is None:
        with _instance_lock:
            if _faiss_manager_instance is None:
                _faiss_manager_instance = FAISSIndexManager()
    return _faiss_manager_instance

