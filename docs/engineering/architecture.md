# Architecture Specification (Milestone 7 Wave 2: Complete Biometric Login Flow)

## Components & Topology
- **Frontend**: React 19 (Vite) on port 5173 with tabbed interface:
  - `Registration` component: captures and registers user biometrics via `POST /upload`.
  - `Login` component: captures user face via `getUserMedia`, converts frame to JPEG via `<canvas>`, submits payload to `POST /login`, and displays dedicated "Access Granted" Welcome screen on authentication success or clear access-denied error states on failure.
- **Backend API**: FastAPI on Python 3.13 (`backend/facenet_env`):
  - Endpoints: `GET /`, `POST /upload`, `POST /register`, `GET /users/{user_id}`, `POST /login`.
- **Detection & Embeddings**: BlazeFace (`.tflite`) & FaceNet singleton with thread locks:
  - `extract_face_embedding(face_bgr)`: Shared preprocessing and inference pipeline accepting in-memory BGR face image, resizing to 160x160, converting to RGB, evaluating FaceNet under `_embedder_lock`, verifying 512 dimensions and finite floats, and applying explicit L2 unit normalization.
- **Biometric Login Flow (`POST /login`)**:
  - Image decoded in memory via `cv2.imdecode` (never written to disk or DB).
  - BlazeFace detects and crops face in memory under `_detector_lock`.
  - Shared `extract_face_embedding` extracts 512-D L2-normalized vector in worker threadpool.
  - Queries `faiss_service` exact inner-product index under `_lock`.
  - Dual-gate verification: Top matching FAISS user ID must exactly match submitted `user_id`, AND cosine similarity must meet or exceed `BIOMETRIC_COSINE_THRESHOLD`.
  - Authentication success returns HTTP 200 with user ID; failure returns HTTP 401 Unauthorized; missing/empty face returns HTTP 400 Bad Request.
  - Similarity scores and internal embeddings are completely suppressed from client responses.
- **Fast Vector Search (FAISS)**:
  - `faiss_service.py`: In-memory `faiss.IndexFlatIP(512)` computing exact inner-product vector search across L2-normalized vectors (mathematically identical to cosine similarity).
  - Protected with `threading.RLock`.
  - Maintains bidirectional mapping between FAISS sequential integer IDs and application string `user_id`s.
- **Database**: PostgreSQL 17 on `localhost:5432`:
  - **Database Name**: `biometricDB`.
  - **Driver**: `psycopg2-binary` (v2.9.13) with `ThreadedConnectionPool`.
  - **System of Record**: PostgreSQL is the permanent, authoritative source of truth. FAISS is strictly a search acceleration index initialized from PostgreSQL on startup and updated post-commit.

## Data Schema
### Table: `users`
- `user_id`: `VARCHAR(255)` (`PRIMARY KEY`) - Unique user identifier.
- `created_at`: `TIMESTAMP` (`DEFAULT CURRENT_TIMESTAMP`) - User registration timestamp.
- `embedding`: `REAL[]` (`CHECK (embedding IS NULL OR (array_ndims(embedding) = 1 AND array_length(embedding, 1) = 512))`) - 512-dimensional FaceNet biometric float vector.

*(Note: Biometric vector storage is implemented natively using PostgreSQL `REAL[]` with dimension constraints; external vector extensions such as pgvector are avoided for stability and environment compatibility).*

## Safe FAISS Synchronization Protocol
1. **Startup Warm-up**: On FastAPI startup, FAISS is initialized by querying enrolled embeddings from PostgreSQL (`SELECT user_id, embedding FROM users WHERE embedding IS NOT NULL`).
2. **Commit-First Sync**: New embeddings are added to FAISS only after `connection.commit()` has successfully completed on PostgreSQL.
3. **Rollback & Conflict Safety**: If registration fails (e.g. duplicate user ID 409 Conflict), the database rolls back and FAISS is never modified, guaranteeing zero orphaned vectors or inconsistent states.
4. **Deletion Sync**: When records are deleted from PostgreSQL, FAISS updates internal vectors and adjusts sequential mapping.

## Trust Boundaries, Security & Limitations
- Database credentials loaded strictly via `python-dotenv` from local `backend/.env`.
- `backend/.env` is ignored by `.gitignore` and never committed.
- Parameterized SQL queries (`%s`) used across all operations to prevent SQL injection.
- Zero disk or DB footprint during login: live probe frames and probe embeddings exist solely in process memory.
- Internal similarity scores are strictly withheld from client-facing responses to eliminate biometric inversion and hill-climbing attack vectors.
- Exact vector indexing (FAISS `IndexFlatIP`) guarantees exact nearest-neighbor search, but biological recognition accuracy is subject to empirical calibration.
- **Known Limitations**:
  - No active/passive liveness detection or anti-spoofing mechanism is implemented in this wave (deferred beyond Milestone 7).
  - The default threshold `0.70` is an empirical test value and requires formal FAR/FRR calibration with genuine and impostor sample distributions before production release.

