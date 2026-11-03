# Decision Log (Milestone 4)

## Decisions & Rationales

1. **PostgreSQL Version & Distribution**
   - **Decision**: PostgreSQL 17.11 with pgAdmin 4 and CLI tools on Windows.
   - **Rationale**: Supported release with official EnterpriseDB installer for Windows x86-64, providing stable service execution and tooling.

2. **Python PostgreSQL Driver**
   - **Decision**: `psycopg2-binary` (v2.9.13).
   - **Rationale**: Provides native prebuilt wheels for Python 3.14 on Windows, eliminating C-compiler build requirements while ensuring maximum performance and stability.

3. **Secrets Management**
   - **Decision**: Environment variables via `python-dotenv` with a committed `.env.example` template and git-ignored `backend/.env`.
   - **Rationale**: Enforces security-by-design; no passwords or connection strings are hardcoded in source control.

4. **Schema Scope**
   - **Decision**: Only `user_id` and `created_at` in the `users` table.
   - **Rationale**: Strictly preserves separation of concerns and obeys prompt constraints against premature vector storage integration.

5. **Milestone 5 Task 1: Vector Storage Format (`REAL[]`)**
   - **Decision**: PostgreSQL native 1-D array of single-precision floating point numbers (`REAL[]`) with `CHECK (embedding IS NULL OR (array_ndims(embedding) = 1 AND array_length(embedding, 1) = 512))`.
   - **Rationale**: Avoids Windows compilation dependencies of `pgvector` on PostgreSQL 17; maps natively to Python float lists with `psycopg2`; guarantees strict 512-dimension vector integrity at the DB constraint layer; non-destructive schema migration.

6. **Milestone 5 Task 4: Concurrency Model & Connection Pooling**
   - **Decision**: `psycopg2.pool.ThreadedConnectionPool` (1-20 connections), `threading.Lock` guards around MediaPipe and FaceNet inference, and `run_in_threadpool` for non-blocking FastAPI async execution.
   - **Rationale**: Eliminates race conditions in underlying C++ / TensorFlow runtimes while keeping the FastAPI async event loop fully responsive. Ensures atomic transactions with automatic rollback on error. Duplicate user IDs are caught at the database unique index layer and surfaced as HTTP 409 Conflict without silent overwrites.

7. **Milestone 6: Login Frontend Capture Interface & Boundary Isolation**
   - **Decision**: Implemented modular `Login.jsx` and preserved existing `Registration.jsx` with tabbed navigation in `App.jsx`. Added `POST /login` on FastAPI to validate User ID and JPEG image receipt.
   - **Rationale**: Cleanly decouples registration and login workflows without introducing new npm dependencies (e.g., react-router). Strictly separates camera capture in Milestone 6 from biometric verification, similarity search, and authentication decision-making (deferred to Milestone 7).

8. **Milestone 7 Wave 1: Shared FaceNet Preprocessing Pipeline (`extract_face_embedding`)**
   - **Decision**: Refactored FaceNet embedding extraction into a shared function `extract_face_embedding(face_bgr: np.ndarray) -> np.ndarray` accepting in-memory BGR images, while preserving `generate_embedding(user_id, image_path, save_json)` as a backward-compatible wrapper.
   - **Rationale**: Guarantees bit-for-bit identical preprocessing (160x160 resize, BGR-to-RGB, FaceNet inference under thread lock, 512-D finite verification, explicit unit L2 normalization) across registration and authentication flows, eliminating code drift and disk I/O bottlenecks.

9. **Milestone 7 Wave 1: FAISS IndexFlatIP Service Architecture & Safe Synchronization**
   - **Decision**: Implemented `faiss.IndexFlatIP(512)` CPU index wrapped in `FAISSIndexManager` singleton with `threading.RLock`, bidirectional `dict` mappings between sequential indices and string `user_id`s, startup loading from PostgreSQL, and commit-first registration synchronization.
   - **Rationale**: FAISS IndexFlatIP provides exact mathematical inner-product calculation (which equals cosine similarity for L2-normalized vectors) without lossy quantization. PostgreSQL remains the sole transactional source of truth; failed DB registrations never touch the FAISS index, preventing orphaned state. Internal similarity scores are strictly held inside backend services to protect against biometric inversion/hill-climbing attacks.

10. **Milestone 7 Wave 2: Dual-Gate Authentication & In-Memory Login Execution**
   - **Decision**: `POST /login` executes face detection, face cropping, FaceNet embedding, and FAISS exact inner-product search purely in memory. Authentication requires dual validation: `top_matching_user_id == submitted_user_id` AND `cosine_similarity >= BIOMETRIC_COSINE_THRESHOLD`.
   - **Rationale**: Eliminates trust in client-claimed IDs; denies unknown users, identity mismatches, and sub-threshold probes; avoids disk clutter and biometric privacy leakage by not saving login frames or vectors to disk or DB; offloads CPU-bound inference to worker threadpool.

11. **Milestone 7 Wave 2: Frontend Welcome Screen & Security Boundary Preservation**
   - **Decision**: Implemented an "Access Granted" Welcome view in `Login.jsx` showing the verified user ID and session details, while suppressing similarity scores and embedding data. Failed authentications surface generic access-denied warnings.
   - **Rationale**: Shields the user interface from mathematical biometric internals (mitigating hill-climbing attacks) while giving immediate, accessible visual confirmation to genuine users and clear guidance on failure.

12. **Milestone 7 Wave 2 Correction: Deterministic Threshold Calibration & Gate Independence**
   - **Decision**: Verified authentication gates independently using Gram-Schmidt orthogonalized vector synthesis ($v = \alpha u + \sqrt{1-\alpha^2}u_\perp$ for exact cosine similarity $\alpha = 0.60$ against probe vector $u$) with candidate isolation.
   - **Rationale**: Replaces fragile non-deterministic tests with mathematically exact vector pairs. Proves that even when the candidate user ID matches the claimed ID (`top_user_id == clean_user_id`), Gate 2 strictly enforces rejection (HTTP 401) when similarity is below threshold ($0.60 < 0.75$), and accepts (HTTP 200) only when similarity meets or exceeds threshold ($0.60 \ge 0.50$). Confirmed that no similarity scores or embeddings are serialized into API response payloads.

