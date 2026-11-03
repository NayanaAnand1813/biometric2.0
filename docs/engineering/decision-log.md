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

