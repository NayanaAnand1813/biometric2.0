# Acceptance Criteria & Verification Evidence (Milestone 4)

| Criteria | Status | Evidence |
|---|---|---|
| PostgreSQL is installed and running | PASSED | Windows Service `postgresql-x64-17` is `Running`; `psql` v17.11 available |
| `biometricDB` exists and accessible | PASSED | `SELECT current_database();` verified on `localhost:5432` |
| Driver installed & credentials managed securely | PASSED | `psycopg2-binary` installed in `backend/venv`; `.env.example` created; `backend/.env` git-ignored |
| `users` table created with correct schema | PASSED | `user_id VARCHAR(255) PRIMARY KEY`, `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP` verified in `information_schema.columns` |
| FastAPI Python environment connects | PASSED | `backend/test_db_connection.py` executed via `backend/venv/Scripts/python.exe` |
| Sample user insertion and retrieval works | PASSED | `INSERT` and `SELECT` verified in `backend/test_db_connection.py` |
| Existing milestones unchanged | PASSED | React frontend, BlazeFace detector, FaceNet embeddings scripts, and test JSON files remain intact |
| Milestone 5 Task 1: Embedding Column & Constraint | PASSED | Added `embedding REAL[]` with `check_embedding_dim` enforcing 512 dimensions; verified with `backend/test_task1_schema.py` |
| Milestone 5 Task 2: FaceNet Integration & Validation | PASSED | Reused FaceNet singleton, verified thread-safety lock, validated 512 dimensions and finite floats, user-embedding isolation; verified with `backend/test_task2_facenet.py` |
| Milestone 5 Task 3: Database Storage Function | PASSED | Implemented `register_user_with_embedding` using parameterized queries and thread-safe connection pooling, verified 512-D retrieval and clean disk footprint; verified with `backend/test_task3_storage.py` |
| Milestone 5 Task 4: Concurrency & Duplicate Protection | PASSED | Verified concurrent user registrations (5 threads), isolated data integrity, race-condition duplicate protection (409 Conflict), and async HTTP concurrency; verified with `backend/test_task4_concurrency.py` and `backend/test_api_concurrency.py` |
| Milestone 5 Task 5: End-to-End Acceptance Suite | PASSED | All 8 required test steps passed: single user register & store, 512-D retrieval, second user register & data immutability, duplicate ID rejection, invalid image handling, 5-thread concurrency isolation, BlazeFace regression, and database cleanup; verified with `backend/test_milestone5_complete.py` |

