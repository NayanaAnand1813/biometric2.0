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

