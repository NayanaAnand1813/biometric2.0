# Architecture Specification (Milestone 4: PostgreSQL Database Setup)

## Components & Topology
- **Frontend**: React (Vite) on port 5173 (untouched).
- **Backend API**: FastAPI on Python 3.14 (`backend/venv`).
- **Detection & Embeddings**: BlazeFace (`.tflite`) & FaceNet (`facenet_env`) (untouched).
- **Database**: PostgreSQL 17 on `localhost:5432`.
- **Database Name**: `biometricDB`.
- **Driver**: `psycopg2-binary` (v2.9.13).

## Data Schema
### Table: `users`
- `user_id`: `VARCHAR(255)` (`PRIMARY KEY`) - Unique user identifier.
- `created_at`: `TIMESTAMP` (`DEFAULT CURRENT_TIMESTAMP`) - User registration timestamp.

*(Note: Biometric vector storage is explicitly deferred to later milestones; no vector embeddings or pgvector extensions are added in this milestone).*

## Trust Boundaries & Security
- Database credentials loaded strictly via `python-dotenv` from local `backend/.env`.
- `backend/.env` is ignored by `.gitignore` and never committed.
- Parameterized SQL queries (`%s`) used across all operations to prevent SQL injection.

