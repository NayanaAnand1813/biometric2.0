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
- `embedding`: `REAL[]` (`CHECK (embedding IS NULL OR (array_ndims(embedding) = 1 AND array_length(embedding, 1) = 512))`) - 512-dimensional FaceNet biometric float vector.

*(Note: Biometric vector storage is implemented natively using PostgreSQL `REAL[]` with dimension constraints; external vector extensions such as pgvector are avoided for stability and environment compatibility).*

## Trust Boundaries & Security
- Database credentials loaded strictly via `python-dotenv` from local `backend/.env`.
- `backend/.env` is ignored by `.gitignore` and never committed.
- Parameterized SQL queries (`%s`) used across all operations to prevent SQL injection.

