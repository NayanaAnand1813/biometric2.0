# Architecture Specification (Milestone 6: Login Frontend & Image Capture)

## Components & Topology
- **Frontend**: React 19 (Vite) on port 5173 with tabbed interface:
  - `Registration` component: captures and registers user biometrics via `POST /upload`.
  - `Login` component: captures user face via `getUserMedia`, converts frame to JPEG via `<canvas>`, and submits payload to `POST /login`.
- **Backend API**: FastAPI on Python 3.13 (`backend/facenet_env`):
  - Endpoints: `GET /`, `POST /upload`, `POST /register`, `GET /users/{user_id}`, `POST /login`.
- **Detection & Embeddings**: BlazeFace (`.tflite`) & FaceNet singleton with thread locks.
- **Database**: PostgreSQL 17 on `localhost:5432`.
- **Database Name**: `biometricDB`.
- **Driver**: `psycopg2-binary` (v2.9.13) with `ThreadedConnectionPool`.

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
- Login capture endpoint validates image bytes and format before server acknowledgment. Authentication decisions and vector comparison strictly isolated to Milestone 7.

