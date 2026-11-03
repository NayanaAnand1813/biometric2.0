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

