import sys
import os
from datetime import datetime

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_db_connection


def run_database_tests() -> bool:
    """
    Performs end-to-end tests for Milestone 4:
    1. Verifies connection to PostgreSQL database `biometricDB`.
    2. Verifies `users` table exists.
    3. Tests inserting a sample user.
    4. Tests retrieving the sample user and verifying timestamp.
    5. Cleans up test record to maintain database cleanliness.
    """
    print("=" * 60)
    print("      BIOMETRIC DATABASE TEST SUITE (MILESTONE 4)      ")
    print("=" * 60)

    # Test 1: Connect to biometricDB
    print("\n[Step 1/4] Testing connection to PostgreSQL database...")
    try:
        conn = get_db_connection()
        dbname = conn.get_dsn_parameters().get("dbname")
        user = conn.get_dsn_parameters().get("user")
        print(f"  [PASS] Successfully connected to database: '{dbname}' as user: '{user}'")
    except Exception as e:
        print(f"  [FAIL] Failed to connect to database: {e}")
        return False

    sample_user_id = f"test_user_{int(datetime.now().timestamp())}"

    try:
        with conn.cursor() as cur:
            # Test 2: Check if table `users` exists
            print("\n[Step 2/4] Verifying 'users' table exists...")
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'users'
                );
            """)
            table_exists = cur.fetchone()[0]
            if not table_exists:
                print("  [FAIL] Table 'users' does not exist in 'biometricDB'.")
                return False
            print("  [PASS] Table 'users' exists in 'biometricDB'.")

            # Test 3: Insert a sample user
            print(f"\n[Step 3/4] Inserting sample user: '{sample_user_id}'...")
            cur.execute(
                "INSERT INTO users (user_id) VALUES (%s) RETURNING user_id, created_at;",
                (sample_user_id,)
            )
            inserted_row = cur.fetchone()
            conn.commit()
            print(f"  [PASS] Inserted: user_id='{inserted_row[0]}', created_at='{inserted_row[1]}'")

            # Test 4: Retrieve the sample user
            print(f"\n[Step 4/4] Querying and retrieving user: '{sample_user_id}'...")
            cur.execute(
                "SELECT user_id, created_at FROM users WHERE user_id = %s;",
                (sample_user_id,)
            )
            retrieved_row = cur.fetchone()
            if not retrieved_row:
                print(f"  [FAIL] Could not retrieve user '{sample_user_id}'.")
                return False

            retrieved_user_id, retrieved_created_at = retrieved_row
            assert retrieved_user_id == sample_user_id, "User ID mismatch!"
            print(f"  [PASS] Retrieved record matches: user_id='{retrieved_user_id}', created_at='{retrieved_created_at}'")

            # Cleanup test record
            cur.execute("DELETE FROM users WHERE user_id = %s;", (sample_user_id,))
            conn.commit()
            print(f"  [PASS] Cleaned up temporary test record '{sample_user_id}'.")

        print("\n" + "=" * 60)
        print(" [ALL TESTS PASSED] Milestone 4 database setup verified!")
        print("=" * 60)
        return True

    except Exception as e:
        conn.rollback()
        print(f"\n  [ERROR] An error occurred during test execution: {e}")
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    success = run_database_tests()
    sys.exit(0 if success else 1)

