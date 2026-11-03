import sys
import os

# Ensure backend directory is in python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_db_connection


def test_schema_and_constraints():
    print("=" * 60)
    print("   TASK 1: DATABASE SCHEMA & CONSTRAINT VERIFICATION   ")
    print("=" * 60)

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # 1. Inspect table columns
        print("\n[Check 1/4] Inspecting column definitions in 'users' table...")
        cur.execute("""
            SELECT column_name, data_type, udt_name, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'users'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        for col in columns:
            print(f"  - Column: {col[0]:<15} Type: {col[1]:<25} UDT: {col[2]:<10} Nullable: {col[3]}")

        col_dict = {col[0]: col for col in columns}
        assert "user_id" in col_dict, "user_id column missing"
        assert "created_at" in col_dict, "created_at column missing"
        assert "embedding" in col_dict, "embedding column missing"
        assert col_dict["embedding"][2] == "_float4", f"Expected _float4 (REAL[]), got {col_dict['embedding'][2]}"
        print("  [PASS] All expected columns are present with correct data types.")

        # 2. Inspect constraints
        print("\n[Check 2/4] Inspecting constraints on 'users' table...")
        cur.execute("""
            SELECT conname, pg_get_constraintdef(c.oid)
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'users';
        """)
        constraints = cur.fetchall()
        for con in constraints:
            print(f"  - Constraint: {con[0]:<25} Definition: {con[1]}")

        con_names = [c[0] for c in constraints]
        assert "users_pkey" in con_names, "Primary key missing on users"
        assert "check_embedding_dim" in con_names, "check_embedding_dim constraint missing"
        print("  [PASS] Primary key and check_embedding_dim constraints are verified.")

        # 3. Test inserting a valid 512-D embedding
        print("\n[Check 3/4] Testing insertion of valid 512-D embedding...")
        test_uid = "schema_test_user_valid"
        dummy_embedding_512 = [0.001 * i for i in range(512)]
        
        cur.execute("DELETE FROM users WHERE user_id = %s;", (test_uid,))
        cur.execute("""
            INSERT INTO users (user_id, embedding)
            VALUES (%s, %s)
            RETURNING user_id, array_length(embedding, 1), created_at;
        """, (test_uid, dummy_embedding_512))
        inserted = cur.fetchone()
        conn.commit()

        print(f"  - Inserted User: {inserted[0]}")
        print(f"  - Embedding Length: {inserted[1]}")
        print(f"  - Created At: {inserted[2]}")
        assert inserted[1] == 512, f"Expected 512 length, got {inserted[1]}"
        print("  [PASS] Valid 512-D embedding inserted and length confirmed.")

        # Clean up valid test record
        cur.execute("DELETE FROM users WHERE user_id = %s;", (test_uid,))
        conn.commit()

        # 4. Test constraint rejection of invalid dimension embedding (e.g. 511 elements)
        print("\n[Check 4/4] Testing constraint rejection for invalid embedding dimension (511 elements)...")
        invalid_uid = "schema_test_user_invalid"
        dummy_embedding_511 = [0.001 * i for i in range(511)]

        rejected = False
        try:
            cur.execute("""
                INSERT INTO users (user_id, embedding)
                VALUES (%s, %s);
            """, (invalid_uid, dummy_embedding_511))
            conn.commit()
        except Exception as err:
            conn.rollback()
            rejected = True
            print(f"  [PASS] Correctly rejected invalid embedding: {err.pgerror.strip() if hasattr(err, 'pgerror') else err}")

        assert rejected, "Database failed to reject an embedding with incorrect dimension (511 elements)!"
        print("  [PASS] Constraint check_embedding_dim successfully enforced at the DB level.")

    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("   TASK 1: ALL SCHEMA VERIFICATION CHECKS PASSED   ")
    print("=" * 60)


if __name__ == "__main__":
    test_schema_and_constraints()

