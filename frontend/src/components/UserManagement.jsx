import { useState, useEffect } from "react";

function UserManagement() {
  const [users, setUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [adminKey, setAdminKey] = useState("admin_secret_key_2026");
  const [feedback, setFeedback] = useState({ type: null, message: "" });
  const [confirmDialog, setConfirmDialog] = useState({ open: false, user: null });

  // Fetch all registered users from FastAPI backend
  const fetchUsers = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/users");
      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }
      const data = await response.json();
      setUsers(data.users || []);
    } catch (err) {
      setFeedback({
        type: "error",
        message: `Failed to load users: ${err.message || "Connection refused"}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  // Open confirmation dialog
  const promptDeleteUser = (user) => {
    setFeedback({ type: null, message: "" });
    setConfirmDialog({ open: true, user });
  };

  // Close confirmation dialog
  const cancelDelete = () => {
    if (!isDeleting) {
      setConfirmDialog({ open: false, user: null });
    }
  };

  // Execute DELETE API request
  const executeDelete = async () => {
    const user = confirmDialog.user;
    if (!user) return;

    setIsDeleting(true);
    setFeedback({ type: null, message: "" });

    try {
      const response = await fetch(`http://127.0.0.1:8000/users/${encodeURIComponent(user.user_id)}`, {
        method: "DELETE",
        headers: {
          "X-Admin-Key": adminKey.trim()
        }
      });

      const data = await response.json().catch(() => ({}));

      if (response.ok) {
        setFeedback({
          type: "success",
          message: data.message || `User '${user.user_id}' was deleted successfully.`
        });
        setConfirmDialog({ open: false, user: null });
        // Automatically refresh user list
        await fetchUsers();
      } else {
        const errorDetail = data.detail?.message || data.detail?.error || data.message || `HTTP ${response.status} Error`;
        setFeedback({
          type: "error",
          message: `Deletion failed: ${errorDetail}`
        });
        setConfirmDialog({ open: false, user: null });
      }
    } catch (err) {
      setFeedback({
        type: "error",
        message: `Network error deleting user: ${err.message || "Could not reach server"}`
      });
      setConfirmDialog({ open: false, user: null });
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div style={{ padding: "10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "18px", flexWrap: "wrap", gap: "10px" }}>
        <div>
          <h2 style={{ margin: "0 0 4px 0", color: "#111827" }}>Registered Users</h2>
          <p style={{ margin: 0, color: "#6b7280", fontSize: "14px" }}>
            Manage biometric profiles, PostgreSQL records, and FAISS vector indices
          </p>
        </div>
        <button
          onClick={fetchUsers}
          disabled={isLoading}
          style={{
            padding: "8px 16px",
            background: "#f3f4f6",
            border: "1px solid #d1d5db",
            borderRadius: "6px",
            cursor: isLoading ? "not-allowed" : "pointer",
            fontWeight: "600",
            fontSize: "14px",
            color: "#374151",
            display: "inline-flex",
            alignItems: "center",
            gap: "6px"
          }}
        >
          🔄 {isLoading ? "Refreshing..." : "Refresh List"}
        </button>
      </div>

      {/* Admin Key Configuration Bar */}
      <div
        style={{
          background: "#f9fafb",
          border: "1px solid #e5e7eb",
          borderRadius: "8px",
          padding: "12px 16px",
          marginBottom: "18px",
          display: "flex",
          alignItems: "center",
          gap: "12px",
          flexWrap: "wrap"
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "#374151", fontSize: "14px", fontWeight: "600" }}>
          <span>🔑 Admin Key:</span>
        </div>
        <input
          type="password"
          value={adminKey}
          onChange={(e) => setAdminKey(e.target.value)}
          placeholder="Enter Admin API Key"
          style={{
            flex: "1 1 220px",
            padding: "6px 12px",
            fontSize: "14px",
            borderRadius: "6px",
            border: "1px solid #d1d5db",
            fontFamily: "monospace"
          }}
        />
        <span style={{ fontSize: "12px", color: "#6b7280" }}>
          (Required to authorize deletions)
        </span>
      </div>

      {/* Feedback Messages (Success / Error) */}
      {feedback.message && (
        <div
          style={{
            padding: "12px 16px",
            borderRadius: "8px",
            marginBottom: "18px",
            fontSize: "14px",
            fontWeight: "500",
            background: feedback.type === "success" ? "#ecfdf5" : "#fef2f2",
            border: `1px solid ${feedback.type === "success" ? "#a7f3d0" : "#fecaca"}`,
            color: feedback.type === "success" ? "#065f46" : "#991b1b",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between"
          }}
        >
          <div>
            <span>{feedback.type === "success" ? "✅ " : "⚠️ "}</span>
            {feedback.message}
          </div>
          <button
            onClick={() => setFeedback({ type: null, message: "" })}
            style={{
              background: "transparent",
              border: "none",
              cursor: "pointer",
              fontSize: "16px",
              color: "inherit"
            }}
          >
            ✕
          </button>
        </div>
      )}

      {/* Users Table / List */}
      {isLoading && users.length === 0 ? (
        <div style={{ textAlign: "center", padding: "40px", color: "#6b7280" }}>
          Loading registered users...
        </div>
      ) : users.length === 0 ? (
        <div
          style={{
            textAlign: "center",
            padding: "40px 20px",
            border: "2px dashed #e5e7eb",
            borderRadius: "8px",
            color: "#6b7280"
          }}
        >
          <div style={{ fontSize: "36px", marginBottom: "8px" }}>👤</div>
          <p style={{ margin: 0, fontWeight: "500" }}>No registered users found.</p>
          <p style={{ margin: "4px 0 0 0", fontSize: "13px" }}>
            Switch to the Registration tab to enroll a face profile.
          </p>
        </div>
      ) : (
        <div style={{ overflowX: "auto", border: "1px solid #e5e7eb", borderRadius: "8px" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "14px" }}>
            <thead>
              <tr style={{ background: "#f9fafb", borderBottom: "1px solid #e5e7eb" }}>
                <th style={{ padding: "12px 16px", fontWeight: "600", color: "#374151" }}>User ID</th>
                <th style={{ padding: "12px 16px", fontWeight: "600", color: "#374151" }}>Registration Date</th>
                <th style={{ padding: "12px 16px", fontWeight: "600", color: "#374151" }}>Biometric Status</th>
                <th style={{ padding: "12px 16px", fontWeight: "600", color: "#374151", textAlign: "right" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user, idx) => (
                <tr
                  key={user.user_id}
                  style={{
                    borderBottom: idx === users.length - 1 ? "none" : "1px solid #f3f4f6",
                    transition: "background-color 0.15s"
                  }}
                >
                  <td style={{ padding: "12px 16px" }}>
                    <code
                      style={{
                        background: "#f3f4f6",
                        padding: "3px 8px",
                        borderRadius: "4px",
                        fontWeight: "600",
                        color: "#1f2937"
                      }}
                    >
                      {user.user_id}
                    </code>
                  </td>
                  <td style={{ padding: "12px 16px", color: "#4b5563" }}>
                    {user.created_at ? new Date(user.created_at).toLocaleString() : "—"}
                  </td>
                  <td style={{ padding: "12px 16px" }}>
                    {user.has_embedding ? (
                      <span
                        style={{
                          display: "inline-block",
                          padding: "2px 8px",
                          borderRadius: "9999px",
                          background: "#dcfce7",
                          color: "#166534",
                          fontSize: "12px",
                          fontWeight: "600"
                        }}
                      >
                        ● 512-D Synced
                      </span>
                    ) : (
                      <span
                        style={{
                          display: "inline-block",
                          padding: "2px 8px",
                          borderRadius: "9999px",
                          background: "#fee2e2",
                          color: "#991b1b",
                          fontSize: "12px",
                          fontWeight: "600"
                        }}
                      >
                        Missing Vector
                      </span>
                    )}
                  </td>
                  <td style={{ padding: "12px 16px", textAlign: "right" }}>
                    <button
                      onClick={() => promptDeleteUser(user)}
                      style={{
                        padding: "6px 14px",
                        fontSize: "13px",
                        fontWeight: "600",
                        cursor: "pointer",
                        borderRadius: "6px",
                        background: "#fee2e2",
                        color: "#dc2626",
                        border: "1px solid #fecaca",
                        transition: "all 0.15s"
                      }}
                      onMouseOver={(e) => {
                        e.currentTarget.style.background = "#dc2626";
                        e.currentTarget.style.color = "#ffffff";
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.style.background = "#fee2e2";
                        e.currentTarget.style.color = "#dc2626";
                      }}
                    >
                      🗑️ Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Confirmation Dialog / Modal */}
      {confirmDialog.open && confirmDialog.user && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(0, 0, 0, 0.5)",
            backdropFilter: "blur(2px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000
          }}
        >
          <div
            style={{
              background: "#ffffff",
              borderRadius: "12px",
              padding: "24px",
              maxWidth: "460px",
              width: "90%",
              boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"
            }}
          >
            <div style={{ textAlign: "center", marginBottom: "16px" }}>
              <div style={{ fontSize: "42px", marginBottom: "8px" }}>⚠️</div>
              <h3 style={{ margin: "0 0 8px 0", color: "#111827", fontSize: "18px" }}>
                Confirm User Deletion
              </h3>
              <p style={{ margin: 0, color: "#4b5563", fontSize: "14px", lineHeight: "1.5" }}>
                Are you sure you want to permanently delete user{" "}
                <strong style={{ color: "#dc2626", wordBreak: "break-all" }}>
                  {confirmDialog.user.user_id}
                </strong>
                ?
              </p>
            </div>

            <div
              style={{
                background: "#fef2f2",
                border: "1px solid #fee2e2",
                borderRadius: "8px",
                padding: "12px",
                marginBottom: "20px",
                fontSize: "13px",
                color: "#991b1b",
                lineHeight: "1.4"
              }}
            >
              This will permanently delete the user's PostgreSQL database record, 512-D FaceNet embedding vector, FAISS search index, and cropped face image files. This action cannot be undone.
            </div>

            <div style={{ display: "flex", gap: "10px", justifyContent: "flex-end" }}>
              <button
                onClick={cancelDelete}
                disabled={isDeleting}
                style={{
                  padding: "8px 16px",
                  borderRadius: "6px",
                  border: "1px solid #d1d5db",
                  background: "#ffffff",
                  color: "#374151",
                  fontWeight: "600",
                  fontSize: "14px",
                  cursor: isDeleting ? "not-allowed" : "pointer"
                }}
              >
                Cancel
              </button>
              <button
                onClick={executeDelete}
                disabled={isDeleting}
                style={{
                  padding: "8px 18px",
                  borderRadius: "6px",
                  border: "none",
                  background: isDeleting ? "#9ca3af" : "#dc2626",
                  color: "#ffffff",
                  fontWeight: "600",
                  fontSize: "14px",
                  cursor: isDeleting ? "not-allowed" : "pointer"
                }}
              >
                {isDeleting ? "Deleting..." : "Yes, Delete User"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default UserManagement;

