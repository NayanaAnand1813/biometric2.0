import { useState, useEffect, useMemo } from "react";
import "./UserManagement.css";

function UserManagement() {
  const [users, setUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  // Security requirement: Never prepopulate or expose admin secrets in source code
  const [adminKey, setAdminKey] = useState("");
  const [showAdminKey, setShowAdminKey] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [feedback, setFeedback] = useState({ type: null, message: "" });
  const [confirmDialog, setConfirmDialog] = useState({ open: false, user: null });

  // Fetch all registered users from FastAPI backend
  const fetchUsers = async () => {
    setIsLoading(true);
    setFeedback({ type: null, message: "" });
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
        message: `Failed to query PostgreSQL registry: ${err.message || "Connection refused"}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    let ignore = false;
    async function initUsers() {
      try {
        const response = await fetch("http://127.0.0.1:8000/users");
        if (!response.ok) {
          throw new Error(`Server returned HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!ignore) {
          setUsers(data.users || []);
        }
      } catch (err) {
        if (!ignore) {
          setFeedback({
            type: "error",
            message: `Failed to query PostgreSQL registry: ${err.message || "Connection refused"}`
          });
        }
      }
    }

    initUsers();
    return () => {
      ignore = true;
    };
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

    if (!adminKey.trim()) {
      setFeedback({
        type: "error",
        message: "Administrator authorization key is required to authorize biometric profile deletion."
      });
      setConfirmDialog({ open: false, user: null });
      return;
    }

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
          message: data.message || `User '${user.user_id}' was completely purged from PostgreSQL, FAISS, and filesystem.`
        });
        setConfirmDialog({ open: false, user: null });
        // Automatically refresh user list
        await fetchUsers();
      } else {
        const errorDetail = data.detail?.message || data.detail?.error || data.message || `HTTP ${response.status} Error`;
        setFeedback({
          type: "error",
          message: `Deletion authorization failed: ${errorDetail}`
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

  // Filtered users calculation
  const filteredUsers = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return users;
    return users.filter((u) => {
      const idMatch = (u.user_id || "").toLowerCase().includes(q);
      const nameMatch = (u.full_name || "").toLowerCase().includes(q);
      const emailMatch = (u.email || "").toLowerCase().includes(q);
      const roleMatch = (u.role || "").toLowerCase().includes(q);
      return idMatch || nameMatch || emailMatch || roleMatch;
    });
  }, [users, searchQuery]);

  const totalSynced = useMemo(() => {
    return users.filter((u) => u.has_embedding).length;
  }, [users]);

  return (
    <div className="user-mgmt-container">
      {/* Header & Stat Summary */}
      <div className="user-mgmt-header">
        <div className="user-mgmt-title-group">
          <h2>
            <span>USER ACCESS MANAGEMENT</span>
            <span className="badge badge-cyber badge-mono">REGISTRY CONSOLE</span>
          </h2>
          <p>
            Audit and manage enrolled biometric profiles, PostgreSQL records, and FAISS vector indices.
          </p>
        </div>

        <div className="user-mgmt-header-actions">
          <div className="stat-chip">
            <span>ENROLLED:</span>
            <span className="stat-chip-count">{totalSynced} / {users.length}</span>
          </div>

          <button
            onClick={fetchUsers}
            disabled={isLoading}
            className="btn btn-secondary"
          >
            <span>{isLoading ? "Refreshing..." : "Refresh Registry"}</span>
          </button>
        </div>
      </div>

      {/* Filter & Security Controls Bar */}
      <div className="user-mgmt-controls-bar">
        {/* Search Input */}
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="cyber-input"
            placeholder="Filter by User ID, Name, Email, or Role..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        {/* Masked Admin Key Authorization Bar */}
        <div className="admin-key-wrapper">
          <div className="admin-key-label">
            <span>🔑 ADMIN KEY:</span>
          </div>
          <input
            type={showAdminKey ? "text" : "password"}
            value={adminKey}
            onChange={(e) => setAdminKey(e.target.value)}
            placeholder="Enter Admin Authorization Key..."
            className="admin-key-input"
            autoComplete="off"
          />
          <button
            type="button"
            onClick={() => setShowAdminKey(!showAdminKey)}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--text-muted)",
              cursor: "pointer",
              padding: "4px 8px",
              fontSize: "12px"
            }}
            title={showAdminKey ? "Mask Key" : "Show Key"}
          >
            {showAdminKey ? "Hide" : "Show"}
          </button>
        </div>
      </div>

      {/* Feedback Messages */}
      {feedback.message && (
        <div
          className={`cyber-feedback-box ${feedback.type === "success" ? "success" : "error"}`}
          style={{ marginBottom: "var(--space-md)", marginTop: 0 }}
        >
          <div className="feedback-icon">
            {feedback.type === "success" ? "✓" : "⚠️"}
          </div>
          <div className="feedback-content">
            <div>{feedback.message}</div>
          </div>
          <button
            onClick={() => setFeedback({ type: null, message: "" })}
            style={{
              background: "transparent",
              border: "none",
              cursor: "pointer",
              fontSize: "16px",
              color: "inherit",
              marginLeft: "auto"
            }}
          >
            ✕
          </button>
        </div>
      )}

      {/* Users Table Card Container */}
      <div className="table-card-container">
        {isLoading && users.length === 0 ? (
          <div className="table-loading-state">
            <span className="status-dot status-dot-cyan" style={{ width: "12px", height: "12px" }}></span>
            <div className="state-title">Querying PostgreSQL Vector Registry...</div>
            <div className="state-subtitle">Loading user profiles and verifying 512-D vector synchronization.</div>
          </div>
        ) : users.length === 0 ? (
          <div className="table-empty-state">
            <div className="state-icon">🛡️</div>
            <div className="state-title">No Registered Users in Registry</div>
            <div className="state-subtitle">
              Navigate to the Registration tab to enroll a new biometric profile.
            </div>
          </div>
        ) : filteredUsers.length === 0 ? (
          <div className="table-empty-state">
            <div className="state-icon">🔍</div>
            <div className="state-title">No Matching Records</div>
            <div className="state-subtitle">
              No users found matching query &ldquo;{searchQuery}&rdquo;. Try another filter term.
            </div>
          </div>
        ) : (
          <div className="table-scroll-wrapper">
            <table className="cyber-table">
              <thead>
                <tr>
                  <th>User ID</th>
                  <th>Full Name</th>
                  <th>Work Email</th>
                  <th>Role</th>
                  <th>Registration Date</th>
                  <th>Face Embedding Status</th>
                  <th className="align-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map((user) => (
                  <tr key={user.user_id}>
                    <td>
                      <span className="user-id-badge">{user.user_id}</span>
                    </td>
                    <td style={{ fontWeight: 600 }}>
                      {user.full_name || "—"}
                    </td>
                    <td style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)", fontSize: "12.5px" }}>
                      {user.email || "—"}
                    </td>
                    <td>
                      {user.role ? (
                        <span className="badge badge-cyber badge-mono">{user.role}</span>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td>
                      <span className="user-date-text">
                        {user.created_at ? new Date(user.created_at).toLocaleString() : "—"}
                      </span>
                    </td>
                    <td>
                      {user.has_embedding ? (
                        <span className="status-badge-synced">
                          ● 512-D VECTOR SYNCED
                        </span>
                      ) : (
                        <span className="status-badge-missing">
                          ● VECTOR MISSING
                        </span>
                      )}
                    </td>
                    <td className="align-right">
                      <button
                        onClick={() => promptDeleteUser(user)}
                        className="btn btn-danger btn-table-delete"
                      >
                        Delete Profile
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {confirmDialog.open && confirmDialog.user && (
        <div className="modal-backdrop" onClick={cancelDelete}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header-section">
              <div className="modal-warning-icon">⚠️</div>
              <div className="modal-title-group">
                <h3>Confirm Biometric Profile Purge</h3>
                <p>ACTION CANNOT BE UNDONE</p>
              </div>
            </div>

            <p className="modal-body-text">
              Are you sure you want to permanently delete user{" "}
              <span className="modal-user-target">{confirmDialog.user.user_id}</span>?
              <br /><br />
              This will purge the PostgreSQL user profile, permanently delete the 512-D FaceNet embedding, remove the vector from FAISS IndexFlatIP, and delete all associated cropped face images.
            </p>

            <div className="modal-actions">
              <button
                onClick={cancelDelete}
                disabled={isDeleting}
                className="btn btn-outline"
              >
                Cancel
              </button>
              <button
                onClick={executeDelete}
                disabled={isDeleting}
                className="btn btn-danger"
              >
                {isDeleting ? "Purging User..." : "Confirm & Delete Profile"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default UserManagement;
