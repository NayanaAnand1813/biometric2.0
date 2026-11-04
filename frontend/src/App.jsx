import { useState } from "react";
import Registration from "./components/Registration";
import Login from "./components/Login";
import UserManagement from "./components/UserManagement";

function App() {
  const [activeTab, setActiveTab] = useState("login"); // default to login, registration, or users

  return (
    <div style={{ maxWidth: "780px", margin: "30px auto", fontFamily: "system-ui, sans-serif" }}>
      <header style={{ textAlign: "center", marginBottom: "25px" }}>
        <h1 style={{ marginBottom: "8px", color: "#111827" }}>Biometric Face Recognition</h1>
        <p style={{ color: "#6b7280", margin: 0 }}>
          FastAPI &amp; React Face Biometrics Platform
        </p>
      </header>

      {/* Navigation Switcher */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: "8px",
          background: "#f3f4f6",
          padding: "6px",
          borderRadius: "8px",
          marginBottom: "20px"
        }}
      >
        <button
          onClick={() => setActiveTab("register")}
          style={{
            flex: 1,
            padding: "10px 18px",
            fontSize: "15px",
            fontWeight: "600",
            cursor: "pointer",
            border: "none",
            borderRadius: "6px",
            background: activeTab === "register" ? "#fff" : "transparent",
            color: activeTab === "register" ? "#1f2937" : "#6b7280",
            boxShadow: activeTab === "register" ? "0 2px 4px rgba(0,0,0,0.08)" : "none",
            transition: "all 0.2s ease"
          }}
        >
          Registration
        </button>

        <button
          onClick={() => setActiveTab("login")}
          style={{
            flex: 1,
            padding: "10px 18px",
            fontSize: "15px",
            fontWeight: "600",
            cursor: "pointer",
            border: "none",
            borderRadius: "6px",
            background: activeTab === "login" ? "#fff" : "transparent",
            color: activeTab === "login" ? "#1f2937" : "#6b7280",
            boxShadow: activeTab === "login" ? "0 2px 4px rgba(0,0,0,0.08)" : "none",
            transition: "all 0.2s ease"
          }}
        >
          Face Login
        </button>

        <button
          onClick={() => setActiveTab("users")}
          style={{
            flex: 1,
            padding: "10px 18px",
            fontSize: "15px",
            fontWeight: "600",
            cursor: "pointer",
            border: "none",
            borderRadius: "6px",
            background: activeTab === "users" ? "#fff" : "transparent",
            color: activeTab === "users" ? "#1f2937" : "#6b7280",
            boxShadow: activeTab === "users" ? "0 2px 4px rgba(0,0,0,0.08)" : "none",
            transition: "all 0.2s ease"
          }}
        >
          Manage Users
        </button>
      </div>

      {/* Tab Content */}
      <main
        style={{
          background: "#ffffff",
          borderRadius: "10px",
          boxShadow: "0 4px 12px rgba(0,0,0,0.06)",
          padding: "20px",
          border: "1px solid #e5e7eb"
        }}
      >
        {activeTab === "register" && <Registration />}
        {activeTab === "login" && <Login />}
        {activeTab === "users" && <UserManagement />}
      </main>
    </div>
  );
}

export default App;

