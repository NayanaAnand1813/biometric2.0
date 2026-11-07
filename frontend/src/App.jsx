import { useState } from "react";
import Registration from "./components/Registration";
import Login from "./components/Login";
import UserManagement from "./components/UserManagement";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState("login"); // default to login, registration, or users

  return (
    <div className="app-shell">
      {/* Enterprise Terminal Header */}
      <header className="app-header">
        <div className="brand-badge">
          <span className="status-dot status-dot-cyan"></span>
          OJIX SECURECORE &bull; BIOMETRIC TERMINAL
        </div>
        <h1 className="app-title">Biometric Identity &amp; Access Terminal</h1>
        <p className="app-subtitle">
          Real-time Face Anti-Spoofing &bull; 1:N FAISS Vector Identification &bull; Zero Disk Storage
        </p>
      </header>

      {/* Segmented Navigation Switcher */}
      <nav className="cyber-nav-container" aria-label="Terminal Navigation">
        <button
          onClick={() => setActiveTab("login")}
          className={`cyber-nav-tab ${activeTab === "login" ? "active" : ""}`}
        >
          LOGIN
        </button>

        <button
          onClick={() => setActiveTab("register")}
          className={`cyber-nav-tab ${activeTab === "register" ? "active" : ""}`}
        >
          REGISTER
        </button>

        <button
          onClick={() => setActiveTab("users")}
          className={`cyber-nav-tab ${activeTab === "users" ? "active" : ""}`}
        >
          USER MANAGEMENT
        </button>
      </nav>

      {/* Main Terminal Panel Content */}
      <main className="app-main-panel">
        {activeTab === "register" && <Registration />}
        {activeTab === "login" && <Login />}
        {activeTab === "users" && <UserManagement />}
      </main>
    </div>
  );
}

export default App;
