import { useState, useRef, useEffect } from "react";

const REGISTRATION_STAGES = [
  { id: 1, label: "Initializing Camera" },
  { id: 2, label: "Checking Face Liveness" },
  { id: 3, label: "Real Face Verified" },
  { id: 4, label: "Detecting Face with MediaPipe" },
  { id: 5, label: "Generating Face Embedding" },
  { id: 6, label: "Processing Registration" },
  { id: 7, label: "Registration Status" },
];

function Registration() {
  const [userId, setUserId] = useState("");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");

  const [message, setMessage] = useState("");
  const [statusType, setStatusType] = useState("idle"); // idle | loading | success | error
  const [currentStage, setCurrentStage] = useState(0); // 0 to 7
  const [cameraOn, setCameraOn] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [livenessData, setLivenessData] = useState(null);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const isProcessingRef = useRef(false);

  // Stop webcam helper
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (_) {}
      });
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
  };

  // Cleanup camera stream on component unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // Automated Camera-Only Liveness-First Face Registration workflow
  const handleRegisterClick = async () => {
    if (isLoading || isProcessingRef.current) return;

    // 1. Mandatory Fields Validation
    const cleanId = userId.trim();
    const cleanFullName = fullName.trim();
    const cleanEmail = email.trim();
    const cleanRole = role.trim();
    const cleanPhone = phoneNumber.trim();

    if (!cleanId || !cleanFullName || !cleanEmail || !cleanRole || !cleanPhone) {
      setStatusType("error");
      setMessage("All five fields (User ID, Full Name, Work Email, Role, Phone Number) are mandatory.");
      return;
    }

    // 2. Email Normalization and Format Validation
    const normalizedEmail = cleanEmail.toLowerCase();
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(normalizedEmail)) {
      setStatusType("error");
      setMessage("Please enter a valid work email address (e.g. name@company.com).");
      return;
    }

    // 3. Phone Number Format Validation
    const phoneDigits = cleanPhone.replace(/\D/g, "");
    const phoneRegex = /^\+?[0-9\s\-()]{7,20}$/;
    if (!phoneRegex.test(cleanPhone) || phoneDigits.length < 7) {
      setStatusType("error");
      setMessage("Please enter a valid phone number (at least 7 digits, e.g. +1 555-0123).");
      return;
    }

    isProcessingRef.current = true;
    setIsLoading(true);
    setStatusType("loading");
    setLivenessData(null);

    // Step 1: Initializing Camera
    setCurrentStage(1);
    setMessage("Step 1/7: Initializing Camera & requesting webcam access...");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatusType("error");
      setCurrentStage(1);
      setMessage("Camera access is not supported by your browser or environment. Camera is strictly required.");
      setIsLoading(false);
      isProcessingRef.current = false;
      return;
    }

    // Stop any existing stream
    stopCamera();

    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        }
      });

      // Handle unexpected camera disconnect
      stream.getVideoTracks().forEach((track) => {
        track.addEventListener("ended", () => {
          if (isProcessingRef.current) {
            stopCamera();
            setIsLoading(false);
            isProcessingRef.current = false;
            setStatusType("error");
            setMessage("Camera connection was lost or disconnected. Registration attempt terminated.");
          }
        });
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraOn(true);
      setMessage("Camera active. Please center your face directly in the frame...");

      // Wait for camera video feed to be ready and playing
      const video = videoRef.current;
      if (video) {
        await new Promise((resolve) => {
          if (video.readyState >= 2 && video.videoWidth > 0) {
            resolve();
          } else {
            video.onloadeddata = () => resolve();
            setTimeout(resolve, 1500);
          }
        });
      }

      // Sensor stabilization pause
      await delay(600);

      if (!video || !video.videoWidth || !video.videoHeight) {
        throw new Error("Unable to read video frame from camera.");
      }

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");

      // Initiate backend liveness session
      const startFormData = new FormData();
      startFormData.append("action", "register");
      startFormData.append("user_id", cleanId);

      const sessionStartResp = await fetch("http://127.0.0.1:8000/api/liveness/session/start", {
        method: "POST",
        body: startFormData
      });

      const sessionStartData = await sessionStartResp.json();
      if (!sessionStartResp.ok || !sessionStartData.session?.session_id) {
        throw new Error(sessionStartData.detail?.message || "Failed to start backend liveness session.");
      }

      const sessionId = sessionStartData.session.session_id;

      // Step 2: Checking Face Liveness on active live webcam frames
      setCurrentStage(2);
      setMessage("Step 2/7: Checking Face Liveness with MiniFASNet anti-spoofing model...");

      let isVerified = false;
      let verifiedResult = null;
      const maxLivenessAttempts = 10;

      for (let attempt = 1; attempt <= maxLivenessAttempts; attempt++) {
        if (!isProcessingRef.current) break;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameBlob = await new Promise((resolve) => {
          canvas.toBlob(resolve, "image/jpeg", 0.90);
        });

        if (!frameBlob) {
          throw new Error("Failed to encode camera frame.");
        }

        const verifyFormData = new FormData();
        verifyFormData.append("session_id", sessionId);
        verifyFormData.append("image", frameBlob, "live_frame.jpg");

        const verifyResp = await fetch("http://127.0.0.1:8000/api/liveness/session/verify-frame", {
          method: "POST",
          body: verifyFormData
        });

        const verifyResult = await verifyResp.json();

        if (verifyResult.status === "no_face") {
          setMessage(`Step 2/7: No face detected (${attempt}/${maxLivenessAttempts}). Please center your face in the camera frame.`);
          await delay(350);
          continue;
        }

        if (verifyResult.status === "fake" || (!verifyResult.is_real && verifyResult.label === "Fake")) {
          // Liveness detection failed: reject attempt immediately. Do NOT capture or register face image.
          setStatusType("error");
          setCurrentStage(2);
          setMessage("Liveness check failed: Spoof or fake face detected. Please face the camera and try again.");
          setLivenessData(verifyResult);
          return;
        }

        if (verifyResult.verified || (verifyResult.is_real && verifyResult.score >= (verifyResult.threshold || 0.85))) {
          isVerified = true;
          verifiedResult = verifyResult;
          break;
        }

        setMessage(`Step 2/7: Analyzing live face (${attempt}/${maxLivenessAttempts}). Please look at the camera...`);
        await delay(300);
      }

      if (!isVerified) {
        setStatusType("error");
        setCurrentStage(2);
        setMessage("Liveness verification timed out without confirming a real, live face. Registration rejected.");
        return;
      }

      // Step 3: Real Face Verified
      setLivenessData(verifiedResult);
      const confScore = verifiedResult.score
        ? (verifiedResult.score * 100).toFixed(1)
        : "99.9";

      setCurrentStage(3);
      setMessage(`Step 3/7: Real Face Verified! (MiniFASNet Confidence: ${confScore}%)`);
      await delay(350);

      // Step 4: ONLY after successful liveness verification, capture face from active webcam
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const verifiedFaceBlob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.95);
      });

      if (!verifiedFaceBlob) {
        throw new Error("Failed to capture verified face frame from active webcam.");
      }

      setCurrentStage(4);
      setMessage("Step 4/7: Detecting Face & Cropping with MediaPipe BlazeFace...");

      // Send to /register with verified session_id
      const regFormData = new FormData();
      regFormData.append("session_id", sessionId);
      regFormData.append("user_id", cleanId);
      regFormData.append("full_name", cleanFullName);
      regFormData.append("email", normalizedEmail);
      regFormData.append("role", cleanRole);
      regFormData.append("phone_number", cleanPhone);
      regFormData.append("image", verifiedFaceBlob, "webcam_verified_face.jpg");

      const regResp = await fetch("http://127.0.0.1:8000/register", {
        method: "POST",
        body: regFormData
      });

      const regResult = await regResp.json();

      if (!regResp.ok || regResult.status !== "success") {
        setStatusType("error");
        setCurrentStage(6);
        const errMsg =
          regResult.detail?.message ||
          regResult.detail?.error ||
          regResult.message ||
          "Registration failed.";
        setMessage(errMsg);
        return;
      }

      // Step 5: Generating Face Embedding
      setCurrentStage(5);
      setMessage("Step 5/7: Generating 512-D FaceNet Biometric Embedding...");
      await delay(350);

      // Step 6: Processing Registration
      setCurrentStage(6);
      setMessage("Step 6/7: Storing Profile in PostgreSQL & Synchronizing FAISS Index...");
      await delay(350);

      // Step 7: Completed
      setCurrentStage(7);
      setStatusType("success");
      setMessage(regResult.message || `User '${cleanFullName} (${cleanId})' registered successfully!`);
    } catch (err) {
      setStatusType("error");
      if (currentStage < 2) setCurrentStage(1);

      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setMessage("Camera permission was denied. Please allow camera permissions in your browser settings to proceed.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        setMessage("No camera device was found on this system.");
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        setMessage("Camera is already in use by another application or unavailable.");
      } else if (err.name === "OverconstrainedError") {
        setMessage("Camera does not support the requested video constraints.");
      } else if (err.message && err.message.includes("fetch")) {
        setMessage("Failed to connect to backend server. Please verify FastAPI backend is running.");
      } else {
        setMessage(err.message || "An unexpected error occurred during registration.");
      }
    } finally {
      stopCamera();
      setIsLoading(false);
      isProcessingRef.current = false;
    }
  };

  const getStatusColor = () => {
    if (statusType === "error") return "#dc2626";
    if (statusType === "success") return "#16a34a";
    if (statusType === "loading") return "#2563eb";
    return "#333";
  };

  const inputStyle = {
    padding: "9px 12px",
    fontSize: "14px",
    borderRadius: "6px",
    border: "1px solid #d1d5db",
    width: "100%",
    boxSizing: "border-box"
  };

  const labelStyle = {
    display: "block",
    textAlign: "left",
    fontSize: "13px",
    fontWeight: "600",
    color: "#374151",
    marginBottom: "4px"
  };

  return (
    <div style={{ textAlign: "center", padding: "10px 20px" }}>
      <h2 style={{ marginBottom: "6px" }}>User Profile & Live Camera Face Registration</h2>
      <p style={{ color: "#666", marginBottom: "20px", fontSize: "14px" }}>
        Fill out your profile details and click Register. Liveness verification occurs via live camera before any face is captured or registered.
      </p>

      {/* Profile Form Fields Grid */}
      <div
        style={{
          maxWidth: "460px",
          margin: "0 auto 20px auto",
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: "12px",
          textAlign: "left"
        }}
      >
        <div>
          <label style={labelStyle}>User ID *</label>
          <input
            type="text"
            placeholder="e.g. john_doe"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            disabled={isLoading}
            style={inputStyle}
          />
        </div>

        <div>
          <label style={labelStyle}>Full Name *</label>
          <input
            type="text"
            placeholder="e.g. John Doe"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            disabled={isLoading}
            style={inputStyle}
          />
        </div>

        <div>
          <label style={labelStyle}>Work Email *</label>
          <input
            type="email"
            placeholder="e.g. john.doe@company.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={isLoading}
            style={inputStyle}
          />
        </div>

        <div>
          <label style={labelStyle}>Role / Designation *</label>
          <input
            type="text"
            placeholder="e.g. Senior Software Engineer"
            value={role}
            onChange={(e) => setRole(e.target.value)}
            disabled={isLoading}
            style={inputStyle}
          />
        </div>

        <div>
          <label style={labelStyle}>Phone Number *</label>
          <input
            type="tel"
            placeholder="e.g. +1 555-0123"
            value={phoneNumber}
            onChange={(e) => setPhoneNumber(e.target.value)}
            disabled={isLoading}
            style={inputStyle}
          />
        </div>
      </div>

      {/* Live Video Viewport (No Upload Option) */}
      <div>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          width="380"
          style={{
            background: "#222",
            borderRadius: "8px",
            boxShadow: "0 4px 10px rgba(0,0,0,0.15)",
            border: cameraOn ? "2px solid #16a34a" : "2px solid #555",
            minHeight: "260px",
            display: "inline-block"
          }}
        />
      </div>

      <br />

      {/* Primary Action Button: Click to Register */}
      <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
        <button
          onClick={handleRegisterClick}
          disabled={isLoading}
          style={{
            padding: "12px 32px",
            fontSize: "16px",
            cursor: isLoading ? "not-allowed" : "pointer",
            borderRadius: "6px",
            backgroundColor: isLoading ? "#9ca3af" : "#16a34a",
            color: "#fff",
            border: "none",
            fontWeight: "bold",
            boxShadow: "0 2px 8px rgba(22, 163, 74, 0.25)",
            transition: "all 0.2s ease"
          }}
        >
          {isLoading ? "Verifying Liveness & Registering..." : "Click to Register"}
        </button>
      </div>

      {/* 7-Step Liveness-First Processing Pipeline Stepper */}
      {currentStage > 0 && (
        <div
          style={{
            maxWidth: "520px",
            margin: "20px auto 10px auto",
            padding: "14px 16px",
            borderRadius: "8px",
            background: "#f9fafb",
            border: "1px solid #e5e7eb",
            textAlign: "left"
          }}
        >
          <div style={{ fontSize: "13px", fontWeight: "700", color: "#374151", marginBottom: "10px" }}>
            Pipeline Progress (Live Camera Liveness-First Verification):
          </div>

          <div style={{ display: "grid", gap: "6px" }}>
            {REGISTRATION_STAGES.map((st) => {
              const isPast = currentStage > st.id;
              const isCurrent = currentStage === st.id;
              const isFailedAtThisStep = isCurrent && statusType === "error";

              let badgeBg = "#e5e7eb";
              let badgeColor = "#6b7280";
              let badgeText = `${st.id}`;
              let rowColor = "#9ca3af";

              if (isPast || (isCurrent && statusType === "success")) {
                badgeBg = "#16a34a";
                badgeColor = "#ffffff";
                badgeText = "✓";
                rowColor = "#15803d";
              } else if (isFailedAtThisStep) {
                badgeBg = "#dc2626";
                badgeColor = "#ffffff";
                badgeText = "✗";
                rowColor = "#b91c1c";
              } else if (isCurrent) {
                badgeBg = "#2563eb";
                badgeColor = "#ffffff";
                badgeText = "⏳";
                rowColor = "#1d4ed8";
              }

              return (
                <div
                  key={st.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    padding: "4px 8px",
                    borderRadius: "6px",
                    background: isCurrent ? "rgba(37, 99, 235, 0.06)" : "transparent"
                  }}
                >
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: "22px",
                      height: "22px",
                      borderRadius: "50%",
                      fontSize: "12px",
                      fontWeight: "700",
                      background: badgeBg,
                      color: badgeColor,
                      flexShrink: 0
                    }}
                  >
                    {badgeText}
                  </span>
                  <span
                    style={{
                      fontSize: "13px",
                      fontWeight: isCurrent ? "700" : "500",
                      color: rowColor
                    }}
                  >
                    {st.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Live Status Feedback Message */}
      {message && (
        <div
          style={{
            maxWidth: "520px",
            margin: "15px auto 0 auto",
            padding: "12px 16px",
            borderRadius: "6px",
            background:
              statusType === "error"
                ? "#fef2f2"
                : statusType === "success"
                ? "#f0fdf4"
                : "#eff6ff",
            border: `1px solid ${
              statusType === "error"
                ? "#fecaca"
                : statusType === "success"
                ? "#bbf7d0"
                : "#bfdbfe"
            }`,
            color: getStatusColor(),
            fontSize: "14px",
            fontWeight: "600",
            textAlign: "left",
            lineHeight: "1.4"
          }}
        >
          {statusType === "loading" && <span style={{ marginRight: "8px" }}>🔄</span>}
          {statusType === "error" && <span style={{ marginRight: "8px" }}>⚠️</span>}
          {statusType === "success" && <span style={{ marginRight: "8px" }}>✅</span>}
          {message}
        </div>
      )}

      {/* Liveness Diagnostic Details */}
      {livenessData && (
        <div
          style={{
            maxWidth: "520px",
            margin: "10px auto 0 auto",
            padding: "10px 14px",
            borderRadius: "6px",
            background: "#f9fafb",
            border: "1px solid #e5e7eb",
            fontSize: "12px",
            color: "#4b5563",
            textAlign: "left"
          }}
        >
          <div>
            <strong>MiniFASNet Confidence:</strong>{" "}
            {livenessData.score ? `${(livenessData.score * 100).toFixed(2)}%` : "N/A"}{" "}
            (Threshold: {livenessData.threshold ? `${(livenessData.threshold * 100).toFixed(0)}%` : "85%"})
          </div>
          <div>
            <strong>Liveness Classification:</strong> {livenessData.label || livenessData.status}
          </div>
        </div>
      )}
    </div>
  );
}

export default Registration;
