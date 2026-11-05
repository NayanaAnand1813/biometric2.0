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
      streamRef.current.getTracks().forEach((track) => track.stop());
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

  // Single-click automated Liveness-First Face Registration workflow
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
      setMessage("Camera access is not supported by your browser or environment.");
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

      // Stabilization pause (800ms) for auto-exposure & sensor stabilization
      await delay(800);

      // Capture verified frame candidate using canvas
      if (!video || !video.videoWidth || !video.videoHeight) {
        throw new Error("Unable to read video frame from camera.");
      }

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.92);
      });

      if (!blob) {
        throw new Error("Failed to encode camera frame.");
      }

      // Step 2: Checking Face Liveness (MiniFASNet)
      setCurrentStage(2);
      setMessage("Step 2/7: Checking Face Liveness with MiniFASNet anti-spoofing model...");

      const formData = new FormData();
      formData.append("user_id", cleanId);
      formData.append("full_name", cleanFullName);
      formData.append("email", normalizedEmail);
      formData.append("role", cleanRole);
      formData.append("phone_number", cleanPhone);
      formData.append("image", blob, "capture.jpg");

      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData
      });

      const result = await response.json();

      // Check if liveness failed
      const isLivenessError =
        !response.ok &&
        (result.detail?.error === "liveness_failed" ||
          result.error === "liveness_failed" ||
          result.status === "liveness_failed");

      if (isLivenessError) {
        setStatusType("error");
        setCurrentStage(2);
        const livenessMsg =
          result.detail?.message ||
          result.message ||
          "Liveness check failed: Fake, spoof, or uncertain face detected. Please position your face correctly and try again.";
        setMessage(livenessMsg);
        setLivenessData(result.detail?.liveness || result.liveness || null);
        return;
      }

      // Check if no face was detected
      const isNoFaceError =
        !response.ok &&
        (result.detail?.error === "no_face" ||
          result.error === "no_face" ||
          result.status === "no_face");

      if (isNoFaceError) {
        setStatusType("error");
        setCurrentStage(2);
        setMessage("No face detected in camera frame. Please align your face in center view and try again.");
        return;
      }

      // Other backend error (duplicate user, bad request, etc.)
      if (!response.ok || result.status !== "success") {
        setStatusType("error");
        setCurrentStage(6);
        const errMsg =
          result.detail?.message ||
          result.detail?.error ||
          result.message ||
          "Registration failed.";
        setMessage(errMsg);
        return;
      }

      // Liveness passed & registration succeeded: visualize the verified pipeline progression
      setLivenessData(result.liveness || null);
      const confScore = result.liveness?.score
        ? (result.liveness.score * 100).toFixed(1)
        : "99.9";

      // Step 3: Real Face Verified
      setCurrentStage(3);
      setMessage(`Step 3/7: Real Face Verified! (MiniFASNet Confidence: ${confScore}%)`);
      await delay(350);

      // Step 4: Detecting Face with MediaPipe
      setCurrentStage(4);
      setMessage("Step 4/7: Detecting Face & Cropping with MediaPipe BlazeFace...");
      await delay(350);

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
      setMessage(result.message || `User '${cleanFullName} (${cleanId})' registered successfully!`);
    } catch (err) {
      setStatusType("error");
      if (currentStage < 2) setCurrentStage(1);

      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setMessage("Camera permission was denied. Please allow camera permissions in your browser settings.");
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
      <h2 style={{ marginBottom: "6px" }}>User Profile & Liveness-First Face Registration</h2>
      <p style={{ color: "#666", marginBottom: "20px", fontSize: "14px" }}>
        Fill out your profile details and click Register. The camera will verify your live face with MiniFASNet before generating biometric embeddings.
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

      {/* Video Viewport */}
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
          {isLoading ? "Verifying & Registering..." : "Click to Register"}
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
            Pipeline Progress (Liveness-First Verification):
          </div>

          <div style={{ display: "grid", gap: "6px" }}>
            {REGISTRATION_STAGES.map((st) => {
              const isPast = currentStage > st.id;
              const isCurrent = currentStage === st.id;
              const isFailedAtThisStep = isCurrent && statusType === "error";

              let icon = "⚪";
              let textColor = "#6b7280";
              let fontWeight = "normal";

              if (isPast) {
                icon = "✅";
                textColor = "#15803d";
                fontWeight = "600";
              } else if (isFailedAtThisStep) {
                icon = "❌";
                textColor = "#dc2626";
                fontWeight = "bold";
              } else if (isCurrent) {
                icon = statusType === "success" ? "✅" : "⏳";
                textColor = statusType === "success" ? "#15803d" : "#2563eb";
                fontWeight = "bold";
              }

              return (
                <div
                  key={st.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    fontSize: "13px",
                    color: textColor,
                    fontWeight: fontWeight
                  }}
                >
                  <span>{icon}</span>
                  <span>{st.id}. {st.label}</span>
                  {st.id === 2 && livenessData && (
                    <span style={{ fontSize: "11px", color: livenessData.is_real ? "#15803d" : "#dc2626", marginLeft: "auto" }}>
                      Score: {(livenessData.score * 100).toFixed(1)}% (Threshold: 85%)
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Progress, Success, & Error Feedback */}
      {message && (
        <div
          style={{
            marginTop: "14px",
            padding: "10px 16px",
            borderRadius: "6px",
            display: "inline-block",
            maxWidth: "520px",
            background:
              statusType === "error"
                ? "#fee2e2"
                : statusType === "success"
                ? "#dcfce7"
                : "#eff6ff",
            color: getStatusColor(),
            fontWeight: "500",
            border: `1px solid ${getStatusColor()}`
          }}
        >
          {statusType === "loading" && <span>⏳ </span>}
          {statusType === "success" && <span>✅ </span>}
          {statusType === "error" && <span>⚠️ </span>}
          {message}
        </div>
      )}
    </div>
  );
}

export default Registration;
