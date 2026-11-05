import { useState, useRef, useEffect } from "react";

function Registration() {
  const [userId, setUserId] = useState("");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");

  const [message, setMessage] = useState("");
  const [statusType, setStatusType] = useState("idle"); // idle | loading | success | error
  const [cameraOn, setCameraOn] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

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

  // Single-click automated Face Registration workflow
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
    setMessage("Requesting camera access...");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatusType("error");
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
      setMessage("Camera started. Align your face...");

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
      setMessage("Capturing biometric face frame...");
      await new Promise((resolve) => setTimeout(resolve, 800));

      // Capture frame using canvas
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

      // Send registration request to backend (POST /upload) with profile data
      setMessage("Storing profile & generating biometric embedding...");
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

      if (response.ok && result.status !== "no_face" && result.status !== "error") {
        setStatusType("success");
        setMessage(result.message || `User '${cleanFullName} (${cleanId})' registered successfully!`);
      } else {
        setStatusType("error");
        const errMsg =
          result.detail?.message ||
          result.detail?.error ||
          result.message ||
          result.error ||
          "Registration failed.";
        setMessage(errMsg);
      }
    } catch (err) {
      setStatusType("error");

      // Handle specific camera permission and availability errors
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
      // Camera is automatically stopped after registration succeeds, fails, or on error
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
      <h2 style={{ marginBottom: "6px" }}>User Profile & Biometric Registration</h2>
      <p style={{ color: "#666", marginBottom: "20px", fontSize: "14px" }}>
        Fill out your profile details and click Register to automatically capture your face
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
          {isLoading ? "Registering..." : "Click to Register"}
        </button>
      </div>

      {/* Progress, Success, & Error Feedback */}
      {message && (
        <div
          style={{
            marginTop: "18px",
            padding: "10px 16px",
            borderRadius: "6px",
            display: "inline-block",
            maxWidth: "480px",
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
