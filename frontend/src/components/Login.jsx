import { useState, useRef, useEffect } from "react";

function Login() {
  const [message, setMessage] = useState("");
  const [statusType, setStatusType] = useState("idle"); // idle | loading | success | error
  const [cameraOn, setCameraOn] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [authenticatedUser, setAuthenticatedUser] = useState(null);
  const [authenticatedProfile, setAuthenticatedProfile] = useState(null);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const isProcessingRef = useRef(false);

  // Stop camera helper
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

  // Single-click automated Face Login workflow
  const handleLoginClick = async () => {
    if (isLoading || isProcessingRef.current) return;
    isProcessingRef.current = true;

    // Reset previous feedback
    setStatusType("loading");
    setMessage("Requesting camera access...");
    setIsLoading(true);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatusType("error");
      setMessage("Camera access is not supported by your browser or environment.");
      setIsLoading(false);
      isProcessingRef.current = false;
      return;
    }

    // Prevent multiple camera streams: stop any existing stream first
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
      setMessage("Camera started. Aligning face...");

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

      // Brief stabilization pause (800ms) for auto-exposure & sensor adjustment
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

      // Send biometric identification request (1:N login, no user_id required)
      setMessage("Identifying biometric profile in FAISS...");
      const formData = new FormData();
      formData.append("image", blob, "login_capture.jpg");

      const response = await fetch("http://127.0.0.1:8000/login", {
        method: "POST",
        body: formData
      });

      const result = await response.json();

      if (response.ok && result.authenticated) {
        setAuthenticatedUser(result.user_id);
        setAuthenticatedProfile(result.user_profile || null);
        setStatusType("success");
        setMessage(result.message || `Welcome, ${result.user_id}!`);
      } else {
        setAuthenticatedUser(null);
        setAuthenticatedProfile(null);
        setStatusType("error");
        const errMsg =
          result.detail?.message ||
          result.detail?.error ||
          result.message ||
          "Access Denied: Biometric verification failed.";
        setMessage(errMsg);
      }
    } catch (err) {
      setAuthenticatedUser(null);
      setAuthenticatedProfile(null);
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
        setMessage(err.message || "An unexpected error occurred during login.");
      }
    } finally {
      // Camera is automatically stopped after authentication succeeds, fails, or on error
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

  // Render Access Granted / Welcome screen if authenticated
  if (authenticatedUser) {
    const displayName = authenticatedProfile?.full_name || authenticatedUser;

    return (
      <div style={{ textAlign: "center", padding: "20px" }}>
        <div
          style={{
            maxWidth: "520px",
            margin: "0 auto",
            padding: "24px",
            borderRadius: "12px",
            background: "#f0fdf4",
            border: "2px solid #16a34a",
            boxShadow: "0 4px 14px rgba(22, 163, 74, 0.15)"
          }}
        >
          <div style={{ fontSize: "48px", marginBottom: "8px" }}>🎉</div>
          <h2 style={{ color: "#15803d", margin: "0 0 6px 0" }}>Access Granted!</h2>
          <p style={{ fontSize: "17px", color: "#166534", margin: "0 0 18px 0" }}>
            Welcome back, <strong>{displayName}</strong>
          </p>

          <div
            style={{
              background: "#ffffff",
              padding: "18px 20px",
              borderRadius: "8px",
              border: "1px solid #bbf7d0",
              textAlign: "left",
              marginBottom: "20px",
              fontSize: "14px",
              color: "#374151"
            }}
          >
            <div style={{ marginBottom: "12px", borderBottom: "1px solid #e5e7eb", paddingBottom: "8px" }}>
              <span style={{ fontWeight: "600", color: "#15803d" }}>Biometric Verification: </span>
              <span style={{ color: "#16a34a", fontWeight: "bold" }}>● Successful</span>
            </div>

            <div style={{ marginBottom: "8px", display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontWeight: "600", color: "#4b5563" }}>Full Name:</span>
              <span style={{ fontWeight: "600", color: "#111827" }}>
                {authenticatedProfile?.full_name || "Not Provided"}
              </span>
            </div>

            <div style={{ marginBottom: "8px", display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontWeight: "600", color: "#4b5563" }}>User ID:</span>
              <code style={{ background: "#f3f4f6", padding: "2px 8px", borderRadius: "4px", fontWeight: "600" }}>
                {authenticatedProfile?.user_id || authenticatedUser}
              </code>
            </div>

            <div style={{ marginBottom: "8px", display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontWeight: "600", color: "#4b5563" }}>Work Email:</span>
              <span style={{ color: "#111827" }}>
                {authenticatedProfile?.email || "Not Provided"}
              </span>
            </div>

            <div style={{ marginBottom: "8px", display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontWeight: "600", color: "#4b5563" }}>Role / Designation:</span>
              <span style={{ color: "#111827" }}>
                {authenticatedProfile?.role || "Not Provided"}
              </span>
            </div>

            <div style={{ marginBottom: "12px", display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontWeight: "600", color: "#4b5563" }}>Phone Number:</span>
              <span style={{ color: "#111827" }}>
                {authenticatedProfile?.phone_number || "Not Provided"}
              </span>
            </div>

            <div style={{ borderTop: "1px solid #f3f4f6", paddingTop: "8px", fontSize: "12px", color: "#6b7280" }}>
              Pipeline: BlazeFace Detection → FaceNet 512-D → FAISS 1:N Exact Match
            </div>
          </div>

          <button
            onClick={() => {
              setAuthenticatedUser(null);
              setAuthenticatedProfile(null);
              setStatusType("idle");
              setMessage("");
            }}
            style={{
              padding: "10px 24px",
              fontSize: "15px",
              fontWeight: "600",
              cursor: "pointer",
              borderRadius: "6px",
              backgroundColor: "#15803d",
              color: "#fff",
              border: "none",
              boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
              transition: "background-color 0.2s"
            }}
          >
            Logout / Authenticate Another User
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h2>User Login</h2>
      <p style={{ color: "#666", marginBottom: "20px" }}>
        Look at the camera and click below to authenticate your biometric profile
      </p>

      {/* Camera Video Stream */}
      <div>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          width="400"
          style={{
            background: "#222",
            borderRadius: "8px",
            boxShadow: "0 4px 10px rgba(0,0,0,0.15)",
            border: cameraOn ? "2px solid #16a34a" : "2px solid #555",
            minHeight: "300px",
            display: "inline-block"
          }}
        />
      </div>

      <br />

      {/* Primary Action Button */}
      <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
        <button
          onClick={handleLoginClick}
          disabled={isLoading}
          style={{
            padding: "12px 28px",
            fontSize: "16px",
            cursor: isLoading ? "not-allowed" : "pointer",
            borderRadius: "6px",
            backgroundColor: isLoading ? "#9ca3af" : "#2563eb",
            color: "#fff",
            border: "none",
            fontWeight: "bold",
            boxShadow: "0 2px 8px rgba(37, 99, 235, 0.25)",
            transition: "all 0.2s ease"
          }}
        >
          {isLoading ? "Authenticating..." : "Click to Login"}
        </button>
      </div>

      {/* Loading & Status Feedback Display */}
      {message && (
        <div
          style={{
            marginTop: "20px",
            padding: "10px 16px",
            borderRadius: "6px",
            display: "inline-block",
            maxWidth: "460px",
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

export default Login;
