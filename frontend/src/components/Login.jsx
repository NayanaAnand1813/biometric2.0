import { useState, useRef, useEffect } from "react";

const LOGIN_STAGES = [
  { id: 1, label: "Initializing Camera" },
  { id: 2, label: "Checking Face Liveness" },
  { id: 3, label: "Real Face Verified" },
  { id: 4, label: "Detecting Face with MediaPipe" },
  { id: 5, label: "Generating Face Embedding" },
  { id: 6, label: "Verifying Identity in FAISS" },
  { id: 7, label: "Access Status" },
];

function Login() {
  const [message, setMessage] = useState("");
  const [statusType, setStatusType] = useState("idle"); // idle | loading | success | error
  const [currentStage, setCurrentStage] = useState(0); // 0 to 7
  const [cameraOn, setCameraOn] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [authenticatedUser, setAuthenticatedUser] = useState(null);
  const [authenticatedProfile, setAuthenticatedProfile] = useState(null);
  const [livenessData, setLivenessData] = useState(null);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const isProcessingRef = useRef(false);

  // Stop camera helper
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

  // Automated Camera-Only Liveness-First Face Login workflow
  const handleLoginClick = async () => {
    if (isLoading || isProcessingRef.current) return;
    isProcessingRef.current = true;

    // Reset previous feedback
    setStatusType("loading");
    setIsLoading(true);
    setAuthenticatedUser(null);
    setAuthenticatedProfile(null);
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

    // Stop any existing stream first
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
            setMessage("Camera connection was lost or disconnected. Login attempt terminated.");
          }
        });
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraOn(true);
      setMessage("Camera active. Aligning face in frame...");

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

      // Stabilization pause
      await delay(600);

      if (!video || !video.videoWidth || !video.videoHeight) {
        throw new Error("Unable to read video frame from camera.");
      }

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");

      // Initiate backend liveness session for 1:N login
      const startFormData = new FormData();
      startFormData.append("action", "login");

      const sessionStartResp = await fetch("http://127.0.0.1:8000/api/liveness/session/start", {
        method: "POST",
        body: startFormData
      });

      const sessionStartData = await sessionStartResp.json();
      if (!sessionStartResp.ok || !sessionStartData.session?.session_id) {
        throw new Error(sessionStartData.detail?.message || "Failed to start backend liveness session.");
      }

      const sessionId = sessionStartData.session.session_id;

      // Step 2: Checking Face Liveness on active live webcam stream
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
        verifyFormData.append("image", frameBlob, "live_login_frame.jpg");

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
          // Spoof detected: reject attempt immediately. Do NOT capture or process any face!
          setStatusType("error");
          setCurrentStage(2);
          setMessage("Liveness verification failed: Fake or spoof face detected. Real face required. Login rejected.");
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
        setMessage("Liveness verification timed out without confirming a live, real face. Login rejected.");
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
      setMessage("Step 4/7: Detecting Face with MediaPipe BlazeFace in memory...");

      // Send to /login with verified session_id (Pure 1:N Identification)
      const loginFormData = new FormData();
      loginFormData.append("session_id", sessionId);
      loginFormData.append("image", verifiedFaceBlob, "webcam_verified_login.jpg");

      const loginResp = await fetch("http://127.0.0.1:8000/login", {
        method: "POST",
        body: loginFormData
      });

      const loginResult = await loginResp.json();

      if (!loginResp.ok || !loginResult.authenticated) {
        setAuthenticatedUser(null);
        setAuthenticatedProfile(null);
        setStatusType("error");
        setCurrentStage(7);
        const errMsg =
          loginResult.detail?.message ||
          loginResult.detail?.error ||
          loginResult.message ||
          "Access Denied: Biometric verification failed. Face does not match registered user.";
        setMessage(errMsg);
        return;
      }

      // Step 5: Generating Face Embedding
      setCurrentStage(5);
      setMessage("Step 5/7: Generating 512-D FaceNet Probe Embedding...");
      await delay(350);

      // Step 6: Verifying Identity in FAISS
      setCurrentStage(6);
      setMessage("Step 6/7: Querying FAISS 1:N Exact Cosine Similarity Index...");
      await delay(350);

      // Step 7: Access Granted
      setCurrentStage(7);
      setAuthenticatedUser(loginResult.user_id);
      setAuthenticatedProfile(loginResult.user_profile || null);
      setStatusType("success");
      setMessage(loginResult.message || `Access Granted: Welcome back, ${loginResult.user_id}!`);
    } catch (err) {
      setAuthenticatedUser(null);
      setAuthenticatedProfile(null);
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
        setMessage(err.message || "An unexpected error occurred during login.");
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

  // Render Access Granted / Welcome screen if authenticated
  if (authenticatedUser) {
    const displayName = authenticatedProfile?.full_name || authenticatedUser;
    const livenessScore = livenessData?.score
      ? (livenessData.score * 100).toFixed(1)
      : "99.9";

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
            <div style={{ marginBottom: "10px", borderBottom: "1px solid #e5e7eb", paddingBottom: "8px" }}>
              <span style={{ fontWeight: "600", color: "#15803d" }}>Liveness Verification: </span>
              <span style={{ color: "#16a34a", fontWeight: "bold" }}>● Real Face Verified (MiniFASNet Confidence: {livenessScore}%)</span>
            </div>

            <div style={{ marginBottom: "10px", borderBottom: "1px solid #e5e7eb", paddingBottom: "8px" }}>
              <span style={{ fontWeight: "600", color: "#15803d" }}>Biometric Verification: </span>
              <span style={{ color: "#16a34a", fontWeight: "bold" }}>● FAISS 1:N Exact Cosine Match</span>
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
              Pipeline: MiniFASNet Liveness → MediaPipe BlazeFace → FaceNet 512-D → FAISS Exact Match
            </div>
          </div>

          <button
            onClick={() => {
              setAuthenticatedUser(null);
              setAuthenticatedProfile(null);
              setStatusType("idle");
              setCurrentStage(0);
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
      <h2>Biometric Login with Live Camera Liveness Verification</h2>
      <p style={{ color: "#666", marginBottom: "20px" }}>
        Look directly at the camera. The system verifies face liveness in real-time and identifies you automatically through 1:N facial search.
      </p>

      {/* Live Camera Video Viewport (No Upload Option) */}
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

      {/* Primary Action Button: Click to Login */}
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
          {isLoading ? "Verifying Liveness & Authenticating..." : "Click to Login"}
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
            {LOGIN_STAGES.map((st) => {
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

      {/* Loading & Status Feedback Display */}
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

export default Login;
