import { useState, useRef, useEffect } from "react";
import "./Login.css";

const LOGIN_STAGES = [
  { id: 1, label: "Initializing Camera & Sensor" },
  { id: 2, label: "Anti-Spoofing Liveness Evaluation" },
  { id: 3, label: "Biometric Authenticity Confirmed" },
  { id: 4, label: "MediaPipe BlazeFace Landmark Alignment" },
  { id: 5, label: "FaceNet 512-D Embedding Extraction" },
  { id: 6, label: "FAISS 1:N Vector Identification" },
  { id: 7, label: "Access Authorization Confirmed" },
];

function Login() {
  const [userId, setUserId] = useState("");
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
        } catch {
          // ignore track stop error
        }
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
    setMessage("Step 1/7: Initializing Camera & requesting secure optical stream...");

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

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
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
      setMessage("Camera active. Align face directly within the reticle...");

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
      await delay(500);

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
      if (userId.trim()) {
        startFormData.append("user_id", userId.trim());
      }

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
      setMessage("Step 2/7: Checking Face Liveness with MiniFASNet anti-spoofing engine...");

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
          setMessage(`Step 2/7: Position face in reticle (${attempt}/${maxLivenessAttempts}). Searching for subject...`);
          await delay(350);
          continue;
        }

        if (verifyResult.status === "fake" || (!verifyResult.is_real && verifyResult.label === "Fake")) {
          // Spoof detected: reject attempt immediately. Do NOT capture or process any face!
          setStatusType("error");
          setCurrentStage(2);
          setMessage("Liveness verification rejected: Synthetic or replay artifact detected. Real person required.");
          setLivenessData(verifyResult);
          return;
        }

        if (verifyResult.verified || (verifyResult.is_real && verifyResult.score >= (verifyResult.threshold || 0.85))) {
          isVerified = true;
          verifiedResult = verifyResult;
          break;
        }

        setMessage(`Step 2/7: Analyzing live face stability (${attempt}/${maxLivenessAttempts}). Look directly at camera...`);
        await delay(300);
      }

      if (!isVerified) {
        setStatusType("error");
        setCurrentStage(2);
        setMessage("Liveness verification timed out without confirming a live person. Login rejected.");
        return;
      }

      // Step 3: Real Face Verified (Zero percentage leakage)
      setLivenessData(verifiedResult);
      setCurrentStage(3);
      setMessage("Step 3/7: Real Face Verified! Biometric authenticity confirmed.");
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
      setMessage("Step 4/7: Detecting facial landmarks with MediaPipe BlazeFace...");

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
      setMessage("Step 5/7: Extracting 512-D L2-normalized FaceNet probe embedding...");
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
        setMessage("Camera permission was denied. Please allow camera permissions in your browser to proceed.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        setMessage("No video capture hardware detected on this workstation.");
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        setMessage("Camera device is locked by another process or unavailable.");
      } else if (err.name === "OverconstrainedError") {
        setMessage("Camera hardware cannot support the requested video resolution.");
      } else if (err.message && err.message.includes("fetch")) {
        setMessage("Backend connection refused. Ensure the OJIX FastAPI server is active.");
      } else {
        setMessage(err.message || "An unexpected error occurred during biometric login.");
      }
    } finally {
      stopCamera();
      setIsLoading(false);
      isProcessingRef.current = false;
    }
  };

  // Viewfinder dynamic status indicator
  const getViewfinderStatus = () => {
    if (!cameraOn) {
      return { text: "Sensor Standby", dotClass: "status-dot" };
    }
    if (currentStage === 3 || statusType === "success") {
      return { text: "Subject Verified", dotClass: "status-dot status-dot-verified" };
    }
    if (statusType === "error") {
      return { text: "Verification Rejected", dotClass: "status-dot status-dot-denied" };
    }
    if (currentStage >= 1 && currentStage < 3) {
      return { text: "Liveness Engine Active", dotClass: "status-dot status-dot-scanning" };
    }
    return { text: "Live Video Feed", dotClass: "status-dot status-dot-cyan" };
  };

  const vStatus = getViewfinderStatus();

  // ==============================================================================
  // PART 2: Access Granted Verification Certificate
  // ==============================================================================
  if (authenticatedUser) {
    const displayName = authenticatedProfile?.full_name || authenticatedUser;

    return (
      <div className="access-certificate-container">
        <div className="cyber-certificate-card">
          {/* Certificate Header */}
          <div className="cert-header">
            <div className="cert-header-left">
              <div className="cert-emblem">🛡️</div>
              <div className="cert-title-group">
                <h3>Identity Verification Certificate</h3>
                <p>Biometric Authentication Successful &bull; Session Active</p>
              </div>
            </div>
            <span className="badge badge-verified badge-mono">
              ● IDENTITY MATCH CONFIRMED
            </span>
          </div>

          {/* Authenticated User Banner */}
          <div className="cert-user-banner">
            <div>
              <div className="cert-subject-label">
                Authenticated Subject
              </div>
              <div className="cert-user-name">{displayName}</div>
            </div>
            <div className="cert-user-id-chip">
              ID: {authenticatedProfile?.user_id || authenticatedUser}
            </div>
          </div>

          {/* Profile Metadata Grid */}
          <div className="cert-metadata-grid">
            <div className="cert-meta-item">
              <span className="cert-meta-label">Full Name</span>
              <span className="cert-meta-value">{authenticatedProfile?.full_name || "Verified Subject"}</span>
            </div>

            <div className="cert-meta-item">
              <span className="cert-meta-label">Work Email</span>
              <span className="cert-meta-value mono">{authenticatedProfile?.email || "Not Provided"}</span>
            </div>

            <div className="cert-meta-item">
              <span className="cert-meta-label">Role / Designation</span>
              <span className="cert-meta-value">{authenticatedProfile?.role || "Authorized Personnel"}</span>
            </div>

            <div className="cert-meta-item">
              <span className="cert-meta-label">Contact Phone</span>
              <span className="cert-meta-value mono">{authenticatedProfile?.phone_number || "Not Provided"}</span>
            </div>
          </div>

          {/* Biometric Processing Details */}
          <div className="cert-audit-rail">
            <div className="cert-audit-row">
              <span className="cert-audit-label">Liveness Verification:</span>
              <span className="cert-audit-status">
                <span className="status-dot status-dot-verified"></span>
                MiniFASNet Real Subject Confirmed
              </span>
            </div>
            <div className="cert-audit-row">
              <span className="cert-audit-label">Biometric Search:</span>
              <span className="cert-audit-status">
                <span className="status-dot status-dot-cyan"></span>
                FAISS 1:N Exact Cosine Match
              </span>
            </div>
            <div className="cert-audit-row">
              <span className="cert-audit-label">Vector Pipeline:</span>
              <span className="cert-audit-status" style={{ color: "var(--text-secondary)" }}>
                MediaPipe BlazeFace &rarr; FaceNet 512-D
              </span>
            </div>
            <div className="cert-audit-row">
              <span className="cert-audit-label">Storage Standard:</span>
              <span className="cert-audit-status" style={{ color: "var(--accent-blue)" }}>
                RAM-Only In-Memory Probe &bull; Zero Disk Storage
              </span>
            </div>
          </div>

          {/* Certificate Footer Action */}
          <div className="cert-footer-action">
            <button
              onClick={() => {
                setAuthenticatedUser(null);
                setAuthenticatedProfile(null);
                setStatusType("idle");
                setCurrentStage(0);
                setMessage("");
                setUserId("");
              }}
              className="btn btn-secondary btn-cert-logout"
            >
              <span>Terminate Session &bull; Authenticate Another User</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ==============================================================================
  // PART 1: Two-Column Face Login Interface (Default View)
  // ==============================================================================
  return (
    <div className="login-container">
      {/* Two-Column Responsive Layout */}
      <div className="login-two-col">
        {/* Left Column: Login Form & Stepper */}
        <div className="login-left-panel">
          <div className="login-panel-header">
            <div className="login-heading-group">
              <h2 className="login-main-heading">USER AUTHENTICATION</h2>
              <p className="login-sub-heading">
                Pure 1:N Face Identification &bull; Zero Password
              </p>
            </div>
            <span className="badge badge-cyber badge-mono">1:N FAISS</span>
          </div>

          {/* Form Fields Area */}
          <div className="login-form-fields">
            <div className="field-group">
              <label className="cyber-label">
                <span>User ID</span>
                <span className="field-hint">Optional for 1:N Facial Match</span>
              </label>
              <input
                type="text"
                className="cyber-input cyber-input-mono"
                placeholder="Enter User ID (Optional)"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                disabled={isLoading}
                autoComplete="off"
              />
              <span className="login-field-helper">
                Pure 1:N facial search automatically identifies enrolled subjects across the FAISS vector index.
              </span>
            </div>

            {/* Primary Action Button: FACE LOGIN */}
            <button
              type="button"
              onClick={handleLoginClick}
              disabled={isLoading}
              className="btn btn-primary btn-face-login"
            >
              {isLoading ? (
                <>
                  <span className="status-dot status-dot-scanning"></span>
                  <span>Scanning Face &amp; Authenticating...</span>
                </>
              ) : (
                <span>FACE LOGIN</span>
              )}
            </button>
          </div>

          {/* Feedback & Alert Box */}
          {message && (
            <div
              className={`cyber-feedback-box ${
                statusType === "error"
                  ? "error"
                  : statusType === "success"
                  ? "success"
                  : "loading"
              }`}
            >
              <div className="feedback-icon">
                {statusType === "error" && "⚠️"}
                {statusType === "success" && "✓"}
                {statusType === "loading" && "●"}
              </div>
              <div className="feedback-content">
                <div>{message}</div>
                {livenessData && (
                  <div className="feedback-meta">
                    Liveness Engine: {livenessData.label || livenessData.status} (Verified)
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 7-Step Technical Pipeline Stepper */}
          <div className="login-stepper-panel">
            <div className="login-stepper-header">
              <span className="login-stepper-title">Verification Pipeline</span>
              <span className="badge badge-cyber badge-mono">
                {currentStage > 0 ? `STAGE ${currentStage} OF 7` : "STANDBY"}
              </span>
            </div>

            <div className="login-stepper-list">
              {LOGIN_STAGES.map((st) => {
                const isPast = currentStage > st.id;
                const isCurrent = currentStage === st.id;
                const isFailed = isCurrent && statusType === "error";

                let stateClass = "pending";
                let indicatorText = `${st.id}`;

                if (isPast || (isCurrent && statusType === "success")) {
                  stateClass = "completed";
                  indicatorText = "✓";
                } else if (isFailed) {
                  stateClass = "failed";
                  indicatorText = "✕";
                } else if (isCurrent) {
                  stateClass = "active";
                  indicatorText = "●";
                }

                return (
                  <div key={st.id} className={`login-stepper-item ${stateClass}`}>
                    <span className="login-step-indicator">{indicatorText}</span>
                    <span className="login-step-label">{st.label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Face Scanner / Viewfinder */}
        <div className="login-right-panel">
          <div
            className={`login-viewfinder ${cameraOn ? "camera-active" : ""} ${
              currentStage === 3 || statusType === "success" ? "liveness-verified" : ""
            } ${statusType === "error" ? "liveness-failed" : ""}`}
          >
            {/* Viewfinder Status Overlay */}
            <div className="login-status-bar">
              <span className={vStatus.dotClass}></span>
              <span>{vStatus.text}</span>
            </div>

            {/* Corner Reticle Brackets */}
            <div className="login-reticle-corner top-left"></div>
            <div className="login-reticle-corner top-right"></div>
            <div className="login-reticle-corner bottom-left"></div>
            <div className="login-reticle-corner bottom-right"></div>

            {/* Centered Face Alignment Target */}
            {cameraOn && (
              <div className="login-alignment-oval">
                <span>Align Face</span>
              </div>
            )}

            {/* Active Webcam Video Stream */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="login-video-stream"
              style={{ display: cameraOn ? "block" : "none" }}
            />

            {/* Idle Placeholder when camera is inactive */}
            {!cameraOn && (
              <div className="login-idle-state">
                <div className="login-idle-icon">🛡️</div>
                <div className="login-idle-title">Biometric Scanner Standby</div>
                <div className="login-idle-subtitle">
                  Optical sensor is offline. Click <strong>FACE LOGIN</strong> to activate camera stream and start liveness verification.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;
