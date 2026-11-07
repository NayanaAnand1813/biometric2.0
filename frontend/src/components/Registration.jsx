import { useState, useRef, useEffect } from "react";
import "./Registration.css";

const REGISTRATION_STAGES = [
  { id: 1, label: "Initializing Camera & Sensor" },
  { id: 2, label: "Anti-Spoofing Liveness Evaluation" },
  { id: 3, label: "Biometric Authenticity Confirmed" },
  { id: 4, label: "MediaPipe BlazeFace Landmark Alignment" },
  { id: 5, label: "FaceNet 512-D Embedding Extraction" },
  { id: 6, label: "PostgreSQL & FAISS Vector Commitment" },
  { id: 7, label: "Identity Enrollment Complete" },
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
    setMessage("Step 1/7: Initializing Camera & requesting secure video stream...");

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
      setMessage("Camera active. Please center your face directly within the reticle...");

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
        verifyFormData.append("image", frameBlob, "live_frame.jpg");

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
          // Liveness detection failed: reject attempt immediately. Do NOT capture or register face image.
          setStatusType("error");
          setCurrentStage(2);
          setMessage("Liveness verification rejected: Synthetic or replay artifact detected. Please face the camera directly.");
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
        setMessage("Liveness verification timed out without confirming a live person. Registration rejected.");
        return;
      }

      // Step 3: Real Face Verified (No score percentage leaked)
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
      setMessage("Step 5/7: Extracting 512-D L2-normalized FaceNet biometric vector...");
      await delay(350);

      // Step 6: Processing Registration
      setCurrentStage(6);
      setMessage("Step 6/7: Storing profile record in PostgreSQL & synchronizing FAISS IndexFlatIP...");
      await delay(350);

      // Step 7: Completed
      setCurrentStage(7);
      setStatusType("success");
      setMessage(regResult.message || `Identity verified and enrolled: ${cleanFullName} (${cleanId}).`);
    } catch (err) {
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
        setMessage(err.message || "An unexpected error occurred during biometric registration.");
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
      return { text: "Sensor Idle", dotClass: "status-dot" };
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

  return (
    <div className="registration-container">
      {/* Header Description */}
      <div className="registration-header">
        <h2>
          <span>Biometric Identity Registration</span>
          <span className="badge badge-cyber badge-mono">ENROLLMENT PORTAL</span>
        </h2>
        <p>
          Complete user credentials and initiate live biometric verification. Liveness detection is enforced prior to embedding extraction.
        </p>
      </div>

      {/* Two-Column Responsive Grid */}
      <div className="registration-grid">
        {/* Left Column: Profile Form */}
        <div className="reg-form-panel">
          <div className="reg-form-title">
            <span>Identity Credentials</span>
          </div>

          <div className="reg-form-fields">
            <div className="field-group">
              <label className="cyber-label">
                <span>User ID <span className="required-star">*</span></span>
                <span className="field-hint">Unique alphanumeric</span>
              </label>
              <input
                type="text"
                className="cyber-input cyber-input-mono"
                placeholder="e.g. john_doe"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                disabled={isLoading}
                autoComplete="off"
              />
            </div>

            <div className="field-group">
              <label className="cyber-label">
                <span>Full Name <span className="required-star">*</span></span>
                <span className="field-hint">First and last name</span>
              </label>
              <input
                type="text"
                className="cyber-input"
                placeholder="e.g. John Doe"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <div className="field-group">
              <label className="cyber-label">
                <span>Work Email <span className="required-star">*</span></span>
                <span className="field-hint">Corporate domain</span>
              </label>
              <input
                type="email"
                className="cyber-input"
                placeholder="e.g. john.doe@company.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <div className="field-group">
              <label className="cyber-label">
                <span>Role / Designation <span className="required-star">*</span></span>
                <span className="field-hint">Access tier</span>
              </label>
              <input
                type="text"
                className="cyber-input"
                placeholder="e.g. Senior Security Architect"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <div className="field-group">
              <label className="cyber-label">
                <span>Phone Number <span className="required-star">*</span></span>
                <span className="field-hint">Direct contact</span>
              </label>
              <input
                type="tel"
                className="cyber-input cyber-input-mono"
                placeholder="e.g. +1 555-0123"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                disabled={isLoading}
              />
            </div>
          </div>

          <button
            onClick={handleRegisterClick}
            disabled={isLoading}
            className="btn btn-primary btn-register-submit"
          >
            {isLoading ? (
              <>
                <span className="status-dot status-dot-scanning"></span>
                <span>Enrolling Biometric Profile...</span>
              </>
            ) : (
              <span>Start Live Camera Enrollment</span>
            )}
          </button>
        </div>

        {/* Right Column: Viewfinder & Pipeline Stepper */}
        <div className="reg-viewfinder-panel">
          {/* Cyber Biometric Viewfinder */}
          <div
            className={`cyber-viewfinder ${cameraOn ? "camera-active" : ""} ${
              currentStage === 3 || statusType === "success" ? "liveness-verified" : ""
            } ${statusType === "error" ? "liveness-failed" : ""}`}
          >
            {/* Viewfinder Status Overlay */}
            <div className="viewfinder-status-bar">
              <span className={vStatus.dotClass}></span>
              <span>{vStatus.text}</span>
            </div>

            {/* Corner Reticle Brackets */}
            <div className="reticle-corner top-left"></div>
            <div className="reticle-corner top-right"></div>
            <div className="reticle-corner bottom-left"></div>
            <div className="reticle-corner bottom-right"></div>

            {/* Centered Face Alignment Target */}
            <div className="face-alignment-oval">
              <span>Align Face</span>
            </div>

            {/* Active Webcam Video Stream */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="cyber-video-stream"
              style={{ display: cameraOn ? "block" : "none" }}
            />

            {/* Idle Placeholder when camera is inactive */}
            {!cameraOn && (
              <div className="viewfinder-idle-state">
                <div className="viewfinder-idle-icon">📷</div>
                <div className="viewfinder-idle-text">Optical Sensor Standby</div>
                <div className="viewfinder-idle-subtext">
                  Camera will activate automatically upon initiating enrollment.
                </div>
              </div>
            )}
          </div>

          {/* 7-Stage Technical Pipeline Stepper */}
          {currentStage > 0 && (
            <div className="reg-stepper-panel">
              <div className="stepper-header">
                <span className="stepper-header-title">Enrollment Pipeline</span>
                <span className="badge badge-cyber badge-mono">
                  STAGE {currentStage} OF 7
                </span>
              </div>

              <div className="stepper-step-list">
                {REGISTRATION_STAGES.map((st) => {
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
                    <div key={st.id} className={`stepper-step-item ${stateClass}`}>
                      <span className="step-indicator">{indicatorText}</span>
                      <span className="step-label">{st.label}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

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
        </div>
      </div>
    </div>
  );
}

export default Registration;
