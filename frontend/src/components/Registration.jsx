import { useState, useRef, useEffect, useCallback } from "react";
import "./Registration.css";

// 8 Workflow States
const WORKFLOW_STATES = {
  DETAILS_REQUIRED: "DETAILS REQUIRED",
  CAMERA_READY: "CAMERA READY",
  CHECKING_LIVENESS: "CHECKING LIVENESS",
  LIVE_FACE_DETECTED: "LIVE FACE DETECTED",
  FACE_CAPTURED: "FACE CAPTURED",
  AWAITING_CONFIRMATION: "AWAITING CONFIRMATION",
  REGISTERING: "REGISTERING",
  REGISTRATION_SUCCESSFUL: "REGISTRATION SUCCESSFUL",
};

const PIPELINE_STAGES = [
  { id: 1, label: "Stage A: Identity Details Entry", states: [WORKFLOW_STATES.DETAILS_REQUIRED] },
  { id: 2, label: "Stage B: Live Anti-Spoofing & Capture", states: [WORKFLOW_STATES.CAMERA_READY, WORKFLOW_STATES.CHECKING_LIVENESS, WORKFLOW_STATES.LIVE_FACE_DETECTED, WORKFLOW_STATES.FACE_CAPTURED] },
  { id: 3, label: "Stage C: Review & Confirm Capture", states: [WORKFLOW_STATES.AWAITING_CONFIRMATION] },
  { id: 4, label: "Stage D: Register, Embed & Index", states: [WORKFLOW_STATES.REGISTERING, WORKFLOW_STATES.REGISTRATION_SUCCESSFUL] },
];

function Registration() {
  // Identity Credential Form Fields
  const [fullName, setFullName] = useState("");
  const [userId, setUserId] = useState("");
  const [email, setEmail] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [role, setRole] = useState("");

  // Workflow State Machine
  const [workflowState, setWorkflowState] = useState(WORKFLOW_STATES.DETAILS_REQUIRED);
  const [message, setMessage] = useState("Enter user identity details and click ADD FACE to initialize optical sensor.");
  const [statusType, setStatusType] = useState("idle"); // idle | loading | success | error

  // Session & Temporary Face Storage (RAM-only)
  const [sessionId, setSessionId] = useState(null);
  const [capturedPreviewUrl, setCapturedPreviewUrl] = useState(null);
  const [isFaceConfirmed, setIsFaceConfirmed] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const isProcessingRef = useRef(false);

  // Stop active webcam and clean tracks
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

    setCameraActive(false);
  };

  // Cleanup camera stream and temporary blob URL on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (capturedPreviewUrl) {
        URL.revokeObjectURL(capturedPreviewUrl);
      }
    };
  }, [capturedPreviewUrl]);

  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // Field validation helper (Stage A)
  const validateFormDetails = useCallback(() => {
    const cleanId = userId.trim();
    const cleanFullName = fullName.trim();
    const cleanEmail = email.trim();
    const cleanRole = role.trim();
    const cleanPhone = phoneNumber.trim();

    if (!cleanId || !cleanFullName || !cleanEmail || !cleanRole || !cleanPhone) {
      return {
        valid: false,
        error: "All five fields (Full Name, User ID, Work Email, Phone Number, Role) are mandatory."
      };
    }

    const normalizedEmail = cleanEmail.toLowerCase();
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(normalizedEmail)) {
      return {
        valid: false,
        error: "Please enter a valid work email address (e.g. name@company.com)."
      };
    }

    const phoneDigits = cleanPhone.replace(/\D/g, "");
    const phoneRegex = /^\+?[0-9\s\-()]{7,20}$/;
    if (!phoneRegex.test(cleanPhone) || phoneDigits.length < 7) {
      return {
        valid: false,
        error: "Please enter a valid phone number (at least 7 digits, e.g. +1 555-0123)."
      };
    }

    return {
      valid: true,
      data: {
        cleanId,
        cleanFullName,
        normalizedEmail,
        cleanRole,
        cleanPhone,
      }
    };
  }, [userId, fullName, email, role, phoneNumber]);

  const isFormValid = validateFormDetails().valid;

  // ---------------------------------------------------------------------------
  // Stage B: ADD FACE — Initialize Camera, Check Liveness, Store in RAM
  // ---------------------------------------------------------------------------
  const handleAddFace = async () => {
    if (isLoading || isProcessingRef.current) return;

    const validation = validateFormDetails();
    if (!validation.valid) {
      setStatusType("error");
      setMessage(validation.error);
      return;
    }

    const validDetails = validation.data;

    // Reset previous capture states
    if (capturedPreviewUrl) {
      URL.revokeObjectURL(capturedPreviewUrl);
      setCapturedPreviewUrl(null);
    }
    setIsFaceConfirmed(false);

    isProcessingRef.current = true;
    setIsLoading(true);
    setStatusType("loading");
    setWorkflowState(WORKFLOW_STATES.CAMERA_READY);
    setMessage("Initializing optical sensor & requesting secure webcam stream...");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatusType("error");
      setWorkflowState(WORKFLOW_STATES.DETAILS_REQUIRED);
      setMessage("Webcam access is not supported by your browser or environment. Camera is strictly required.");
      setIsLoading(false);
      isProcessingRef.current = false;
      return;
    }

    stopCamera();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
      });

      stream.getVideoTracks().forEach((track) => {
        track.addEventListener("ended", () => {
          if (isProcessingRef.current) {
            stopCamera();
            setIsLoading(false);
            isProcessingRef.current = false;
            setStatusType("error");
            setWorkflowState(WORKFLOW_STATES.DETAILS_REQUIRED);
            setMessage("Camera connection was interrupted or disconnected.");
          }
        });
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraActive(true);

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

      await delay(500);

      if (!video || !video.videoWidth || !video.videoHeight) {
        throw new Error("Unable to read video frame from camera sensor.");
      }

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");

      // Step B.1: Backend-Issued Registration Liveness Session (TTL 5 minutes)
      const startFormData = new FormData();
      startFormData.append("action", "register");
      startFormData.append("user_id", validDetails.cleanId);

      const sessionStartResp = await fetch("http://127.0.0.1:8000/api/liveness/session/start", {
        method: "POST",
        body: startFormData,
      });

      const sessionStartData = await sessionStartResp.json();
      if (!sessionStartResp.ok || !sessionStartData.session?.session_id) {
        throw new Error(sessionStartData.detail?.message || "Failed to initiate backend liveness session.");
      }

      const activeSessionId = sessionStartData.session.session_id;
      setSessionId(activeSessionId);

      // Step B.2: Evaluate Live Anti-Spoofing Frames with MiniFASNet
      setWorkflowState(WORKFLOW_STATES.CHECKING_LIVENESS);
      setMessage("CHECKING LIVENESS: MiniFASNet anti-spoofing engine evaluating facial stream...");

      let isVerified = false;
      const maxLivenessAttempts = 15;

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
        verifyFormData.append("session_id", activeSessionId);
        verifyFormData.append("image", frameBlob, "live_frame.jpg");

        const verifyResp = await fetch("http://127.0.0.1:8000/api/liveness/session/verify-frame", {
          method: "POST",
          body: verifyFormData,
        });

        const verifyResult = await verifyResp.json();

        if (verifyResult.status === "no_face") {
          setMessage(`Position face inside reticle (${attempt}/${maxLivenessAttempts}). Searching for subject...`);
          await delay(350);
          continue;
        }

        if (verifyResult.status === "fake" || (!verifyResult.is_real && verifyResult.label === "Fake")) {
          // Reject spoofed, replayed, or invalid face captures immediately
          stopCamera();
          setStatusType("error");
          setWorkflowState(WORKFLOW_STATES.DETAILS_REQUIRED);
          setMessage("Liveness verification rejected: Synthetic or replay artifact detected. Real person required.");
          return;
        }

        if (verifyResult.verified || (verifyResult.is_real && verifyResult.score >= (verifyResult.threshold || 0.85))) {
          isVerified = true;
          break;
        }

        setMessage(`Analyzing live face stability (${attempt}/${maxLivenessAttempts}). Look directly at camera...`);
        await delay(300);
      }

      if (!isVerified) {
        stopCamera();
        setStatusType("error");
        setWorkflowState(WORKFLOW_STATES.DETAILS_REQUIRED);
        setMessage("Liveness verification timed out without confirming an authentic live face.");
        return;
      }

      // Step B.3: LIVE FACE DETECTED
      setWorkflowState(WORKFLOW_STATES.LIVE_FACE_DETECTED);
      setMessage("LIVE FACE DETECTED: Authentic live subject confirmed. Capturing biometric frame...");
      await delay(400);

      // Step B.4: Capture suitable face frame and store temporarily in backend RAM
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const capturedBlob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.95);
      });

      if (!capturedBlob) {
        throw new Error("Failed to capture verified face frame.");
      }

      const captureFormData = new FormData();
      captureFormData.append("session_id", activeSessionId);
      captureFormData.append("user_id", validDetails.cleanId);
      captureFormData.append("image", capturedBlob, "captured_face.jpg");

      const captureResp = await fetch("http://127.0.0.1:8000/api/registration/session/capture-frame", {
        method: "POST",
        body: captureFormData,
      });

      const captureResult = await captureResp.json();
      if (!captureResp.ok || captureResult.status !== "success") {
        throw new Error(captureResult.detail?.message || "Failed to store captured face frame in backend RAM.");
      }

      // Stop camera after capture
      stopCamera();

      // Display the captured face preview on the registration page (in-memory object URL)
      const previewUrl = URL.createObjectURL(capturedBlob);
      setCapturedPreviewUrl(previewUrl);

      // Transition to Stage C: AWAITING CONFIRMATION
      setWorkflowState(WORKFLOW_STATES.AWAITING_CONFIRMATION);
      setStatusType("idle");
      setMessage("FACE CAPTURED in temporary RAM. Review preview and click CONFIRM FACE to enable registration.");
    } catch (err) {
      stopCamera();
      setStatusType("error");
      setWorkflowState(WORKFLOW_STATES.DETAILS_REQUIRED);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setMessage("Camera permission was denied. Please allow camera permissions to proceed.");
      } else if (err.message && err.message.includes("fetch")) {
        setMessage("Backend connection refused. Ensure the OJIX FastAPI server is active.");
      } else {
        setMessage(err.message || "An unexpected error occurred during face capture.");
      }
    } finally {
      setIsLoading(false);
      isProcessingRef.current = false;
    }
  };

  // ---------------------------------------------------------------------------
  // Stage C: CONFIRM FACE — Accept captured frame and enable registration
  // ---------------------------------------------------------------------------
  const handleConfirmFace = async () => {
    if (!sessionId || isLoading) return;

    try {
      setIsLoading(true);
      setStatusType("loading");
      setMessage("Confirming biometric capture with backend session...");

      const confirmFormData = new FormData();
      confirmFormData.append("session_id", sessionId);
      confirmFormData.append("user_id", userId.trim());

      const resp = await fetch("http://127.0.0.1:8000/api/registration/session/confirm", {
        method: "POST",
        body: confirmFormData,
      });

      const result = await resp.json();
      if (!resp.ok || result.status !== "success") {
        throw new Error(result.detail?.message || "Failed to confirm face capture.");
      }

      stopCamera();
      setIsFaceConfirmed(true);
      setStatusType("success");
      setWorkflowState(WORKFLOW_STATES.AWAITING_CONFIRMATION);
      setMessage("FACE CAPTURE CONFIRMED: Ready to enroll user. Click REGISTER USER to complete enrollment.");
    } catch (err) {
      setStatusType("error");
      setMessage(err.message || "Confirmation failed.");
    } finally {
      setIsLoading(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Stage C: RETAKE — Discard temporary capture, restart camera and verify again
  // ---------------------------------------------------------------------------
  const handleRetake = async () => {
    if (isLoading) return;

    // Discard previous temporary capture
    if (capturedPreviewUrl) {
      URL.revokeObjectURL(capturedPreviewUrl);
      setCapturedPreviewUrl(null);
    }
    setIsFaceConfirmed(false);

    if (sessionId) {
      try {
        const retakeFormData = new FormData();
        retakeFormData.append("session_id", sessionId);
        retakeFormData.append("user_id", userId.trim());

        await fetch("http://127.0.0.1:8000/api/registration/session/retake", {
          method: "POST",
          body: retakeFormData,
        });
      } catch {
        // ignore network error on retake
      }
    }

    // Restart the camera and perform liveness verification again
    handleAddFace();
  };

  // ---------------------------------------------------------------------------
  // Stage D: REGISTER USER — Commit Profile, Extract Embedding & Sync FAISS
  // ---------------------------------------------------------------------------
  const handleRegisterUser = async () => {
    if (!isFaceConfirmed || !sessionId || isLoading) return;

    const validation = validateFormDetails();
    if (!validation.valid) {
      setStatusType("error");
      setMessage(validation.error);
      return;
    }

    const validDetails = validation.data;

    try {
      setIsLoading(true);
      setStatusType("loading");
      setWorkflowState(WORKFLOW_STATES.REGISTERING);
      setMessage("REGISTERING: BlazeFace landmark alignment, FaceNet 512-D embedding extraction, and atomic PostgreSQL commit...");

      const regFormData = new FormData();
      regFormData.append("session_id", sessionId);
      regFormData.append("user_id", validDetails.cleanId);
      regFormData.append("full_name", validDetails.cleanFullName);
      regFormData.append("email", validDetails.normalizedEmail);
      regFormData.append("role", validDetails.cleanRole);
      regFormData.append("phone_number", validDetails.cleanPhone);

      const resp = await fetch("http://127.0.0.1:8000/register", {
        method: "POST",
        body: regFormData,
      });

      const result = await resp.json();

      if (!resp.ok || result.status !== "success") {
        const errMsg =
          result.detail?.message ||
          result.detail?.error ||
          result.message ||
          "Registration failed.";
        throw new Error(errMsg);
      }

      setWorkflowState(WORKFLOW_STATES.REGISTRATION_SUCCESSFUL);
      setStatusType("success");

      let successMsg = `Identity enrolled successfully: ${validDetails.cleanFullName} (${validDetails.cleanId}).`;
      if (result.indexing_status === "reconciliation_pending" || result.faiss_synced === false) {
        successMsg += " (Note: FAISS index reconciliation is pending).";
      }
      setMessage(successMsg);
    } catch (err) {
      setStatusType("error");
      setWorkflowState(WORKFLOW_STATES.AWAITING_CONFIRMATION);
      setMessage(err.message || "Registration transaction failed.");
    } finally {
      setIsLoading(false);
      stopCamera();
    }
  };

  // Dynamic status bar indicator
  const getViewfinderStatusBar = () => {
    switch (workflowState) {
      case WORKFLOW_STATES.CAMERA_READY:
        return { text: "Sensor Initialized", dotClass: "status-dot status-dot-cyan" };
      case WORKFLOW_STATES.CHECKING_LIVENESS:
        return { text: "Checking Face Liveness", dotClass: "status-dot status-dot-scanning" };
      case WORKFLOW_STATES.LIVE_FACE_DETECTED:
        return { text: "Live Face Confirmed", dotClass: "status-dot status-dot-verified" };
      case WORKFLOW_STATES.FACE_CAPTURED:
      case WORKFLOW_STATES.AWAITING_CONFIRMATION:
        return {
          text: isFaceConfirmed ? "Face Capture Confirmed" : "Face Captured — Review Preview",
          dotClass: isFaceConfirmed ? "status-dot status-dot-verified" : "status-dot status-dot-scanning"
        };
      case WORKFLOW_STATES.REGISTERING:
        return { text: "Registering Profile & Embedding", dotClass: "status-dot status-dot-scanning" };
      case WORKFLOW_STATES.REGISTRATION_SUCCESSFUL:
        return { text: "Registration Successful", dotClass: "status-dot status-dot-verified" };
      default:
        return { text: "Sensor Standby", dotClass: "status-dot" };
    }
  };

  const vStatus = getViewfinderStatusBar();

  // Helper to determine active stage
  const getCurrentStageId = () => {
    if (workflowState === WORKFLOW_STATES.DETAILS_REQUIRED) return 1;
    if ([WORKFLOW_STATES.CAMERA_READY, WORKFLOW_STATES.CHECKING_LIVENESS, WORKFLOW_STATES.LIVE_FACE_DETECTED, WORKFLOW_STATES.FACE_CAPTURED].includes(workflowState)) return 2;
    if (workflowState === WORKFLOW_STATES.AWAITING_CONFIRMATION) return 3;
    if ([WORKFLOW_STATES.REGISTERING, WORKFLOW_STATES.REGISTRATION_SUCCESSFUL].includes(workflowState)) return 4;
    return 1;
  };

  const currentStageId = getCurrentStageId();

  return (
    <div className="registration-container">
      {/* Two-Column Responsive Layout */}
      <div className="registration-two-col">
        {/* Left Column: User Details & Dedicated ADD FACE Section */}
        <div className="reg-left-panel">
          <div className="reg-panel-header">
            <div className="reg-heading-group">
              <h2 className="reg-main-heading">USER REGISTRATION</h2>
              <p className="reg-sub-heading">
                Two-Stage Biometric Profile Enrollment
              </p>
            </div>
            <span className="workflow-badge badge-mono">{workflowState}</span>
          </div>

          {/* Form Fields: Full Name, User ID, Work Email, Phone Number, Role */}
          <div className="reg-form-fields">
            <div className="field-group">
              <label className="cyber-label">
                <span>Full Name <span className="required-star">*</span></span>
                <span className="field-hint">First and last name</span>
              </label>
              <input
                type="text"
                className="cyber-input"
                placeholder="e.g. Jacqueline Miller"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                disabled={isLoading || workflowState === WORKFLOW_STATES.REGISTRATION_SUCCESSFUL}
                autoComplete="name"
              />
            </div>

            <div className="field-group">
              <label className="cyber-label">
                <span>User ID <span className="required-star">*</span></span>
                <span className="field-hint">Unique alphanumeric</span>
              </label>
              <input
                type="text"
                className="cyber-input cyber-input-mono"
                placeholder="e.g. j_miller"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                disabled={isLoading || workflowState === WORKFLOW_STATES.REGISTRATION_SUCCESSFUL}
                autoComplete="off"
              />
            </div>

            <div className="field-group">
              <label className="cyber-label">
                <span>Work Email <span className="required-star">*</span></span>
                <span className="field-hint">Corporate address</span>
              </label>
              <input
                type="email"
                className="cyber-input"
                placeholder="e.g. j.miller@cybercorp.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading || workflowState === WORKFLOW_STATES.REGISTRATION_SUCCESSFUL}
                autoComplete="email"
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
                placeholder="e.g. +1 555-0199"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                disabled={isLoading || workflowState === WORKFLOW_STATES.REGISTRATION_SUCCESSFUL}
                autoComplete="tel"
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
                placeholder="e.g. Lead Cryptanalyst"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                disabled={isLoading || workflowState === WORKFLOW_STATES.REGISTRATION_SUCCESSFUL}
              />
            </div>
          </div>

          {/* Dedicated ADD FACE Section */}
          <div className="dedicated-add-face-section">
            <div className="add-face-header-bar">
              <span className="add-face-section-title">BIOMETRIC FACE CAPTURE</span>
              {isFaceConfirmed && (
                <div className="face-confirmed-pill">
                  <span className="confirmed-check-symbol">✓</span>
                  <span className="confirmed-status-text">FACE CAPTURE CONFIRMED</span>
                </div>
              )}
            </div>

            {/* Interaction States */}
            {!capturedPreviewUrl && !isFaceConfirmed && (
              <div className="add-face-unconfirmed-box">
                <button
                  type="button"
                  onClick={handleAddFace}
                  disabled={isLoading || cameraActive || workflowState === WORKFLOW_STATES.REGISTRATION_SUCCESSFUL}
                  className="btn btn-primary btn-add-face"
                >
                  {cameraActive ? (
                    <>
                      <span className="status-dot status-dot-scanning"></span>
                      <span>Sensor Active &bull; Verifying Liveness...</span>
                    </>
                  ) : (
                    <span>📷 ADD FACE</span>
                  )}
                </button>
                <div className="face-neutral-status">
                  <span className="status-dot"></span>
                  <span>Biometric Capture Required</span>
                </div>
              </div>
            )}

            {/* Stage C: Captured preview awaiting confirmation */}
            {capturedPreviewUrl && !isFaceConfirmed && (
              <div className="add-face-review-box">
                <button
                  type="button"
                  onClick={handleConfirmFace}
                  disabled={isLoading}
                  className="btn btn-primary btn-confirm-face"
                >
                  <span>CONFIRM FACE</span>
                </button>

                <button
                  type="button"
                  onClick={handleRetake}
                  disabled={isLoading}
                  className="btn btn-outline btn-retake"
                >
                  <span>🔄 RETAKE</span>
                </button>
              </div>
            )}

            {/* Face confirmed state actions */}
            {isFaceConfirmed && workflowState !== WORKFLOW_STATES.REGISTRATION_SUCCESSFUL && (
              <div className="add-face-confirmed-box">
                <button
                  type="button"
                  onClick={handleRetake}
                  disabled={isLoading}
                  className="btn btn-outline btn-retake-sm"
                >
                  <span>🔄 Retake Face Capture</span>
                </button>
              </div>
            )}
          </div>

          {/* Primary Action Button: REGISTER USER (Disabled until details & confirmed) */}
          <button
            type="button"
            onClick={handleRegisterUser}
            disabled={!isFaceConfirmed || !isFormValid || isLoading || workflowState === WORKFLOW_STATES.REGISTRATION_SUCCESSFUL}
            className={`btn btn-register-submit ${isFaceConfirmed && isFormValid ? "btn-primary btn-ready" : "btn-disabled"}`}
          >
            {isLoading && workflowState === WORKFLOW_STATES.REGISTERING ? (
              <>
                <span className="status-dot status-dot-scanning"></span>
                <span>Enrolling Profile &amp; Embedding...</span>
              </>
            ) : (
              <span>REGISTER USER</span>
            )}
          </button>

          {/* Form helper note if incomplete */}
          {(!isFaceConfirmed || !isFormValid) && workflowState !== WORKFLOW_STATES.REGISTRATION_SUCCESSFUL && (
            <span className="button-state-note">
              * Valid user details and confirmed biometric face capture are required to enable registration.
            </span>
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
                {statusType === "idle" && "ℹ️"}
              </div>
              <div className="feedback-content">
                <div>{message}</div>
                {sessionId && (
                  <div className="feedback-meta">
                    Session: {sessionId} (In-Memory RAM Storage)
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Camera Viewfinder & Pipeline Stepper */}
        <div className="reg-right-panel">
          {/* Cyber Biometric Viewfinder */}
          <div
            className={`cyber-viewfinder ${cameraActive ? "camera-active" : ""} ${
              isFaceConfirmed || workflowState === WORKFLOW_STATES.REGISTRATION_SUCCESSFUL ? "liveness-verified" : ""
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

            {/* Active Webcam Video Stream */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="cyber-video-stream"
              style={{ display: cameraActive ? "block" : "none" }}
            />

            {/* Centered Face Alignment Target */}
            {cameraActive && (
              <div className="face-alignment-oval">
                <span>Align Face</span>
              </div>
            )}

            {/* Captured Face Preview (In-memory Object URL) */}
            {capturedPreviewUrl && !cameraActive && (
              <div className="captured-preview-container">
                <img
                  src={capturedPreviewUrl}
                  alt="Captured biometric preview"
                  className="captured-face-preview"
                />
                <div className="preview-overlay-badge">
                  {isFaceConfirmed ? "CONFIRMED CAPTURE" : "TEMPORARY RAM CAPTURE"}
                </div>
              </div>
            )}

            {/* Idle Placeholder when camera is inactive and no preview */}
            {!cameraActive && !capturedPreviewUrl && (
              <div className="viewfinder-idle-state">
                <div className="viewfinder-idle-icon">📷</div>
                <div className="viewfinder-idle-text">Optical Sensor Standby</div>
                <div className="viewfinder-idle-subtext">
                  Complete user details and click <strong>ADD FACE</strong> to initialize camera stream.
                </div>
              </div>
            )}
          </div>

          {/* 4-Stage Architectural Pipeline Stepper */}
          <div className="reg-stepper-panel">
            <div className="stepper-header">
              <span className="stepper-header-title">Registration Workflow</span>
              <span className="badge badge-cyber badge-mono">
                STAGE {currentStageId} OF 4
              </span>
            </div>

            <div className="stepper-step-list">
              {PIPELINE_STAGES.map((st) => {
                const isPast = currentStageId > st.id;
                const isCurrent = currentStageId === st.id;
                const isFailed = isCurrent && statusType === "error";

                let stateClass = "pending";
                let indicatorText = `${st.id}`;

                if (isPast || (isCurrent && workflowState === WORKFLOW_STATES.REGISTRATION_SUCCESSFUL)) {
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
        </div>
      </div>
    </div>
  );
}

export default Registration;
