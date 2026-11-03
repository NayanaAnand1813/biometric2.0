import { useState, useRef, useEffect } from "react";

function Login() {
  const [userId, setUserId] = useState("");
  const [message, setMessage] = useState("");
  const [statusType, setStatusType] = useState("idle"); // idle | loading | success | error
  const [cameraOn, setCameraOn] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const videoRef = useRef(null);
  const streamRef = useRef(null);

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

  // Start webcam with detailed error handling
  const startCamera = async () => {
    setStatusType("idle");
    setMessage("");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatusType("error");
      setMessage("Camera access is not supported by your browser or environment.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
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
      setStatusType("idle");
      setMessage("Camera started. Look at the camera and click Capture & Login.");
    } catch (err) {
      setCameraOn(false);
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
      } else {
        setMessage(`Unable to access camera: ${err.message || "Unknown error"}`);
      }
    }
  };

  // Capture face image via canvas and convert to JPEG
  const captureAndLogin = () => {
    const cleanId = userId.trim();
    if (!cleanId) {
      setStatusType("error");
      setMessage("Please enter your registered User ID before capturing.");
      return;
    }

    const video = videoRef.current;
    if (!video || !cameraOn || !video.videoWidth) {
      setStatusType("error");
      setMessage("Camera is not ready. Please start the camera first.");
      return;
    }

    // Create canvas matching video frame resolution
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas frame to JPEG blob
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setStatusType("error");
          setMessage("Failed to capture image frame from camera.");
          return;
        }
        sendLoginRequest(cleanId, blob);
      },
      "image/jpeg",
      0.92
    );
  };

  // Send User ID and JPEG image to FastAPI /login endpoint
  const sendLoginRequest = async (cleanId, imageBlob) => {
    setIsLoading(true);
    setStatusType("loading");
    setMessage("Capturing face and sending login request...");

    const formData = new FormData();
    formData.append("user_id", cleanId);
    formData.append("image", imageBlob, "login_capture.jpg");

    try {
      const response = await fetch("http://127.0.0.1:8000/login", {
        method: "POST",
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        setStatusType("success");
        setMessage(result.message || "Login image received successfully.");
      } else {
        setStatusType("error");
        const errMsg = result.detail?.message || result.detail || result.error || "Failed to process login image.";
        setMessage(errMsg);
      }
    } catch (netErr) {
      setStatusType("error");
      setMessage("Failed to connect to backend server. Please verify FastAPI backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = () => {
    if (statusType === "error") return "#dc2626";
    if (statusType === "success") return "#16a34a";
    if (statusType === "loading") return "#2563eb";
    return "#333";
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h2>User Login</h2>
      <p style={{ color: "#666", marginBottom: "15px" }}>
        Enter your registered User ID and capture your face to authenticate
      </p>

      {/* User ID Input */}
      <div>
        <input
          type="text"
          placeholder="Enter Registered User ID"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          disabled={isLoading}
          style={{
            padding: "8px 14px",
            fontSize: "16px",
            borderRadius: "6px",
            border: "1px solid #ccc",
            width: "260px"
          }}
        />
      </div>

      <br />

      {/* Camera Video Stream */}
      <div>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          width="400"
          style={{
            background: "#222",
            borderRadius: "8px",
            boxShadow: "0 4px 10px rgba(0,0,0,0.15)",
            border: cameraOn ? "2px solid #16a34a" : "2px solid #555"
          }}
        />
      </div>

      <br />

      {/* Camera & Action Buttons */}
      <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
        <button
          onClick={startCamera}
          disabled={cameraOn || isLoading}
          style={{
            padding: "8px 16px",
            fontSize: "15px",
            cursor: cameraOn || isLoading ? "not-allowed" : "pointer",
            borderRadius: "6px",
            backgroundColor: "#2563eb",
            color: "#fff",
            border: "none"
          }}
        >
          Start Camera
        </button>

        <button
          onClick={captureAndLogin}
          disabled={!cameraOn || isLoading}
          style={{
            padding: "8px 16px",
            fontSize: "15px",
            cursor: !cameraOn || isLoading ? "not-allowed" : "pointer",
            borderRadius: "6px",
            backgroundColor: isLoading ? "#9ca3af" : "#16a34a",
            color: "#fff",
            border: "none",
            fontWeight: "bold"
          }}
        >
          {isLoading ? "Sending..." : "Capture & Login"}
        </button>

        <button
          onClick={() => {
            stopCamera();
            setMessage("Camera stopped.");
            setStatusType("idle");
          }}
          disabled={!cameraOn || isLoading}
          style={{
            padding: "8px 16px",
            fontSize: "15px",
            cursor: !cameraOn || isLoading ? "not-allowed" : "pointer",
            borderRadius: "6px",
            backgroundColor: "#ef4444",
            color: "#fff",
            border: "none"
          }}
        >
          Stop Camera
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

