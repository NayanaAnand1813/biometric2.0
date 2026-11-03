import { useState, useRef } from "react";

function Registration() {
  const [userId, setUserId] = useState("");
  const [message, setMessage] = useState("");
  const [cameraOn, setCameraOn] = useState(false);

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // Start webcam
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraOn(true);
      setMessage("Camera started");
    } catch (error) {
      setMessage("Unable to access camera");
    }
  };

  // Capture image
  const captureImage = () => {
    const video = videoRef.current;

    if (!video || !video.videoWidth) {
      setMessage("Camera is not ready");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (blob) {
        sendImage(blob);
      }
    }, "image/jpeg");
  };

  // Send image and User ID to FastAPI (/upload)
  const sendImage = async (imageBlob) => {
    if (!userId.trim()) {
      setMessage("Please enter User ID");
      return;
    }

    const formData = new FormData();
    formData.append("user_id", userId.trim());
    formData.append("image", imageBlob, "capture.jpg");

    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        setMessage(result.message || "User registered successfully");
      } else {
        setMessage(result.detail?.message || result.error || "Registration failed");
      }
    } catch (error) {
      setMessage("Failed to send image to backend");
    }
  };

  // Stop webcam
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
    setMessage("Camera stopped");
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h2>User Registration</h2>
      <p style={{ color: "#666", marginBottom: "15px" }}>
        Capture and register your face biometric profile
      </p>

      <input
        type="text"
        placeholder="Enter User ID"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
        style={{
          padding: "8px 14px",
          fontSize: "16px",
          borderRadius: "6px",
          border: "1px solid #ccc",
          width: "260px"
        }}
      />

      <br /><br />

      <video
        ref={videoRef}
        autoPlay
        playsInline
        width="400"
        style={{
          background: "#222",
          borderRadius: "8px",
          boxShadow: "0 4px 10px rgba(0,0,0,0.15)"
        }}
      />

      <br /><br />

      <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
        <button
          onClick={startCamera}
          style={{
            padding: "8px 16px",
            fontSize: "15px",
            cursor: "pointer",
            borderRadius: "6px"
          }}
        >
          Start Camera
        </button>

        <button
          onClick={captureImage}
          disabled={!cameraOn}
          style={{
            padding: "8px 16px",
            fontSize: "15px",
            cursor: cameraOn ? "pointer" : "not-allowed",
            borderRadius: "6px"
          }}
        >
          Capture & Register
        </button>

        <button
          onClick={stopCamera}
          disabled={!cameraOn}
          style={{
            padding: "8px 16px",
            fontSize: "15px",
            cursor: cameraOn ? "pointer" : "not-allowed",
            borderRadius: "6px"
          }}
        >
          Stop Camera
        </button>
      </div>

      <h3 style={{ marginTop: "20px", color: "#333" }}>{message}</h3>
    </div>
  );
}

export default Registration;

