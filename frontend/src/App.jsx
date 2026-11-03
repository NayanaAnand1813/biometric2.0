
import { useState, useRef } from "react";

function App() {
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
      videoRef.current.srcObject = stream;
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

  // Send image and User ID to FastAPI
  const sendImage = async (imageBlob) => {
    if (!userId.trim()) {
      setMessage("Please enter User ID");
      return;
    }

    const formData = new FormData();
    formData.append("user_id", userId);
    formData.append("image", imageBlob, "capture.jpg");

    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData
      });

      const result = await response.json();

      setMessage(result.message);
    } catch (error) {
      setMessage("Failed to send image to backend");
    }
  };

  // Stop webcam
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
    setMessage("Camera stopped");
  };

  return (
    <div style={{ textAlign: "center", padding: "30px" }}>
      <h1>Biometric Face Recognition</h1>

      <input
        type="text"
        placeholder="Enter User ID"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
      />

      <br /><br />

      <video
        ref={videoRef}
        autoPlay
        playsInline
        width="400"
        style={{ background: "#222" }}
      />

      <br /><br />

      <button onClick={startCamera}>Start Camera</button>

      <button onClick={captureImage} disabled={!cameraOn}>
        Capture & Send
      </button>

      <button onClick={stopCamera} disabled={!cameraOn}>
        Stop Camera
      </button>

      <h3>{message}</h3>
    </div>
  );
}

export default App;
