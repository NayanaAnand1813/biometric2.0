"""
Backend-Issued Liveness & Two-Stage Registration Session Manager.

Enforces strict verification sequencing & temporary in-memory biometric frame storage:
1. Client initiates a live camera session via /api/liveness/session/start.
   - For registration flows, grants a configurable short TTL (default: 300s / 5 minutes).
   - For login flows, default TTL is 90s.
2. Client streams/submits live webcam frames to /api/liveness/session/verify-frame.
3. Backend validates frame liveness with MiniFASNet.
   - Spoofed, replayed, or invalid face captures are immediately rejected.
   - Genuine verified face frames are temporarily retained strictly in RAM.
4. Stage B: Face frames are held temporarily in backend RAM (never written to disk or PostgreSQL).
5. Stage C: User confirms face (CONFIRM FACE) or requests a retake (RETAKE).
   - CONFIRM FACE marks face capture as ready.
   - RETAKE purges previous temporary frames and resets liveness verification.
6. Stage D: Register User (/register) strictly requires an active, verified, unexpired,
   confirmed, and unconsumed session bound to the user.
7. Cleanup: Frames are automatically purged on registration success, cancellation, retake,
   terminal validation failure, and session expiry.
"""

import os
import time
import uuid
import secrets
import threading
import logging
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL_SECONDS = 90  # 90 seconds for live login camera interaction
REGISTRATION_SESSION_TTL_SECONDS = int(os.getenv("REGISTRATION_SESSION_TTL_SECONDS", "300"))  # 5 minutes for registration
MAX_CONCURRENT_REGISTRATION_SESSIONS = int(os.getenv("MAX_CONCURRENT_REGISTRATION_SESSIONS", "50"))
SESSION_CLEANUP_INTERVAL_SECONDS = int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", "60"))

# RAM-only storage limits to safeguard server memory
MAX_TEMP_FRAMES = 5
MAX_FRAME_WIDTH = 1920
MAX_FRAME_HEIGHT = 1080
MAX_SESSION_BYTES = 10 * 1024 * 1024  # 10 MB per session limit


class RegistrationCapacityExceededError(Exception):
    """Raised when the maximum number of concurrent registration sessions has been reached."""
    pass


class LivenessSession:
    """Represents an active biometric liveness verification & temporary registration session."""

    def __init__(
        self,
        session_id: str,
        action: str,
        user_id: str | None = None,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS
    ) -> None:
        self.session_id = session_id
        self.action = action  # "register" or "login"
        self.user_id = user_id
        self.challenge = secrets.token_hex(16)
        self.ttl_seconds = ttl_seconds
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds

        self.is_verified = False
        self.verified_at: float | None = None
        self.verified_score: float = 0.0
        self.real_frame_count = 0
        self.total_frames_checked = 0
        self.is_consumed = False
        self.consumed_at: float | None = None

        # Two-stage registration temporary face frames (RAM ONLY - never written to disk/DB)
        self.temp_frames: list[np.ndarray] = []
        self._last_verified_frame: np.ndarray | None = None
        self.face_captured: bool = False
        self.captured_at: float | None = None
        self.is_confirmed: bool = False
        self.confirmed_at: float | None = None
        self.total_frame_bytes: int = 0

    def is_expired(self) -> bool:
        """Returns True if the session exceeded its TTL."""
        return time.time() > self.expires_at

    def add_temp_frame(self, frame: np.ndarray) -> bool:
        """
        Stores a verified face frame temporarily in RAM.
        Validates dimensions, bounds, and memory limits.
        Never saves to disk or database.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False

        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH or h > MAX_FRAME_HEIGHT:
            scale = min(MAX_FRAME_WIDTH / w, MAX_FRAME_HEIGHT / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            import cv2
            frame = cv2.resize(frame, (new_w, new_h))

        frame_bytes = frame.nbytes
        while self.temp_frames and (self.total_frame_bytes + frame_bytes > MAX_SESSION_BYTES):
            removed = self.temp_frames.pop(0)
            self.total_frame_bytes -= removed.nbytes

        if len(self.temp_frames) >= MAX_TEMP_FRAMES:
            removed = self.temp_frames.pop(0)
            self.total_frame_bytes -= removed.nbytes

        self.temp_frames.append(frame.copy())
        self.total_frame_bytes += frame.nbytes
        self.face_captured = True
        self.captured_at = time.time()
        return True

    def get_primary_temp_frame(self) -> np.ndarray | None:
        """Returns the primary (latest) verified frame from RAM."""
        if self.temp_frames:
            return self.temp_frames[-1].copy()
        return None

    def get_temp_frames(self) -> list[np.ndarray]:
        """Returns copies of all temporarily held frames."""
        return [f.copy() for f in self.temp_frames]

    def confirm_capture(self) -> None:
        """Stage C: Marks face capture as confirmed by user."""
        self.is_confirmed = True
        self.confirmed_at = time.time()

    def clear_temp_frames(self) -> None:
        """Completely purges all temporary frames from memory."""
        self.temp_frames.clear()
        self._last_verified_frame = None
        self.face_captured = False
        self.captured_at = None
        self.is_confirmed = False
        self.confirmed_at = None
        self.total_frame_bytes = 0

    def reset_for_retake(self) -> None:
        """
        Stage C Retake: Discards temporary capture, resets confirmation and verification state
        so user can repeat liveness verification.
        """
        self.clear_temp_frames()
        self.is_verified = False
        self.verified_at = None
        self.verified_score = 0.0
        self.real_frame_count = 0
        self.total_frames_checked = 0

    def to_dict(self) -> dict:
        """Returns client-safe session representation."""
        return {
            "session_id": self.session_id,
            "action": self.action,
            "user_id": self.user_id,
            "challenge": self.challenge,
            "is_verified": self.is_verified,
            "verified_score": self.verified_score,
            "face_captured": self.face_captured,
            "is_confirmed": self.is_confirmed,
            "frame_count": len(self.temp_frames),
            "expires_in": max(0, int(self.expires_at - time.time())),
            "is_consumed": self.is_consumed
        }


class LivenessSessionManager:
    """
    Thread-safe in-memory session manager with automatic expiration cleanup.
    Ensures that only authentic live webcam sessions can proceed to registration/login.
    Guarantees strict isolation of concurrent registration sessions and bounded memory.
    """

    def __init__(
        self,
        default_ttl: int = DEFAULT_SESSION_TTL_SECONDS,
        max_concurrent_registration_sessions: int = MAX_CONCURRENT_REGISTRATION_SESSIONS,
        cleanup_interval_seconds: int = SESSION_CLEANUP_INTERVAL_SECONDS
    ) -> None:
        self._sessions: dict[str, LivenessSession] = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl
        self.max_concurrent_registration_sessions = max_concurrent_registration_sessions
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._cleanup_task: asyncio.Task | None = None

    def get_active_registration_count(self) -> int:
        """Returns the current number of active, unconsumed registration sessions."""
        with self._lock:
            self._cleanup_expired_locked()
            return sum(
                1 for s in self._sessions.values()
                if s.action == "register" and not s.is_consumed
            )

    def get_active_login_count(self) -> int:
        """Returns the current number of active, unconsumed login sessions."""
        with self._lock:
            self._cleanup_expired_locked()
            return sum(
                1 for s in self._sessions.values()
                if s.action == "login" and not s.is_consumed
            )

    def create_session(
        self,
        action: str,
        user_id: str | None = None,
        ttl_seconds: int | None = None
    ) -> LivenessSession:
        """Creates and registers a new liveness session."""
        if action not in ("register", "login"):
            raise ValueError(f"Invalid action '{action}'. Must be 'register' or 'login'.")

        session_id = f"live_{uuid.uuid4().hex}"
        if ttl_seconds is not None:
            ttl = ttl_seconds
        elif action == "register":
            ttl = REGISTRATION_SESSION_TTL_SECONDS
        else:
            ttl = self.default_ttl

        session = LivenessSession(
            session_id=session_id,
            action=action,
            user_id=user_id,
            ttl_seconds=ttl
        )

        with self._lock:
            # 1. Purge expired sessions first to ensure accurate capacity accounting
            self._cleanup_expired_locked()

            # 2. Enforce global registration concurrency limit atomically
            if action == "register":
                active_registration_count = sum(
                    1 for s in self._sessions.values()
                    if s.action == "register" and not s.is_consumed
                )
                if active_registration_count >= self.max_concurrent_registration_sessions:
                    logger.warning(
                        f"[LIVENESS SESSION] Registration capacity limit reached "
                        f"({active_registration_count}/{self.max_concurrent_registration_sessions}). Session rejected."
                    )
                    raise RegistrationCapacityExceededError(
                        f"Server registration capacity reached ({self.max_concurrent_registration_sessions} concurrent sessions). "
                        f"Please try again shortly."
                    )

            # 3. Store new session in isolated session dictionary
            self._sessions[session_id] = session

        logger.info(f"[LIVENESS SESSION] Created session {session_id} for action '{action}' (TTL: {ttl}s).")
        return session

    def get_session(self, session_id: str) -> LivenessSession | None:
        """Retrieves a session if it exists and has not expired."""
        with self._lock:
            self._cleanup_expired_locked()
            return self._sessions.get(session_id)

    def record_verification_result(
        self,
        session_id: str,
        liveness_result: dict,
        verified_frame: np.ndarray | None = None
    ) -> tuple[bool, str, dict]:
        """
        Updates session state based on incoming live camera frame evaluation.
        Marks session verified when is_real is True and score meets threshold.
        Automatically retains genuine verified frame in RAM if provided.
        """
        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)

            if not session:
                return False, "Liveness session not found or expired. Please restart the camera.", {}

            if session.is_expired():
                session.clear_temp_frames()
                return False, "Liveness session has expired. Please restart the camera.", {}

            if session.is_consumed:
                return False, "This liveness session has already been used and cannot be reused.", {}

            session.total_frames_checked += 1
            is_real = bool(liveness_result.get("is_real", False))
            score = float(liveness_result.get("score", 0.0))

            if is_real:
                session.real_frame_count += 1
                session.is_verified = True
                session.verified_at = time.time()
                session.verified_score = max(session.verified_score, score)

                # Store verified frame in RAM for subsequent capture
                if verified_frame is not None and isinstance(verified_frame, np.ndarray) and verified_frame.size > 0:
                    session._last_verified_frame = verified_frame.copy()

                logger.info(
                    f"[LIVENESS SESSION] Session {session_id} marked VERIFIED "
                    f"(score: {score:.4f}, real_frames: {session.real_frame_count})."
                )

                return True, "Real face confirmed. Liveness verification succeeded.", session.to_dict()

            # If fake/spoof, discard any previously captured temp frames
            if session.action == "register" and not is_real:
                session.clear_temp_frames()

            return False, liveness_result.get("message", "Liveness verification failed."), session.to_dict()

    def store_temp_frame(
        self,
        session_id: str,
        frame: np.ndarray,
        user_id: str | None = None
    ) -> tuple[bool, str, dict]:
        """
        Stage B: Stores a verified face frame in temporary backend RAM.
        Never saves to disk or database.
        """
        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)

            if not session:
                return False, "Liveness session not found or expired.", {}

            if session.is_expired():
                session.clear_temp_frames()
                return False, "Liveness session has expired.", {}

            if session.is_consumed:
                return False, "Session has already been consumed.", {}

            if session.action != "register":
                return False, "Temporary frame storage is only permitted for registration flows.", {}

            if user_id and session.user_id and session.user_id != user_id.strip():
                return False, f"Unauthorized: Session belongs to '{session.user_id}', not '{user_id}'.", {}

            if not session.is_verified:
                return False, "Face has not passed liveness verification. Cannot store unverified frame.", {}

            success = session.add_temp_frame(frame)
            if not success:
                return False, "Invalid or empty image frame.", {}

            return True, "Face frame held in temporary RAM.", session.to_dict()

    def confirm_capture(
        self,
        session_id: str,
        user_id: str | None = None
    ) -> tuple[bool, str, dict]:
        """
        Stage C: Confirms the captured face frame in RAM, enabling the Register User action.
        """
        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)

            if not session:
                return False, "Liveness session not found or expired.", {}

            if session.is_expired():
                session.clear_temp_frames()
                return False, "Liveness session has expired.", {}

            if session.is_consumed:
                return False, "Session has already been consumed.", {}

            if session.action != "register":
                return False, "Capture confirmation is only permitted for registration flows.", {}

            if user_id and session.user_id and session.user_id != user_id.strip():
                return False, f"Unauthorized: Session belongs to '{session.user_id}', not '{user_id}'.", {}

            if not session.is_verified:
                return False, "Face has not passed liveness verification.", {}

            if not session.temp_frames:
                return False, "No verified face frame available to confirm.", {}

            session.confirm_capture()
            return True, "Face capture confirmed. Ready for registration.", session.to_dict()

    def retake_capture(
        self,
        session_id: str,
        user_id: str | None = None
    ) -> tuple[bool, str, dict]:
        """
        Stage C: Discards previously captured temporary face frames from RAM and resets liveness state.
        """
        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)

            if not session:
                return False, "Liveness session not found or expired.", {}

            if session.is_consumed:
                return False, "Session has already been consumed.", {}

            if user_id and session.user_id and session.user_id != user_id.strip():
                return False, f"Unauthorized: Session belongs to '{session.user_id}', not '{user_id}'.", {}

            session.reset_for_retake()
            return True, "Previous temporary capture discarded. Liveness verification reset.", session.to_dict()

    def cancel_session(
        self,
        session_id: str,
        user_id: str | None = None
    ) -> tuple[bool, str]:
        """
        Cancels registration and purges all temporary frames from memory immediately.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False, "Session not found."

            if user_id and session.user_id and session.user_id != user_id.strip():
                return False, f"Unauthorized: Session belongs to '{session.user_id}', not '{user_id}'."

            session.clear_temp_frames()
            del self._sessions[session_id]
            logger.info(f"[LIVENESS SESSION] Session {session_id} cancelled and temporary frames purged.")
            return True, "Registration session cancelled and RAM purged."

    def validate_and_consume(
        self,
        session_id: str | None,
        expected_action: str,
        expected_user_id: str | None = None,
        require_confirmed: bool = False
    ) -> tuple[bool, str, LivenessSession | None]:
        """
        Validates that the session is active, matches expected action, is verified,
        bound to user_id, has confirmed capture (if required), and has not been consumed.
        If valid, marks it consumed immediately (one-time use to prevent replay).
        """
        if not session_id or not str(session_id).strip():
            return (
                False,
                "Liveness session ID is missing. Direct API requests or file uploads without "
                "live webcam verification are strictly prohibited.",
                None
            )

        clean_id = str(session_id).strip()

        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(clean_id)

            if not session:
                return (
                    False,
                    "Invalid or expired liveness session. Live webcam verification is strictly required.",
                    None
                )

            if session.action != expected_action:
                return (
                    False,
                    f"Session action mismatch: expected '{expected_action}', session is for '{session.action}'.",
                    None
                )

            if session.is_expired():
                session.clear_temp_frames()
                return (
                    False,
                    "Liveness session has expired. Live camera verification must be repeated.",
                    None
                )

            if expected_user_id and session.user_id and session.user_id != str(expected_user_id).strip():
                return (
                    False,
                    f"Session user mismatch: session was issued for '{session.user_id}', not '{expected_user_id}'.",
                    None
                )

            if not session.is_verified:
                return (
                    False,
                    "Liveness verification has not been completed. Live face must be verified before proceeding.",
                    None
                )

            if session.is_consumed:
                return (
                    False,
                    "Liveness session has already been used. Replay attempt rejected.",
                    None
                )

            # In two-stage registration, if face capture occurred or require_confirmed is True,
            # enforce that the user confirmed the face capture in Stage C.
            if require_confirmed or (expected_action == "register" and session.face_captured):
                if not session.is_confirmed:
                    return (
                        False,
                        "Face capture must be confirmed before registration can proceed.",
                        None
                    )

            # Consume session
            session.is_consumed = True
            session.consumed_at = time.time()

            logger.info(f"[LIVENESS SESSION] Session {clean_id} successfully validated and consumed.")
            return True, "Session verified and consumed.", session

    def _cleanup_expired_locked(self) -> int:
        """Removes expired sessions and clears temporary frames. Must be called with self._lock held."""
        now = time.time()
        expired_ids = [sid for sid, s in self._sessions.items() if now > s.expires_at]
        for sid in expired_ids:
            s = self._sessions.pop(sid, None)
            if s:
                s.clear_temp_frames()
        return len(expired_ids)

    def cleanup_expired_sessions(self) -> int:
        """
        Public thread-safe method to remove all expired sessions and clear temporary NumPy frames.
        Returns the number of expired sessions removed.
        Safe for use by background cleanup tasks and direct invocations.
        """
        with self._lock:
            count = self._cleanup_expired_locked()
            if count > 0:
                logger.info(f"[LIVENESS SESSION] Cleaned up {count} expired session(s) and purged temporary RAM frames.")
            return count

    async def _periodic_cleanup_loop(self, interval_seconds: int) -> None:
        """
        Background loop executed as an asyncio.Task.
        Periodically invokes cleanup_expired_sessions in a background thread to prevent
        blocking the FastAPI event loop.
        """
        logger.info(f"[LIVENESS SESSION] Background cleanup task running (every {interval_seconds}s).")
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                # Offload cleanup to a worker thread so event loop is never blocked
                await asyncio.to_thread(self.cleanup_expired_sessions)
        except asyncio.CancelledError:
            logger.info("[LIVENESS SESSION] Background cleanup task cancelled.")
            raise

    def start_background_cleanup(self, interval_seconds: int | None = None) -> asyncio.Task | None:
        """
        Starts the periodic background cleanup task if not already running.
        Guarantees idempotency — will not create duplicate background tasks on reload or multiple calls.
        """
        interval = interval_seconds if interval_seconds is not None else self.cleanup_interval_seconds
        with self._lock:
            if self._cleanup_task is not None and not self._cleanup_task.done():
                logger.debug("[LIVENESS SESSION] Background cleanup task already running.")
                return self._cleanup_task

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning("[LIVENESS SESSION] No running asyncio event loop; background cleanup task not started.")
                return None

            self._cleanup_task = loop.create_task(
                self._periodic_cleanup_loop(interval),
                name="liveness_session_periodic_cleanup"
            )
            logger.info(f"[LIVENESS SESSION] Initiated periodic background cleanup task (interval: {interval}s).")
            return self._cleanup_task

    async def stop_background_cleanup(self) -> None:
        """
        Cancels and cleanly awaits completion of the background cleanup task during application shutdown.
        """
        task = None
        with self._lock:
            if self._cleanup_task is not None:
                task = self._cleanup_task
                self._cleanup_task = None

        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info("[LIVENESS SESSION] Background cleanup task stopped cleanly.")

    def get_cleanup_task(self) -> asyncio.Task | None:
        """Returns the current background cleanup task, if any."""
        with self._lock:
            return self._cleanup_task


# Global singleton
_session_manager_instance: LivenessSessionManager | None = None
_manager_lock = threading.Lock()


def get_liveness_session_manager() -> LivenessSessionManager:
    """Returns the global singleton instance of LivenessSessionManager."""
    global _session_manager_instance
    if _session_manager_instance is None:
        with _manager_lock:
            if _session_manager_instance is None:
                _session_manager_instance = LivenessSessionManager()
    return _session_manager_instance
