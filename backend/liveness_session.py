"""
Backend-Issued Liveness Session Manager for Camera-Only Biometric Workflows.

Enforces strict verification sequencing:
1. Client initiates a live camera session via /api/liveness/session/start.
2. Client streams/submits live webcam frames to /api/liveness/session/verify-frame.
3. Backend validates frame liveness with MiniFASNet. Only genuine frames with
   confidence >= threshold mark the session as verified.
4. Client captures the verified face only AFTER session verification succeeds.
5. Registration (/register) and Login (/login) strictly require an active, verified,
   unexpired, and unconsumed liveness session. Direct API requests or static file uploads
   without a verified session are rejected with HTTP 403 Forbidden.
6. Once consumed, the session cannot be replayed.
"""

import time
import uuid
import secrets
import threading
import logging

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL_SECONDS = 90  # 90 seconds to complete live camera interaction


class LivenessSession:
    """Represents an active biometric liveness verification session."""

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
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds

        self.is_verified = False
        self.verified_at: float | None = None
        self.verified_score: float = 0.0
        self.real_frame_count = 0
        self.total_frames_checked = 0
        self.is_consumed = False
        self.consumed_at: float | None = None

    def is_expired(self) -> bool:
        """Returns True if the session exceeded its TTL."""
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        """Returns client-safe session representation."""
        return {
            "session_id": self.session_id,
            "action": self.action,
            "challenge": self.challenge,
            "is_verified": self.is_verified,
            "verified_score": self.verified_score,
            "expires_in": max(0, int(self.expires_at - time.time())),
            "is_consumed": self.is_consumed
        }


class LivenessSessionManager:
    """
    Thread-safe in-memory session manager with automatic expiration cleanup.
    Ensures that only authentic live webcam sessions can proceed to registration/login.
    """

    def __init__(self, default_ttl: int = DEFAULT_SESSION_TTL_SECONDS) -> None:
        self._sessions: dict[str, LivenessSession] = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl

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
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        session = LivenessSession(
            session_id=session_id,
            action=action,
            user_id=user_id,
            ttl_seconds=ttl
        )

        with self._lock:
            self._cleanup_expired_locked()
            self._sessions[session_id] = session

        logger.info(f"[LIVENESS SESSION] Created session {session_id} for action '{action}'.")
        return session

    def get_session(self, session_id: str) -> LivenessSession | None:
        """Retrieves a session if it exists and has not expired."""
        with self._lock:
            self._cleanup_expired_locked()
            return self._sessions.get(session_id)

    def record_verification_result(
        self,
        session_id: str,
        liveness_result: dict
    ) -> tuple[bool, str, dict]:
        """
        Updates session state based on incoming live camera frame evaluation.
        Marks session verified when is_real is True and score meets threshold.
        """
        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)

            if not session:
                return False, "Liveness session not found or expired. Please restart the camera.", {}

            if session.is_expired():
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

                logger.info(
                    f"[LIVENESS SESSION] Session {session_id} marked VERIFIED "
                    f"(score: {score:.4f}, real_frames: {session.real_frame_count})."
                )

                return True, "Real face confirmed. Liveness verification succeeded.", session.to_dict()

            return False, liveness_result.get("message", "Liveness verification failed."), session.to_dict()

    def validate_and_consume(
        self,
        session_id: str | None,
        expected_action: str
    ) -> tuple[bool, str, LivenessSession | None]:
        """
        Validates that the session is active, matches expected action, is verified,
        and has not been previously consumed.
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
                return (
                    False,
                    "Liveness session has expired. Live camera verification must be repeated.",
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

            # Consume session
            session.is_consumed = True
            session.consumed_at = time.time()

            logger.info(f"[LIVENESS SESSION] Session {clean_id} successfully validated and consumed.")
            return True, "Session verified and consumed.", session

    def _cleanup_expired_locked(self) -> None:
        """Removes expired sessions. Must be called with self._lock held."""
        now = time.time()
        expired_ids = [sid for sid, s in self._sessions.items() if now > s.expires_at]
        for sid in expired_ids:
            del self._sessions[sid]


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

