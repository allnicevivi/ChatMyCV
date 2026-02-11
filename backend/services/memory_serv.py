"""
Redis Memory Service

Provides persistent session-based conversational memory using Redis.
Replaces the in-memory _ConversationStore with a durable Redis-backed solution.
"""

import json
import logging
import os
import time
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Lazy import Redis to allow graceful degradation
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis not installed. Memory will fall back to in-memory storage.")
    REDIS_AVAILABLE = False
    redis = None


class RedisMemoryStore:
    """
    Redis-backed conversation memory store with session persistence.

    Stores conversation history in Redis lists, with automatic expiration
    and configurable message limits per session.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        max_messages: int = 20,
        session_ttl: int = 86400  # 24 hours
    ):
        """
        Initialize Redis memory store

        Args:
            host: Redis host (defaults to env REDIS_HOST or localhost)
            port: Redis port (defaults to env REDIS_PORT or 6379)
            db: Redis database number (defaults to env REDIS_DB or 0)
            password: Redis password (defaults to env REDIS_PASSWORD or None)
            max_messages: Maximum messages to keep per session (default 20)
            session_ttl: Session TTL in seconds (default 86400 = 24 hours)
        """
        self.max_messages = max_messages
        self.session_ttl = session_ttl
        self._redis_client = None

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Using in-memory fallback.")
            self._use_fallback = True
            self._fallback_store: Dict[str, List[Dict[str, str]]] = {}
            self._fallback_activity: Dict[str, float] = {}
            return

        # Get Redis configuration
        redis_host = host or os.getenv("REDIS_HOST", "localhost")
        redis_port = port or int(os.getenv("REDIS_PORT", "6379"))
        redis_db = db if db is not None else int(os.getenv("REDIS_DB", "0"))
        redis_password = password or os.getenv("REDIS_PASSWORD") or None

        try:
            self._redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password if redis_password else None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )

            # Test connection
            self._redis_client.ping()
            logger.info(f"Redis memory store initialized successfully. Host: {redis_host}:{redis_port}")
            self._use_fallback = False

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}. Using in-memory fallback.")
            self._redis_client = None
            self._use_fallback = True
            self._fallback_store = {}
            self._fallback_activity = {}

    def _get_session_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"chatmycv:session:{session_id}:messages"

    def _get_activity_key(self, session_id: str) -> str:
        """Generate Redis key for session activity timestamp"""
        return f"chatmycv:session:{session_id}:activity"

    def save_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Save a message to session history

        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content

        Returns:
            True if successful, False otherwise
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }

        if self._use_fallback:
            # In-memory fallback
            if session_id not in self._fallback_store:
                self._fallback_store[session_id] = []

            self._fallback_store[session_id].append({"role": role, "content": content})
            self._fallback_activity[session_id] = time.time()

            # Trim to max_messages
            if len(self._fallback_store[session_id]) > self.max_messages:
                self._fallback_store[session_id] = self._fallback_store[session_id][-self.max_messages:]

            return True

        try:
            session_key = self._get_session_key(session_id)
            activity_key = self._get_activity_key(session_id)

            # Add message to list
            self._redis_client.rpush(session_key, json.dumps(message))

            # Trim to keep only last max_messages
            self._redis_client.ltrim(session_key, -self.max_messages, -1)

            # Update activity timestamp
            self._redis_client.set(activity_key, str(time.time()))

            # Set TTL for both keys
            self._redis_client.expire(session_key, self.session_ttl)
            self._redis_client.expire(activity_key, self.session_ttl)

            logger.debug(f"Saved {role} message to session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving message to Redis: {e}")
            return False

    def load_history(self, session_id: str, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Load conversation history for a session

        Args:
            session_id: Session identifier
            max_messages: Maximum messages to retrieve (defaults to instance max_messages)

        Returns:
            List of message dictionaries [{"role": "user/assistant", "content": "..."}]
        """
        max_msg = max_messages or self.max_messages

        if self._use_fallback:
            # In-memory fallback
            messages = self._fallback_store.get(session_id, [])
            self._fallback_activity[session_id] = time.time()
            return messages[-max_msg:] if messages else []

        try:
            session_key = self._get_session_key(session_id)
            activity_key = self._get_activity_key(session_id)

            # Get messages from Redis (last max_msg messages)
            messages_json = self._redis_client.lrange(session_key, -max_msg, -1)

            if not messages_json:
                return []

            # Parse JSON messages and extract role/content
            messages = []
            for msg_json in messages_json:
                try:
                    msg = json.loads(msg_json)
                    messages.append({
                        "role": msg.get("role"),
                        "content": msg.get("content")
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse message JSON: {msg_json}")
                    continue

            # Update activity timestamp
            self._redis_client.set(activity_key, str(time.time()))
            self._redis_client.expire(activity_key, self.session_ttl)

            logger.debug(f"Loaded {len(messages)} messages for session {session_id}")
            return messages

        except Exception as e:
            logger.error(f"Error loading history from Redis: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a specific session

        Args:
            session_id: Session identifier

        Returns:
            True if cleared, False otherwise
        """
        if self._use_fallback:
            # In-memory fallback
            if session_id in self._fallback_store:
                del self._fallback_store[session_id]
                if session_id in self._fallback_activity:
                    del self._fallback_activity[session_id]
                return True
            return False

        try:
            session_key = self._get_session_key(session_id)
            activity_key = self._get_activity_key(session_id)

            # Delete both keys
            deleted = self._redis_client.delete(session_key, activity_key)

            logger.info(f"Cleared session {session_id}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Error clearing session from Redis: {e}")
            return False

    def clear_all_sessions(self) -> int:
        """
        Clear all conversation sessions

        Returns:
            Number of sessions cleared
        """
        if self._use_fallback:
            # In-memory fallback
            count = len(self._fallback_store)
            self._fallback_store.clear()
            self._fallback_activity.clear()
            logger.info(f"Cleared all {count} fallback sessions")
            return count

        try:
            # Find all session keys
            pattern = "chatmycv:session:*"
            keys = self._redis_client.keys(pattern)

            if not keys:
                return 0

            # Delete all keys
            deleted = self._redis_client.delete(*keys)

            logger.info(f"Cleared {deleted} keys from Redis")
            return deleted // 2  # Divide by 2 since each session has messages + activity keys

        except Exception as e:
            logger.error(f"Error clearing all sessions from Redis: {e}")
            return 0

    def get_last_session(self) -> Tuple[Optional[str], float]:
        """
        Get the most recent session ID and its last activity time

        Returns:
            Tuple of (session_id, last_activity_time) or (None, 0.0) if no sessions
        """
        if self._use_fallback:
            # In-memory fallback
            if not self._fallback_activity:
                return None, 0.0

            most_recent_id = max(self._fallback_activity, key=self._fallback_activity.get)
            most_recent_time = self._fallback_activity[most_recent_id]
            return most_recent_id, most_recent_time

        try:
            # Find all activity keys
            pattern = "chatmycv:session:*:activity"
            activity_keys = self._redis_client.keys(pattern)

            if not activity_keys:
                return None, 0.0

            # Find the most recent one
            most_recent_time = 0.0
            most_recent_id = None

            for key in activity_keys:
                timestamp_str = self._redis_client.get(key)
                if timestamp_str:
                    timestamp = float(timestamp_str)
                    if timestamp > most_recent_time:
                        most_recent_time = timestamp
                        # Extract session_id from key: chatmycv:session:{session_id}:activity
                        most_recent_id = key.split(":")[2]

            return most_recent_id, most_recent_time

        except Exception as e:
            logger.error(f"Error getting last session from Redis: {e}")
            return None, 0.0

    @property
    def is_redis_connected(self) -> bool:
        """Check if Redis is connected"""
        if self._use_fallback:
            return False

        try:
            self._redis_client.ping()
            return True
        except:
            return False


# Singleton instance for use across the application
redis_memory_store = RedisMemoryStore()


# Export key components
__all__ = [
    'RedisMemoryStore',
    'redis_memory_store',
    'REDIS_AVAILABLE'
]
