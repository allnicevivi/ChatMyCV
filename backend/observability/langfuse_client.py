"""
Langfuse Observability Client

Provides singleton Langfuse client for tracing LLM operations across the RAG pipeline.
Includes helper functions for creating traces and spans with proper metadata.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Lazy import Langfuse to allow graceful degradation
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    logger.warning("Langfuse not installed. Observability features will be disabled.")
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None
    langfuse_context = None


class LangfuseClient:
    """Singleton Langfuse client for observability across the application"""

    _instance: Optional['LangfuseClient'] = None
    _langfuse: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Langfuse client with configuration from environment"""
        if not LANGFUSE_AVAILABLE:
            logger.warning("Langfuse not available. Tracing will be disabled.")
            self._langfuse = None
            return

        try:
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

            if not public_key or not secret_key:
                logger.warning(
                    "Langfuse credentials not configured. "
                    "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env"
                )
                self._langfuse = None
                return

            self._langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            logger.info(f"Langfuse initialized successfully. Host: {host}")

        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self._langfuse = None

    @property
    def client(self):
        """Get the underlying Langfuse client"""
        return self._langfuse

    @property
    def is_enabled(self) -> bool:
        """Check if Langfuse is properly configured and available"""
        return self._langfuse is not None

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Create a new trace for an operation

        Args:
            name: Name of the operation (e.g., "chat_request", "document_processing")
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional metadata dictionary
            tags: Optional list of tags

        Returns:
            Trace object or None if Langfuse is not available
        """
        if not self.is_enabled:
            return None

        try:
            return self._langfuse.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or []
            )
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None

    def flush(self):
        """Flush any pending traces to Langfuse"""
        if self.is_enabled:
            try:
                self._langfuse.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")


# Global singleton instance
langfuse_client = LangfuseClient()


def trace_operation(name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace a function as a Langfuse span

    Usage:
        @trace_operation("retrieval", metadata={"k": 5})
        def retrieve_docs(query):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not langfuse_client.is_enabled:
                # If Langfuse not available, just run the function
                return func(*args, **kwargs)

            try:
                # This would require langfuse_context from decorators
                # For now, we'll just log and execute
                logger.debug(f"Tracing operation: {name}")
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error in traced operation {name}: {e}")
                raise

        return wrapper
    return decorator


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID from Langfuse context"""
    if not LANGFUSE_AVAILABLE or not langfuse_client.is_enabled:
        return None

    try:
        if langfuse_context:
            trace = langfuse_context.get_current_trace()
            return trace.id if trace else None
    except Exception as e:
        logger.debug(f"Could not get current trace ID: {e}")

    return None


# Export key components
__all__ = [
    'langfuse_client',
    'LangfuseClient',
    'trace_operation',
    'get_current_trace_id',
    'LANGFUSE_AVAILABLE'
]
