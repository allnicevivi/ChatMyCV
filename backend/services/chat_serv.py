#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

from typing import List, Dict, Optional, Any
import time
import threading
from modules import azure_client
from vectorstores.chroma_vectordb import chroma_usage
from utils.app_logger import LoggerSetup

logger = LoggerSetup("ChatService").logger


class ChatService:
    """Service for handling chat interactions with RAG (Retrieval Augmented Generation)."""
    
    def __init__(self):
        self.llm_client = azure_client
        self.vectorstore = chroma_usage
        self._conversation_store = _ConversationStore()
        self.default_system_prompt = (
            "You are the job candidate. Answer ONLY with facts grounded in the provided resume/CV context. \n"
            "Goal: make the interviewer want to know more by delivering direct, high-signal value.\n\n"
            "Output rules (must follow):\n"
            "- Answer directly in 2–5 short bullet points; no preamble, no summaries.\n"
            "- Lead with outcomes and metrics (impact, scale, revenue, costs, latency, users, reliability).\n"
            "- Map each point to the role’s needs (why it matters for them).\n"
            "- Use compact STAR (Situation/Task in 1 clause) → Action → Result with numbers.\n"
            "- If context is missing, state the gap in one line and propose 1 targeted follow-up you’d clarify.\n"
            "- Never invent facts; never restate the question; avoid generic claims (e.g., ‘strong leadership’).\n"
            "- End with one single-sentence teaser that invites a natural next question (no more than 15 words)."
        )

    def clear_history(self, session_id: str) -> bool:
        """Manually clear a single session's history. Returns True if removed."""
        return self._conversation_store.clear(session_id)

    def clear_all_histories(self) -> int:
        """Manually clear all sessions. Returns number of sessions removed."""
        return self._conversation_store.clear_all()
    
    def _retrieve_context(self, query: str, k: int = 5) -> List[tuple]:
        """
        Retrieve relevant context from the vectorstore.
        
        Args:
            query: User query text
            k: Number of documents to retrieve
            
        Returns:
            List of tuples (document, metadata, distance)
        """
        try:
            # Get embedding for the query
            query_embedding = self.llm_client.get_embedding(query)
            
            # Query the vectorstore
            results = self.vectorstore.query_collection(
                query_embedding=query_embedding,
                k=k
            )
            
            logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return []
    
    def _format_context(self, retrieved_docs: List[tuple]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            retrieved_docs: List of tuples (document, metadata, distance)
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for doc, metadata, distance in retrieved_docs:
            context_parts.append(f"[Source: {metadata.get('filename', 'unknown')}]\n{doc}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_messages(
        self,
        user_query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build messages for the LLM including system prompt, context, and conversation history.
        
        Args:
            user_query: Current user query
            context: Retrieved context from vectorstore
            conversation_history: Previous conversation messages
            system_prompt: Custom system prompt (optional)
            
        Returns:
            List of message dictionaries
        """
        system_prompt = system_prompt or self.default_system_prompt
        
        # Add context to system prompt if available
        if context:
            enhanced_system_prompt = f"{system_prompt}\n\nContext from documents:\n{context}"
        else:
            enhanced_system_prompt = system_prompt
        
        messages = [{"role": "system", "content": enhanced_system_prompt}]
        
        # Add conversation history (excluding system messages)
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") != "system":
                    messages.append(msg)
        
        # Add current user query
        messages.append({"role": "user", "content": user_query})
        
        return messages
    
    def chat(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a chat query with RAG.
        
        Args:
            query: User query text
            conversation_history: Previous conversation messages
            k: Number of documents to retrieve
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens to generate
            system_prompt: Custom system prompt (optional)
            model: Model name to use (optional)
            **kwargs: Additional LLM parameters
            
        Returns:
            Dictionary with response content, usage, and retrieved context info
        """
        try:
            # Cleanup expired conversations
            self._conversation_store.cleanup_expired()

            # Load persisted conversation if session_id is provided
            if session_id is not None and not conversation_history:
                conversation_history = self._conversation_store.get_history(session_id)

            # Retrieve relevant context
            retrieved_docs = self._retrieve_context(query, k=k)
            context = self._format_context(retrieved_docs)
            
            # Build messages
            messages = self._build_messages(
                user_query=query,
                context=context,
                conversation_history=conversation_history,
                system_prompt=system_prompt
            )
            
            # Get LLM response
            response = self.llm_client.get_chat_completion_with_usage(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Persist conversation state if session_id provided
            if session_id is not None:
                self._conversation_store.append(
                    session_id=session_id,
                    user_message=query,
                    assistant_message=response["content"],
                )

            return {
                "content": response["content"],
                "usage": response.get("usage", {}),
                "retrieved_docs_count": len(retrieved_docs),
                "context_used": bool(context)
            }
        except Exception as e:
            logger.error(f"Error in chat service: {e}", exc_info=True)
            raise
    
    def stream_chat(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Process a chat query with RAG and stream the response.
        
        Args:
            query: User query text
            conversation_history: Previous conversation messages
            k: Number of documents to retrieve
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens to generate
            system_prompt: Custom system prompt (optional)
            model: Model name to use (optional)
            **kwargs: Additional LLM parameters
            
        Yields:
            Chunks of the generated response
        """
        try:
            # Cleanup expired conversations
            self._conversation_store.cleanup_expired()

            # Load persisted conversation if session_id is provided
            if session_id is not None and not conversation_history:
                conversation_history = self._conversation_store.get_history(session_id)

            # Retrieve relevant context
            retrieved_docs = self._retrieve_context(query, k=k)
            context = self._format_context(retrieved_docs)
            
            # Build messages
            messages = self._build_messages(
                user_query=query,
                context=context,
                conversation_history=conversation_history,
                system_prompt=system_prompt
            )
            
            # Stream LLM response
            accumulated = []
            for chunk in self.llm_client.stream_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                if session_id is not None:
                    accumulated.append(chunk)
                yield chunk

            # After stream finishes, persist concatenated assistant reply
            if session_id is not None and accumulated:
                self._conversation_store.append(
                    session_id=session_id,
                    user_message=query,
                    assistant_message="".join(accumulated),
                )
        except Exception as e:
            logger.error(f"Error in stream chat service: {e}", exc_info=True)
            raise


class _ConversationStore:
    """In-memory session conversation store with idle expiry."""

    def __init__(self, idle_timeout_seconds: int = 300):
        self._idle_timeout = idle_timeout_seconds
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            session["last_activity"] = time.time()
            # Return a shallow copy to prevent outside mutation
            return list(session["messages"])  # type: ignore[return-value]

    def append(self, session_id: str, user_message: str, assistant_message: str) -> None:
        with self._lock:
            now = time.time()
            session = self._sessions.setdefault(session_id, {"messages": [], "last_activity": now})
            session["messages"].append({"role": "user", "content": user_message})
            session["messages"].append({"role": "assistant", "content": assistant_message})
            session["last_activity"] = now

    def cleanup_expired(self) -> None:
        with self._lock:
            now = time.time()
            expired = [sid for sid, s in self._sessions.items() if now - s.get("last_activity", 0) > self._idle_timeout]
            for sid in expired:
                del self._sessions[sid]

    def clear(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def clear_all(self) -> int:
        with self._lock:
            n = len(self._sessions)
            self._sessions.clear()
            return n


if __name__ == "__main__":
    # Example usage
    chat_service = ChatService()
    
    # # Simple query
    # response = chat_service.chat(
    #     query="What is the candidate's experience?",
    #     k=3
    # )
    # print("Response:", response["content"])
    # print("Usage:", response["usage"])
    
    # With conversation history
    history = [
        {"role": "user", "content": "What is the candidate's name?"},
        {"role": "assistant", "content": "The candidate's name is John Doe."}
    ]
    response = chat_service.chat(
        query="What skills do you have?",
        conversation_history=history,
        k=3
    )
    print("\nResponse with history:", response["content"])

