#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

from typing import List, Dict, Optional, Any, Tuple
import re
import time
import threading
import uuid
from modules import azure_client
from vectorstores.chroma_vectordb import ChromaUsage
from utils.app_logger import LoggerSetup
from . import prompter

# Import Langfuse for observability
try:
    from observability.langfuse_client import langfuse_client
    LANGFUSE_ENABLED = langfuse_client.is_enabled
except ImportError:
    langfuse_client = None
    LANGFUSE_ENABLED = False

# Import Redis memory store
try:
    from .memory_serv import redis_memory_store, REDIS_AVAILABLE
    MEMORY_STORE_AVAILABLE = True
except ImportError:
    redis_memory_store = None
    REDIS_AVAILABLE = False
    MEMORY_STORE_AVAILABLE = False

# Import router and agents
try:
    from .router import query_router
    from .agents import rag_agent, chat_agent, memory_agent
    ROUTER_AVAILABLE = True
except ImportError:
    query_router = None
    rag_agent = None
    chat_agent = None
    memory_agent = None
    ROUTER_AVAILABLE = False

# Import HITL service
try:
    from .hitl_serv import hitl_service, POSTGRES_AVAILABLE
    HITL_AVAILABLE = True
except ImportError:
    hitl_service = None
    POSTGRES_AVAILABLE = False
    HITL_AVAILABLE = False

logger = LoggerSetup("ChatService").logger

chroma_usage_en = ChromaUsage(collection_name="chat_cv_en")
chroma_usage_zhtw = ChromaUsage(collection_name="chat_cv_zhtw")
class ChatService:
    """Service for handling chat interactions with RAG (Retrieval Augmented Generation)."""

    def __init__(self, memory_store=None):
        self.llm_client = azure_client
        self.vectorstore = chroma_usage_en

        # Use Redis memory store if available, otherwise fallback to in-memory
        if memory_store:
            self._conversation_store = memory_store
        elif MEMORY_STORE_AVAILABLE and redis_memory_store:
            self._conversation_store = redis_memory_store
            logger.info(f"Using Redis memory store (connected: {redis_memory_store.is_redis_connected})")
        else:
            self._conversation_store = _ConversationStore()
            logger.warning("Using in-memory conversation store (no persistence)")

    def clear_history(self, session_id: str) -> bool:
        """Manually clear a single session's history. Returns True if removed."""
        # Support both Redis and in-memory interfaces
        if hasattr(self._conversation_store, 'clear_session'):
            return self._conversation_store.clear_session(session_id)
        else:
            return self._conversation_store.clear(session_id)

    def clear_all_histories(self) -> int:
        """Manually clear all sessions. Returns number of sessions removed."""
        # Support both Redis and in-memory interfaces
        if hasattr(self._conversation_store, 'clear_all_sessions'):
            return self._conversation_store.clear_all_sessions()
        else:
            return self._conversation_store.clear_all()
    
    def get_system_prompt(self, character: Optional[str] = None) -> str:
        """
        Get the system prompt based on the interviewer character.
        
        Args:
            character: "hr" or "engineer" or None for default
            
        Returns:
            System prompt string
        """
        if character is None:
            return prompter.HR_cot_system_prompt
            # return self.default_system_prompt
        
        if character == "engineer":
            return prompter.EM_cot_system_prompt
            # return self.engineer_prompt
        else:
            return prompter.HR_cot_system_prompt
            # return self.hr_prompt

    def get_or_create_session_id(self, session_id: Optional[str] = None, timeout_seconds: int = 180) -> str:
        """
        Get or create a session_id based on the timeout rule.
        
        Args:
            session_id: Optional session_id from request
            timeout_seconds: Timeout in seconds (default 3 minutes = 180)
            
        Returns:
            Session ID to use (either provided, last session if recent, or new)
        """
        if session_id is not None:
            return session_id
        
        # Get the most recent session
        last_session_id, last_activity_time = self._conversation_store.get_last_session()
        
        if last_session_id is None:
            # No previous session, create new one
            return str(uuid.uuid4())
        
        # Check if last activity was within timeout
        now = time.time()
        time_since_last_activity = now - last_activity_time
        
        if time_since_last_activity > timeout_seconds:
            # Last activity was more than timeout_seconds ago, create new session
            return str(uuid.uuid4())
        else:
            # Use the last session
            return last_session_id
    
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
    
    def _compose_retrieval_query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_history_chars: int = 2000,
    ) -> str:
        """
        Combine relevant conversation history with the latest user query for retrieval.
        
        Args:
            user_query: Current user query
            conversation_history: Prior conversation messages
            max_history_chars: Max characters from history to include to limit prompt size
            
        Returns:
            Combined retrieval query string
        """
        if not conversation_history:
            return user_query
        
        accumulated: List[str] = []
        char_count = 0
        
        # Traverse history backwards to capture most recent context
        for message in reversed(conversation_history):
            role = message.get("role")
            if role == "system":
                continue
            content = message.get("content", "").strip()
            if not content:
                continue
            
            formatted = f"{role}: {content}"
            char_count += len(formatted)
            if char_count > max_history_chars:
                break
            accumulated.append(formatted)
        
        if not accumulated:
            return user_query
        
        history_block = "\n".join(reversed(accumulated))
        return f"Conversation so far:\n{history_block}\n\nCurrent user query:\n{user_query}"
    
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
        
        # Add conversation history (excluding system messages)
        conversation_history_str = ""
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role")
                content = msg.get("content")
                conversation_history_str += f"{role}: {content}\n"

        # Add context to system prompt if available
        # enhanced_system_prompt = f"{system_prompt}\n\nContext from documents:\n{context}"

        user_prompt = prompter.cot_user_prompt.format(
            context_str=context,
            history=conversation_history_str,
            query_str=user_query
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(
                context_str=(
                    "company: ABC Corp\n"
                    "position: software engineer\n"
                    "period: 2021/03/01-present\n"
                    "contribution: \n"
                    "- led a small team and helped improve process efficiency\n\n"
                    "side projects:\n"
                    "- ..."),
                history="",
                query_str="Can you tell me about your previous work experience?"
            )},
            {"role":"assistant", "content": (
                "<thinking>\n"
                "Step 1: Identify user wants a summary of prior experience. \n"
                "Step 2: Check <context> for job history, titles, companies, duration, key responsibilities.\n"
                "Step 3: Facts are present: context shows candidate worked at ABC Corp as a software engineer for 3 years, led a small team, and improved process efficiency.\n"
                "Step 4: Compose a concise, HR-friendly summary.\n"
                "</thinking>\n\n"
                "<answer>\n"
                "I worked at ABC Corp as a software engineer for three years, where I led a small team and helped improve process efficiency. I really enjoyed collaborating with colleagues and contributing to team success. If you'd like, I can share more details about my specific projects or team contributions.\n"
                "</answer>"
            )},

            {"role": "user", "content": user_prompt},
        ]
        
        return messages
    
    def chat(
        self,
        lang: str,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        character: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a chat query with RAG.

        Args:
            lang: Language ("en" or "zhtw")
            query: User query text
            conversation_history: Previous conversation messages
            session_id: Session ID for conversation persistence
            k: Number of documents to retrieve
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens to generate
            system_prompt: Custom system prompt (optional, overrides character)
            character: Interviewer character ("hr" or "engineer") for prompt selection
            model: Model name to use (optional)
            **kwargs: Additional LLM parameters

        Returns:
            Dictionary with response content, usage, retrieved context info, and trace_id
        """
        self.lang = lang
        if self.lang == "zhtw":
            self.vectorstore = chroma_usage_zhtw
        elif self.lang == "en":
            self.vectorstore = chroma_usage_en

        # Create Langfuse trace for this request
        trace = None
        trace_id = None
        if LANGFUSE_ENABLED and langfuse_client:
            trace = langfuse_client.create_trace(
                name="chat_request",
                session_id=session_id,
                metadata={
                    "lang": lang,
                    "query": query,
                    "k": k,
                    "temperature": temperature,
                    "character": character,
                    "model": model
                },
                tags=["rag", "chat"]
            )
            if trace:
                trace_id = trace.id

        try:
            # ROUTING: Determine which agent should handle this query
            route = "rag"  # Default to RAG
            if ROUTER_AVAILABLE and query_router:
                # Load history for routing decision
                routing_history = None
                if session_id is not None:
                    if hasattr(self._conversation_store, 'load_history'):
                        routing_history = self._conversation_store.load_history(session_id, max_messages=10)
                    elif hasattr(self._conversation_store, 'get_history'):
                        routing_history = self._conversation_store.get_history(session_id)

                route = query_router.route_query(query, routing_history, lang)
                logger.info(f"Query routed to: {route}")

            # Delegate to appropriate agent
            if route == "chat" and ROUTER_AVAILABLE:
                return chat_agent(
                    query=query,
                    session_id=session_id,
                    lang=lang,
                    trace=trace,
                    chat_service=self
                )
            elif route == "memory" and ROUTER_AVAILABLE:
                return memory_agent(
                    query=query,
                    session_id=session_id,
                    lang=lang,
                    trace=trace,
                    chat_service=self
                )

            # If route is "rag" or router not available, continue with RAG pipeline below
            # Add router span to trace
            if trace and LANGFUSE_ENABLED:
                router_span = trace.span(
                    name="router-decision",
                    metadata={
                        "route": route,
                        "router_available": ROUTER_AVAILABLE
                    }
                )
                router_span.end()

        except Exception as routing_error:
            logger.warning(f"Routing failed: {routing_error}. Falling back to RAG agent.")
            route = "rag"

        # Continue with RAG pipeline (this is the RAG agent logic)
        try:
            # Cleanup expired conversations (only for in-memory store)
            if hasattr(self._conversation_store, 'cleanup_expired'):
                self._conversation_store.cleanup_expired()

            # SPAN: Memory Load
            if trace and LANGFUSE_ENABLED:
                memory_span = trace.span(
                    name="memory-load",
                    metadata={"session_id": session_id}
                )

            # Load persisted conversation if session_id is provided
            if session_id is not None and not conversation_history:
                # Support both Redis and in-memory interfaces
                if hasattr(self._conversation_store, 'load_history'):
                    conversation_history = self._conversation_store.load_history(session_id)
                else:
                    conversation_history = self._conversation_store.get_history(session_id)

            if trace and LANGFUSE_ENABLED and memory_span:
                memory_span.end(
                    metadata={
                        "history_length": len(conversation_history) if conversation_history else 0
                    }
                )

            # SPAN: Vector Retrieval
            if trace and LANGFUSE_ENABLED:
                retrieval_span = trace.span(
                    name="vector-retrieval",
                    metadata={
                        "k": k,
                        "lang": lang
                    }
                )

            # Retrieve relevant context
            retrieval_query = self._compose_retrieval_query(query, conversation_history)
            retrieved_docs = self._retrieve_context(retrieval_query, k=k)

            # Extract similarity scores
            similarity_scores = [1.0 - distance for _, _, distance in retrieved_docs]
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

            if trace and LANGFUSE_ENABLED and retrieval_span:
                retrieval_span.end(
                    metadata={
                        "retrieved_count": len(retrieved_docs),
                        "similarity_scores": similarity_scores,
                        "avg_similarity": avg_similarity,
                        "retrieval_query": retrieval_query[:200]  # Truncate for readability
                    }
                )

            context = self._format_context(retrieved_docs)

            if not context:
                if trace and LANGFUSE_ENABLED:
                    trace.update(
                        metadata={"error": "No context retrieved"}
                    )
                return {
                "content": None,
                "usage": None,
                "retrieved_docs_count": 0,
                "context_used": bool(context),
                "trace_id": trace_id,
                "avg_similarity": 0.0,
                "route": "rag"
            }

            # HITL CHECK: Determine if human review is needed
            hitl_triggered = False
            hitl_reason = None
            hitl_review_id = None

            if HITL_AVAILABLE and hitl_service:
                hitl_reason = hitl_service.should_trigger_hitl(
                    similarity_score=avg_similarity,
                    question=query,
                    lang=lang
                )

                if hitl_reason:
                    hitl_triggered = True

                    # Save HITL request to database
                    hitl_review_id = hitl_service.save_hitl_request(
                        session_id=session_id or "unknown",
                        question=query,
                        retrieved_docs=retrieved_docs,
                        similarity_score=avg_similarity,
                        reason=hitl_reason,
                        lang=lang,
                        character=character or "hr"
                    )

                    # Add HITL span to trace
                    if trace and LANGFUSE_ENABLED:
                        hitl_span = trace.span(
                            name="hitl-trigger",
                            metadata={
                                "triggered": True,
                                "reason": hitl_reason,
                                "review_id": hitl_review_id,
                                "avg_similarity": avg_similarity
                            }
                        )
                        hitl_span.end()

                    # Return HITL message instead of proceeding to LLM
                    hitl_message = (
                        "Thank you for your question. This query requires human review for accuracy. "
                        f"Your request has been logged (Review ID: {hitl_review_id}). "
                        "A human expert will review and provide an answer soon."
                    ) if lang == "en" else (
                        "感謝您的提問。此問題需要人工審核以確保準確性。"
                        f"您的請求已記錄(審核編號: {hitl_review_id})。"
                        "專家將盡快審核並提供答案。"
                    )

                    logger.info(f"HITL triggered for session {session_id}: {hitl_reason}")

                    # Update trace
                    if trace and LANGFUSE_ENABLED:
                        trace.update(
                            output=hitl_message,
                            metadata={
                                "hitl_triggered": True,
                                "hitl_reason": hitl_reason,
                                "hitl_review_id": hitl_review_id
                            }
                        )

                    return {
                        "content": hitl_message,
                        "usage": {},
                        "retrieved_docs_count": len(retrieved_docs),
                        "context_used": bool(context),
                        "trace_id": trace_id,
                        "avg_similarity": avg_similarity,
                        "route": "rag",
                        "hitl_triggered": True,
                        "hitl_reason": hitl_reason,
                        "hitl_review_id": hitl_review_id
                    }

            # SPAN: Prompt Construction
            if trace and LANGFUSE_ENABLED:
                prompt_span = trace.span(
                    name="prompt-construction",
                    metadata={
                        "character": character,
                        "has_system_prompt": system_prompt is not None
                    }
                )

            # Get system prompt: use provided one, or get based on character, or use default
            if system_prompt is None and character is not None:
                system_prompt = self.get_system_prompt(character.lower())

            # Build messages
            messages = self._build_messages(
                user_query=query,
                context=context,
                conversation_history=conversation_history,
                system_prompt=system_prompt
            )

            if trace and LANGFUSE_ENABLED and prompt_span:
                prompt_span.end(
                    metadata={
                        "message_count": len(messages),
                        "context_length": len(context)
                    }
                )

            # SPAN: LLM Call
            if trace and LANGFUSE_ENABLED:
                llm_span = trace.span(
                    name="llm-call",
                    metadata={
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )

            # Get LLM response
            response = self.llm_client.get_chat_completion_with_usage(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            response_content = response["content"]
            logger.info(f'chat response: {response_content}')

            if trace and LANGFUSE_ENABLED and llm_span:
                llm_span.end(
                    metadata={
                        "usage": response.get("usage", {}),
                        "response_length": len(response_content)
                    }
                )

            # Parse final answer
            match = re.search(r'<answer>(.*?)</answer>', response_content, re.DOTALL)
            match2 = re.search(r'<answer>(.*)', response_content, re.DOTALL)

            if match:
                final_answer = match.group(1).strip()
                if final_answer in ["None", "Empty Response"]:
                    final_answer = None
            elif match2:
                final_answer = match2.group(1).strip()
                if final_answer in ["None", "Empty Response"]:
                    final_answer = None
            else:
                final_answer = None

            # Persist conversation state if session_id provided
            if session_id is not None and final_answer:
                # Support both Redis and in-memory interfaces
                if hasattr(self._conversation_store, 'save_message'):
                    # Redis interface: save user and assistant messages separately
                    self._conversation_store.save_message(session_id, "user", query)
                    self._conversation_store.save_message(session_id, "assistant", final_answer)
                else:
                    # In-memory interface: append both at once
                    self._conversation_store.append(
                        session_id=session_id,
                        user_message=query,
                        assistant_message=final_answer,
                    )

            # Finalize trace
            if trace and LANGFUSE_ENABLED:
                trace.update(
                    output=final_answer,
                    metadata={
                        "success": True,
                        "answer_parsed": final_answer is not None
                    }
                )

            return {
                "content": final_answer,
                "usage": response.get("usage", {}),
                "retrieved_docs_count": len(retrieved_docs),
                "context_used": bool(context),
                "trace_id": trace_id,
                "avg_similarity": avg_similarity,
                "route": "rag",  # This path is always RAG agent
                "hitl_triggered": False  # HITL was not triggered
            }
        except Exception as e:
            logger.error(f"Error in chat service: {e}", exc_info=True)
            if trace and LANGFUSE_ENABLED:
                trace.update(
                    metadata={"error": str(e), "success": False}
                )
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
        character: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Process a chat query with RAG and stream the response.
        
        Args:
            query: User query text
            conversation_history: Previous conversation messages
            session_id: Session ID for conversation persistence
            k: Number of documents to retrieve
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens to generate
            system_prompt: Custom system prompt (optional, overrides character)
            character: Interviewer character ("hr" or "engineer") for prompt selection
            model: Model name to use (optional)
            **kwargs: Additional LLM parameters
            
        Yields:
            Chunks of the generated response
        """
        try:
            # Cleanup expired conversations (only for in-memory store)
            if hasattr(self._conversation_store, 'cleanup_expired'):
                self._conversation_store.cleanup_expired()

            # Load persisted conversation if session_id is provided
            if session_id is not None and not conversation_history:
                # Support both Redis and in-memory interfaces
                if hasattr(self._conversation_store, 'load_history'):
                    conversation_history = self._conversation_store.load_history(session_id)
                else:
                    conversation_history = self._conversation_store.get_history(session_id)

            # Retrieve relevant context
            retrieval_query = self._compose_retrieval_query(query, conversation_history)
            retrieved_docs = self._retrieve_context(retrieval_query, k=k)
            context = self._format_context(retrieved_docs)
            
            # Get system prompt: use provided one, or get based on character, or use default
            if system_prompt is None and character is not None:
                system_prompt = self.get_system_prompt(character)
            
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
                assistant_message = "".join(accumulated)
                # Support both Redis and in-memory interfaces
                if hasattr(self._conversation_store, 'save_message'):
                    # Redis interface: save user and assistant messages separately
                    self._conversation_store.save_message(session_id, "user", query)
                    self._conversation_store.save_message(session_id, "assistant", assistant_message)
                else:
                    # In-memory interface: append both at once
                    self._conversation_store.append(
                        session_id=session_id,
                        user_message=query,
                        assistant_message=assistant_message,
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
    
    def get_last_session(self) -> Tuple[Optional[str], float]:
        """
        Get the most recent session_id and its last activity time.
        
        Returns:
            Tuple of (session_id, last_activity_time) or (None, 0.0) if no sessions exist
        """
        with self._lock:
            if not self._sessions:
                return None, 0.0
            
            # Find the session with the most recent activity
            most_recent_id = None
            most_recent_time = 0.0
            
            for session_id, session_data in self._sessions.items():
                last_activity = session_data.get("last_activity", 0.0)
                if last_activity > most_recent_time:
                    most_recent_time = last_activity
                    most_recent_id = session_id
            
            return most_recent_id, most_recent_time


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
    # history = [
    #     {"role": "user", "content": "What is the candidate's name?"},
    #     {"role": "assistant", "content": "The candidate's name is John Doe."}
    # ]
    response = chat_service.chat(
        lang="zhtw",
        query="你現在在哪裡工作",
        # conversation_history=history,
        k=3
    )
    print("\nResponse with history:", response["content"])

