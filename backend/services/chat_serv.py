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
import asyncio
from llm import llm_client, embed_client
from db.chroma_vectordb import ChromaUsage
from utils.app_logger import LoggerSetup
from config import prompts

logger = LoggerSetup("ChatService").logger

chroma_usage_en = ChromaUsage(collection_name="chat_cv_en")
chroma_usage_zhtw = ChromaUsage(collection_name="chat_cv_zhtw")
class ChatService:
    """Service for handling chat interactions with RAG (Retrieval Augmented Generation)."""
    
    def __init__(self):
        self.llm = llm_client
        self.embed_client = embed_client
        self.vectorstore = chroma_usage_en
        self._conversation_store = _ConversationStore()

    def clear_history(self, session_id: str) -> bool:
        """Manually clear a single session's history. Returns True if removed."""
        return self._conversation_store.clear(session_id)

    def clear_all_histories(self) -> int:
        """Manually clear all sessions. Returns number of sessions removed."""
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
            return prompts.HR_cot_system_prompt
        
        if character == "engineer":
            return prompts.EM_cot_system_prompt
        else:
            return prompts.HR_cot_system_prompt

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
        
        last_session_id, last_activity_time = self._conversation_store.get_last_session()
        
        if last_session_id is None:
            return str(uuid.uuid4())
        
        now = time.time()
        time_since_last_activity = now - last_activity_time
        
        if time_since_last_activity > timeout_seconds:
            return str(uuid.uuid4())
        else:
            return last_session_id
    
    async def _aretrieve_context(self, query: str, k: int = 5) -> List[tuple]:
        """
        Asynchronously retrieve relevant context from the vectorstore using hybrid search.
        """
        try:
            # Get embedding for the query
            query_embedding = await self.embed_client.embed(query)
            
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
        """
        if not conversation_history:
            return user_query
        
        accumulated: List[str] = []
        char_count = 0
        
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
        """
        if not retrieved_docs:
            return ""
        
        context_parts = [f"[Source: {metadata.get('filename', 'unknown')}]\n{doc}" for doc, metadata, distance in retrieved_docs]
        return "\n\n---\n\n".join(context_parts)
    
    def _build_messages(
        self,
        user_query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        lang: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build messages for the LLM.
        """
        system_prompt = system_prompt or self.get_system_prompt()

        conversation_history_str = ""
        if conversation_history:
            for msg in conversation_history:
                conversation_history_str += f"{msg.get('role')}: {msg.get('content')}\n"

        user_prompt = prompts.cot_user_prompt.format(
            context_str=context,
            history=conversation_history_str,
            query_str=user_query
        )

        # Add explicit language instruction based on lang parameter
        if lang == "zhtw":
            user_prompt += "\n\n<language_instruction>你必須使用繁體中文回答。</language_instruction>"
        elif lang == "en":
            user_prompt += "\n\n<language_instruction>You must respond in English.</language_instruction>"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    async def achat(self, **kwargs) -> Dict[str, Any]:
        """
        Process a chat query with RAG asynchronously.
        """
        self.lang = kwargs.get("lang", "en")
        if self.lang == "zhtw":
            self.vectorstore = chroma_usage_zhtw
        else:
            self.vectorstore = chroma_usage_en

        try:
            self._conversation_store.cleanup_expired()
            session_id = kwargs.get("session_id")
            conversation_history = kwargs.get("conversation_history")

            if session_id and not conversation_history:
                conversation_history = self._conversation_store.get_history(session_id)

            query = kwargs["query"]
            retrieval_query = self._compose_retrieval_query(query, conversation_history)
            retrieved_docs = await self._aretrieve_context(retrieval_query, k=kwargs.get("k", 5))
            context = self._format_context(retrieved_docs)
            
            if not context:
                return {"content": None, "usage": None, "retrieved_docs_count": 0, "context_used": False}

            system_prompt = kwargs.get("system_prompt") or self.get_system_prompt(kwargs.get("character"))
            
            messages = self._build_messages(
                user_query=query,
                context=context,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
                lang=self.lang
            )

            response = await self.llm.chat(messages=messages, **kwargs)
            response_content = response.get("content", "")
            
            match = re.search(r'<answer>(.*?)</answer>', response_content, re.DOTALL)
            final_answer = match.group(1).strip() if match else None
            
            if session_id and final_answer:
                self._conversation_store.append(session_id, query, final_answer)

            return {
                "content": final_answer,
                "usage": response.get("usage", {}),
                "retrieved_docs_count": len(retrieved_docs),
                "context_used": bool(context)
            }
        except Exception as e:
            logger.error(f"Error in async chat service: {e}", exc_info=True)
            raise
    
    async def astream_chat(self, **kwargs):
        """
        Process a chat query with RAG and stream the response asynchronously.
        """
        self.lang = kwargs.get("lang", "en")
        if self.lang == "zhtw":
            self.vectorstore = chroma_usage_zhtw
        else:
            self.vectorstore = chroma_usage_en

        try:
            self._conversation_store.cleanup_expired()
            session_id = kwargs.get("session_id")
            conversation_history = kwargs.get("conversation_history")

            if session_id and not conversation_history:
                conversation_history = self._conversation_store.get_history(session_id)
            
            query = kwargs["query"]
            retrieval_query = self._compose_retrieval_query(query, conversation_history)
            retrieved_docs = await self._aretrieve_context(retrieval_query, k=kwargs.get("k", 5))
            context = self._format_context(retrieved_docs)
            
            system_prompt = kwargs.get("system_prompt") or self.get_system_prompt(kwargs.get("character"))
            
            messages = self._build_messages(
                user_query=query,
                context=context,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
                lang=self.lang
            )

            buffer = ""
            final_answer = ""
            start_tag_checked = False  # Whether we've determined if <answer> exists
            is_answer_ended = False

            logger.info("Starting LLM stream...")
            chunk_count = 0
            async for chunk in self.llm.stream(messages=messages):
                chunk_count += 1
                if chunk_count <= 3:
                    logger.info(f"Received chunk {chunk_count}: {chunk[:50] if chunk else 'empty'}...")
                if is_answer_ended:
                    continue

                buffer += chunk

                # Phase 1: Determine if <answer> tag exists
                if not start_tag_checked:
                    print(buffer)
                    start_match = re.search(r'<answer>', buffer)
                    if start_match:
                        start_tag_checked = True
                        # Skip <answer> tag, discard content before it
                        buffer = buffer[start_match.end():]
                    elif len(buffer) > 8:
                        # No <answer> tag found after sufficient content, output directly
                        start_tag_checked = True
                    else:
                        # Not enough content to determine yet
                        continue

                # Phase 2: Output content while checking for </answer>
                end_match = re.search(r'</answer>', buffer)
                if end_match:
                    content = buffer[:end_match.start()]
                    if content:
                        yield content
                        final_answer += content
                    is_answer_ended = True
                else:
                    # Keep last 9 chars for potential incomplete '</answer>'
                    safe_length = len(buffer) - 9
                    if safe_length > 0:
                        content = buffer[:safe_length]
                        yield content
                        final_answer += content
                        buffer = buffer[safe_length:]

            # Output remaining buffer if no </answer> was found
            if not is_answer_ended and buffer:
                yield buffer
                final_answer += buffer

            logger.info(f"LLM stream completed. Total chunks: {chunk_count}, final_answer length: {len(final_answer)}")

            if session_id and final_answer:
                self._conversation_store.append(session_id, query, final_answer)
        except Exception as e:
            logger.error(f"Error in async stream chat service: {e}", exc_info=True)
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
            return list(session["messages"])

    def append(self, session_id: str, user_message: str, assistant_message: str) -> None:
        with self._lock:
            now = time.time()
            session = self._sessions.setdefault(session_id, {"messages": [], "last_activity": now})
            session["messages"].extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ])
            session["last_activity"] = now

    def cleanup_expired(self) -> None:
        with self._lock:
            now = time.time()
            expired_sids = [sid for sid, s in self._sessions.items() if now - s.get("last_activity", 0) > self._idle_timeout]
            for sid in expired_sids:
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
        with self._lock:
            if not self._sessions:
                return None, 0.0
            
            most_recent_id, most_recent_time = max(self._sessions.items(), key=lambda item: item[1].get("last_activity", 0.0))
            return most_recent_id, most_recent_time


if __name__ == "__main__":
    chat_service = ChatService()
    
    async def main():
        response = await chat_service.achat(
            lang="zhtw",
            query="你現在在哪裡工作",
            k=3
        )
        print("\nResponse with history:", response["content"])

    asyncio.run(main())
