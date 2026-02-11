"""
Agent Implementations

Three specialized agents for different query types:
1. RAG Agent - Document-based queries about CV content
2. Chat Agent - Greetings and casual conversation
3. Memory Agent - Questions about previous conversation
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def rag_agent(
    query: str,
    session_id: str,
    lang: str,
    k: int = 5,
    temperature: float = 0.7,
    character: Optional[str] = None,
    model: Optional[str] = None,
    trace: Optional[Any] = None,
    chat_service: Optional[Any] = None
) -> Dict[str, Any]:
    """
    RAG Agent - Handles document-based queries using retrieval-augmented generation.

    This is the main agent for answering questions about CV content.
    It retrieves relevant context from the vector store and generates answers.

    Args:
        query: User query
        session_id: Session identifier
        lang: Language ("en" or "zhtw")
        k: Number of documents to retrieve
        temperature: LLM temperature
        character: Interviewer character ("hr" or "engineer")
        model: Model name (optional)
        trace: Langfuse trace object (optional)
        chat_service: ChatService instance (optional, for dependency injection)

    Returns:
        Response dictionary with content, usage, and metadata
    """
    logger.info(f"RAG agent processing query: {query[:50]}...")

    # Import chat_service here to avoid circular imports
    if chat_service is None:
        from services import chat_service as default_chat_service
        chat_service = default_chat_service

    # Add router span to trace if available
    if trace:
        try:
            router_span = trace.span(
                name="router-decision",
                metadata={
                    "route": "rag",
                    "reason": "Document-based query about CV content"
                }
            )
            router_span.end()
        except Exception as e:
            logger.warning(f"Failed to add router span: {e}")

    # Use the standard chat method (which includes RAG pipeline)
    response = chat_service.chat(
        lang=lang,
        query=query,
        session_id=session_id,
        k=k,
        temperature=temperature,
        character=character,
        model=model
    )

    # Add route information to response
    response["route"] = "rag"
    return response


def chat_agent(
    query: str,
    session_id: str,
    lang: str = "en",
    trace: Optional[Any] = None,
    chat_service: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Chat Agent - Handles greetings and casual conversation.

    Provides friendly responses to greetings without querying the vector store.
    Keeps the conversation natural and engaging.

    Args:
        query: User query (typically a greeting)
        session_id: Session identifier
        lang: Language ("en" or "zhtw")
        trace: Langfuse trace object (optional)
        chat_service: ChatService instance (optional)

    Returns:
        Response dictionary with content and metadata
    """
    logger.info(f"Chat agent processing query: {query[:50]}...")

    # Add router span to trace if available
    if trace:
        try:
            router_span = trace.span(
                name="router-decision",
                metadata={
                    "route": "chat",
                    "reason": "Greeting or casual conversation"
                }
            )
            router_span.end()
        except Exception as e:
            logger.warning(f"Failed to add router span: {e}")

    # Predefined friendly responses for greetings
    responses_en = {
        "hi": "Hello! I'm here to help you learn about this candidate's background and experience. What would you like to know?",
        "hello": "Hi there! Feel free to ask me anything about the candidate's skills, experience, or education.",
        "hey": "Hey! I'm ready to answer your questions about this CV. What would you like to explore?",
        "thanks": "You're welcome! Is there anything else you'd like to know about the candidate?",
        "thank you": "Happy to help! Feel free to ask more questions about the candidate's background.",
        "goodbye": "Goodbye! Feel free to come back if you have more questions.",
        "bye": "See you later! Don't hesitate to return if you need more information."
    }

    responses_zhtw = {
        "你好": "您好!我可以協助您了解這位候選人的背景和經驗。請問您想了解什麼?",
        "嗨": "嗨!歡迎詢問候選人的技能、經驗或教育背景相關問題。",
        "謝謝": "不客氣!還有什麼想了解關於候選人的資訊嗎?",
        "再見": "再見!如果之後還有問題,歡迎隨時回來詢問。"
    }

    # Select response based on language
    responses = responses_zhtw if lang == "zhtw" else responses_en

    # Find matching response
    query_lower = query.lower().strip()
    content = None

    for trigger, response in responses.items():
        if trigger in query_lower or query_lower in trigger:
            content = response
            break

    # Default response if no match
    if content is None:
        content = (
            "Hello! I'm here to help you learn about this candidate. "
            "Feel free to ask about their experience, skills, education, or any other aspect of their background."
        ) if lang == "en" else (
            "您好!我可以協助您了解這位候選人的背景。"
            "歡迎詢問他們的經驗、技能、教育或其他相關資訊。"
        )

    # Save to memory if we have chat_service
    if chat_service and session_id:
        try:
            # Import memory store
            if hasattr(chat_service, '_conversation_store'):
                store = chat_service._conversation_store
                if hasattr(store, 'save_message'):
                    store.save_message(session_id, "user", query)
                    store.save_message(session_id, "assistant", content)
        except Exception as e:
            logger.warning(f"Failed to save chat to memory: {e}")

    return {
        "content": content,
        "route": "chat",
        "usage": {},
        "retrieved_docs_count": 0,
        "context_used": False,
        "trace_id": trace.id if trace else None
    }


def memory_agent(
    query: str,
    session_id: str,
    lang: str = "en",
    trace: Optional[Any] = None,
    chat_service: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Memory Agent - Handles queries about previous conversation.

    Retrieves and summarizes information from the conversation history
    without querying the vector store.

    Args:
        query: User query (asking about previous conversation)
        session_id: Session identifier
        lang: Language ("en" or "zhtw")
        trace: Langfuse trace object (optional)
        chat_service: ChatService instance (optional)

    Returns:
        Response dictionary with content and metadata
    """
    logger.info(f"Memory agent processing query: {query[:50]}...")

    # Import chat_service here to avoid circular imports
    if chat_service is None:
        from services import chat_service as default_chat_service
        chat_service = default_chat_service

    # Add router span to trace if available
    if trace:
        try:
            router_span = trace.span(
                name="router-decision",
                metadata={
                    "route": "memory",
                    "reason": "Query about previous conversation"
                }
            )
            router_span.end()
        except Exception as e:
            logger.warning(f"Failed to add router span: {e}")

    # Retrieve conversation history
    history = []
    if hasattr(chat_service, '_conversation_store'):
        store = chat_service._conversation_store
        if hasattr(store, 'load_history'):
            history = store.load_history(session_id, max_messages=20)
        elif hasattr(store, 'get_history'):
            history = store.get_history(session_id)

    # Generate response based on history
    if not history:
        content = (
            "I don't have any record of our previous conversation. "
            "This might be a new session. What would you like to know about the candidate?"
        ) if lang == "en" else (
            "我沒有找到我們之前的對話記錄。這可能是一個新的對話。"
            "您想了解候選人的什麼資訊?"
        )
    else:
        # Format recent history for user
        if lang == "en":
            content = "Here's a summary of our recent conversation:\n\n"
        else:
            content = "以下是我們最近的對話摘要:\n\n"

        # Show last 5 exchanges
        recent = history[-10:]  # Last 5 exchanges (10 messages)
        for msg in recent:
            role = msg.get("role", "")
            text = msg.get("content", "")
            if role == "user":
                prefix = "You asked: " if lang == "en" else "您問: "
            else:
                prefix = "I answered: " if lang == "en" else "我回答: "

            # Truncate long messages
            if len(text) > 200:
                text = text[:200] + "..."

            content += f"{prefix}{text}\n\n"

    # Save interaction to memory
    if chat_service and session_id:
        try:
            store = chat_service._conversation_store
            if hasattr(store, 'save_message'):
                store.save_message(session_id, "user", query)
                store.save_message(session_id, "assistant", content)
        except Exception as e:
            logger.warning(f"Failed to save memory query: {e}")

    return {
        "content": content,
        "route": "memory",
        "usage": {},
        "retrieved_docs_count": 0,
        "context_used": False,
        "history_length": len(history),
        "trace_id": trace.id if trace else None
    }


# Export agents
__all__ = [
    'rag_agent',
    'chat_agent',
    'memory_agent'
]
