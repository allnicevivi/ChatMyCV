"""
Multi-Agent Router

Rule-based routing system that directs queries to appropriate agents:
- RAG Agent: Document-based questions about CV content
- Chat Agent: Greetings and general conversation
- Memory Agent: Questions about previous conversation

Supports both English and Traditional Chinese queries.
"""

import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Rule-based router for multi-agent orchestration.

    Routes queries based on keyword matching and pattern detection,
    supporting bilingual (English and Traditional Chinese) queries.
    """

    # Memory agent triggers (asking about previous conversation)
    MEMORY_TRIGGERS_ZH = [
        "剛剛", "之前", "我說過", "上次", "前面", "先前",
        "剛才", "早些時候", "方才", "你說過", "你提到",
        "我問過", "我們討論", "我們談到", "我剛", "稍早"
    ]

    MEMORY_TRIGGERS_EN = [
        "earlier", "previously", "before", "just now", "what did i",
        "i said", "you said", "you mentioned", "we discussed",
        "we talked about", "i asked", "last time", "ago"
    ]

    # Chat agent triggers (greetings and casual conversation)
    CHAT_TRIGGERS_ZH = [
        "你好", "嗨", "哈囉", "您好", "早安", "午安", "晚安",
        "嘿", "hi", "hello", "謝謝", "感謝", "再見", "掰掰"
    ]

    CHAT_TRIGGERS_EN = [
        "hi", "hello", "hey", "greetings", "good morning",
        "good afternoon", "good evening", "thanks", "thank you",
        "goodbye", "bye", "see you"
    ]

    # RAG-specific keywords (strongly indicate document-based queries)
    RAG_KEYWORDS_ZH = [
        "經驗", "工作", "專案", "技能", "學歷", "教育",
        "公司", "職位", "負責", "成就", "能力", "證書",
        "背景", "任職", "參與", "開發", "設計", "實作"
    ]

    RAG_KEYWORDS_EN = [
        "experience", "work", "project", "skill", "education",
        "company", "position", "role", "achievement", "ability",
        "background", "responsibility", "involvement", "development",
        "design", "implementation", "certificate", "degree"
    ]

    def __init__(self):
        """Initialize the query router"""
        logger.info("Query router initialized with bilingual support")

    def route_query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        lang: str = "en"
    ) -> str:
        """
        Route a query to the appropriate agent based on content analysis.

        Routing logic:
        1. Check for memory triggers (references to past conversation)
        2. Check for chat/greeting triggers (casual conversation)
        3. Check for RAG keywords (document-specific queries)
        4. Default to RAG agent if no clear match

        Args:
            query: User query text
            history: Conversation history (optional, used for context)
            lang: Language code ("en" or "zhtw")

        Returns:
            Agent name: "rag", "chat", or "memory"
        """
        query_lower = query.lower()
        query_original = query  # Keep original case for Chinese matching

        # Determine which trigger sets to use based on language
        if lang == "zhtw":
            memory_triggers = self.MEMORY_TRIGGERS_ZH
            chat_triggers = self.CHAT_TRIGGERS_ZH
            rag_keywords = self.RAG_KEYWORDS_ZH
        else:
            memory_triggers = self.MEMORY_TRIGGERS_EN
            chat_triggers = self.CHAT_TRIGGERS_EN
            rag_keywords = self.RAG_KEYWORDS_EN

        # Rule 1: Memory Agent
        # Check for references to previous conversation
        for trigger in memory_triggers:
            # Case-insensitive for English, case-sensitive for Chinese
            if lang == "en":
                if trigger.lower() in query_lower:
                    logger.info(f"Routed to MEMORY agent (trigger: {trigger})")
                    return "memory"
            else:
                if trigger in query_original:
                    logger.info(f"Routed to MEMORY agent (trigger: {trigger})")
                    return "memory"

        # Additional memory detection: "what did I..." pattern
        if lang == "en" and re.search(r'\bwhat (did|do) (i|we)\b', query_lower):
            logger.info("Routed to MEMORY agent (pattern: what did I/we)")
            return "memory"

        # Rule 2: Chat Agent
        # Check for greetings and casual conversation
        # Only trigger if query is short (< 50 chars) to avoid false positives
        if len(query) < 50:
            for trigger in chat_triggers:
                if lang == "en":
                    # Use word boundaries for English to avoid partial matches
                    if re.search(rf'\b{re.escape(trigger)}\b', query_lower):
                        logger.info(f"Routed to CHAT agent (trigger: {trigger})")
                        return "chat"
                else:
                    if trigger in query_original:
                        logger.info(f"Routed to CHAT agent (trigger: {trigger})")
                        return "chat"

        # Rule 3: RAG Agent (Strong Indicators)
        # Check for CV-specific keywords
        rag_keyword_count = 0
        for keyword in rag_keywords:
            if lang == "en":
                if re.search(rf'\b{re.escape(keyword)}\b', query_lower):
                    rag_keyword_count += 1
            else:
                if keyword in query_original:
                    rag_keyword_count += 1

        # If multiple RAG keywords present, definitely route to RAG
        if rag_keyword_count >= 2:
            logger.info(f"Routed to RAG agent (keyword count: {rag_keyword_count})")
            return "rag"

        # Rule 4: Default to RAG
        # If no clear chat/memory trigger, assume it's a document-based question
        # This is the safe default for a CV chatbot
        logger.info("Routed to RAG agent (default)")
        return "rag"

    def get_route_explanation(
        self,
        query: str,
        route: str,
        lang: str = "en"
    ) -> str:
        """
        Get a human-readable explanation of why a query was routed to a specific agent.

        Args:
            query: User query
            route: Route decision ("rag", "chat", "memory")
            lang: Language code

        Returns:
            Explanation string
        """
        explanations = {
            "en": {
                "memory": "This question asks about our previous conversation.",
                "chat": "This appears to be a greeting or casual conversation.",
                "rag": "This question is about the CV content."
            },
            "zhtw": {
                "memory": "這個問題詢問我們之前的對話內容。",
                "chat": "這似乎是問候或閒聊。",
                "rag": "這個問題關於履歷內容。"
            }
        }

        return explanations.get(lang, explanations["en"]).get(route, "Unknown route")


# Singleton instance
query_router = QueryRouter()


# Export key components
__all__ = [
    'QueryRouter',
    'query_router'
]
