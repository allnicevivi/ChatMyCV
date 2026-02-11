"""
HITL (Human-In-The-Loop) Service

Provides decision gate logic for queries requiring human review.
Integrates with PostgreSQL for audit trail and review workflow.

Triggers HITL when:
1. Average similarity score is below threshold (low confidence)
2. Query contains sensitive/risky keywords
3. LLM indicates uncertainty
"""

import json
import logging
import os
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Lazy import psycopg2 to allow graceful degradation
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    logger.warning("psycopg2 not installed. HITL features will be disabled.")
    POSTGRES_AVAILABLE = False
    psycopg2 = None


class HITLService:
    """
    Human-In-The-Loop decision and audit service.

    Determines when queries need human review and logs them to PostgreSQL.
    """

    # Similarity threshold for triggering HITL (0.0 to 1.0)
    SIMILARITY_THRESHOLD = 0.5

    # Risky keywords that might require human review
    RISKY_KEYWORDS_ZH = [
        "理賠", "條款", "金額", "保單", "法律", "責任",
        "賠償", "訴訟", "合約", "違約", "罰款", "爭議",
        "機密", "敏感", "隱私"
    ]

    RISKY_KEYWORDS_EN = [
        "claim", "policy", "amount", "legal", "liability", "contract",
        "compensation", "lawsuit", "agreement", "breach", "penalty",
        "dispute", "confidential", "sensitive", "privacy"
    ]

    # Uncertainty indicators
    UNCERTAINTY_KEYWORDS_ZH = [
        "不確定", "可能", "也許", "大概", "不清楚", "不知道"
    ]

    UNCERTAINTY_KEYWORDS_EN = [
        "unsure", "maybe", "might", "probably", "unclear", "don't know",
        "not certain", "not sure"
    ]

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize HITL service with PostgreSQL connection.

        Args:
            host: PostgreSQL host (defaults to env POSTGRES_HOST)
            port: PostgreSQL port (defaults to env POSTGRES_PORT)
            database: Database name (defaults to env POSTGRES_DB)
            user: Database user (defaults to env POSTGRES_USER)
            password: Database password (defaults to env POSTGRES_PASSWORD)
        """
        self._conn = None
        self._use_fallback = False

        if not POSTGRES_AVAILABLE:
            logger.warning("PostgreSQL not available. HITL features will be disabled.")
            self._use_fallback = True
            return

        # Get PostgreSQL configuration
        pg_host = host or os.getenv("POSTGRES_HOST", "localhost")
        pg_port = port or int(os.getenv("POSTGRES_PORT", "5433"))
        pg_db = database or os.getenv("POSTGRES_DB", "chatmycv")
        pg_user = user or os.getenv("POSTGRES_USER", "chatmycv_user")
        pg_pass = password or os.getenv("POSTGRES_PASSWORD", "chatmycv_pass")

        try:
            self._conn = psycopg2.connect(
                host=pg_host,
                port=pg_port,
                database=pg_db,
                user=pg_user,
                password=pg_pass,
                connect_timeout=5
            )
            logger.info(f"HITL service initialized successfully. Database: {pg_db}")

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}. HITL features will be disabled.")
            self._conn = None
            self._use_fallback = True

    def should_trigger_hitl(
        self,
        similarity_score: float,
        question: str,
        lang: str = "en",
        response_content: Optional[str] = None
    ) -> Optional[str]:
        """
        Determine if HITL should be triggered for a query.

        Args:
            similarity_score: Average similarity score from vector retrieval (0.0 to 1.0)
            question: User question
            lang: Language ("en" or "zhtw")
            response_content: LLM response content (optional, for uncertainty detection)

        Returns:
            Reason string if HITL should trigger, None otherwise
        """
        reasons = []

        # Check 1: Low similarity score
        if similarity_score < self.SIMILARITY_THRESHOLD:
            reasons.append(f"Low similarity score: {similarity_score:.3f} < {self.SIMILARITY_THRESHOLD}")

        # Check 2: Risky keywords in question
        risky_keywords = self.RISKY_KEYWORDS_ZH if lang == "zhtw" else self.RISKY_KEYWORDS_EN
        question_lower = question.lower()

        for keyword in risky_keywords:
            if lang == "en":
                if re.search(rf'\b{re.escape(keyword)}\b', question_lower):
                    reasons.append(f"Risky keyword detected: {keyword}")
                    break
            else:
                if keyword in question:
                    reasons.append(f"Risky keyword detected: {keyword}")
                    break

        # Check 3: Uncertainty in response
        if response_content:
            uncertainty_keywords = self.UNCERTAINTY_KEYWORDS_ZH if lang == "zhtw" else self.UNCERTAINTY_KEYWORDS_EN
            response_lower = response_content.lower()

            for keyword in uncertainty_keywords:
                if lang == "en":
                    if re.search(rf'\b{re.escape(keyword)}\b', response_lower):
                        reasons.append(f"Uncertainty detected in response: {keyword}")
                        break
                else:
                    if keyword in response_content:
                        reasons.append(f"Uncertainty detected in response: {keyword}")
                        break

        # Return combined reason if any triggers matched
        if reasons:
            return "; ".join(reasons)

        return None

    def save_hitl_request(
        self,
        session_id: str,
        question: str,
        retrieved_docs: List[Tuple[str, Dict, float]],
        similarity_score: float,
        reason: str,
        lang: str = "en",
        character: str = "hr"
    ) -> Optional[int]:
        """
        Save a HITL request to the database.

        Args:
            session_id: Session identifier
            question: User question
            retrieved_docs: List of retrieved documents (doc, metadata, distance)
            similarity_score: Average similarity score
            reason: Why HITL was triggered
            lang: Language ("en" or "zhtw")
            character: Interviewer character ("hr" or "engineer")

        Returns:
            Review ID if successful, None otherwise
        """
        if self._use_fallback:
            logger.warning("HITL save skipped - PostgreSQL not available")
            return None

        try:
            # Format retrieved docs as JSONB
            docs_json = []
            for doc, metadata, distance in retrieved_docs:
                docs_json.append({
                    "content": doc[:500],  # Truncate for storage
                    "metadata": metadata,
                    "distance": distance,
                    "similarity": 1.0 - distance
                })

            # Insert into database
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO hitl_reviews
                    (session_id, question, retrieved_docs, similarity_score, reason, lang, character, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
                    RETURNING id
                """, (
                    session_id,
                    question,
                    json.dumps(docs_json),
                    similarity_score,
                    reason,
                    lang,
                    character
                ))

                review_id = cur.fetchone()[0]
                self._conn.commit()

                logger.info(f"HITL request saved with ID: {review_id}")
                return review_id

        except Exception as e:
            logger.error(f"Error saving HITL request: {e}")
            if self._conn:
                self._conn.rollback()
            return None

    def get_pending_reviews(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve pending HITL reviews.

        Args:
            limit: Maximum number of reviews to retrieve

        Returns:
            List of review dictionaries
        """
        if self._use_fallback:
            return []

        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        id, session_id, question, retrieved_docs, similarity_score,
                        reason, lang, character, created_at
                    FROM hitl_reviews
                    WHERE status = 'pending'
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))

                reviews = cur.fetchall()
                return [dict(review) for review in reviews]

        except Exception as e:
            logger.error(f"Error retrieving pending reviews: {e}")
            return []

    def approve_review(
        self,
        review_id: int,
        final_answer: str,
        reviewer: str = "human_reviewer"
    ) -> bool:
        """
        Approve a HITL review and provide the final answer.

        Args:
            review_id: Review ID to approve
            final_answer: Human-provided answer
            reviewer: Reviewer identifier

        Returns:
            True if successful, False otherwise
        """
        if self._use_fallback:
            logger.warning("HITL approve skipped - PostgreSQL not available")
            return False

        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    UPDATE hitl_reviews
                    SET status = 'approved',
                        final_answer = %s,
                        reviewer = %s,
                        reviewed_at = NOW()
                    WHERE id = %s
                """, (final_answer, reviewer, review_id))

                self._conn.commit()
                logger.info(f"HITL review {review_id} approved by {reviewer}")
                return True

        except Exception as e:
            logger.error(f"Error approving HITL review: {e}")
            if self._conn:
                self._conn.rollback()
            return False

    def reject_review(
        self,
        review_id: int,
        reviewer: str = "human_reviewer",
        reason: Optional[str] = None
    ) -> bool:
        """
        Reject a HITL review.

        Args:
            review_id: Review ID to reject
            reviewer: Reviewer identifier
            reason: Optional rejection reason

        Returns:
            True if successful, False otherwise
        """
        if self._use_fallback:
            logger.warning("HITL reject skipped - PostgreSQL not available")
            return False

        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    UPDATE hitl_reviews
                    SET status = 'rejected',
                        reviewer = %s,
                        reviewed_at = NOW(),
                        final_answer = %s
                    WHERE id = %s
                """, (reviewer, reason, review_id))

                self._conn.commit()
                logger.info(f"HITL review {review_id} rejected by {reviewer}")
                return True

        except Exception as e:
            logger.error(f"Error rejecting HITL review: {e}")
            if self._conn:
                self._conn.rollback()
            return False

    def get_review_by_id(self, review_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific review by ID"""
        if self._use_fallback:
            return None

        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM hitl_reviews WHERE id = %s
                """, (review_id,))

                review = cur.fetchone()
                return dict(review) if review else None

        except Exception as e:
            logger.error(f"Error retrieving review {review_id}: {e}")
            return None

    @property
    def is_connected(self) -> bool:
        """Check if PostgreSQL is connected"""
        if self._use_fallback or not self._conn:
            return False

        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except:
            return False

    def __del__(self):
        """Close database connection on cleanup"""
        if self._conn:
            try:
                self._conn.close()
            except:
                pass


# Singleton instance
hitl_service = HITLService()


# Export key components
__all__ = [
    'HITLService',
    'hitl_service',
    'POSTGRES_AVAILABLE'
]
