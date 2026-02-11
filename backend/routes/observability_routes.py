"""
Observability Routes

API endpoints for accessing Langfuse traces and system statistics.
Provides visibility into RAG pipeline performance and debugging information.
"""

import sys
sys.path.append("./")
sys.path.append("../")

from flask import Blueprint, request, jsonify
from observability.langfuse_client import langfuse_client, LANGFUSE_AVAILABLE
from services.memory_serv import redis_memory_store, REDIS_AVAILABLE
from services.hitl_serv import hitl_service, POSTGRES_AVAILABLE
from utils.app_logger import LoggerSetup

logger = LoggerSetup("Observability_Routes").logger

observability_bp = Blueprint("observability", __name__)


@observability_bp.get("/health")
def health_check():
    """
    Comprehensive health check for all LLMOps components.

    Returns:
        JSON object with status of each component
    """
    try:
        health_status = {
            "langfuse": {
                "available": LANGFUSE_AVAILABLE,
                "enabled": langfuse_client.is_enabled if langfuse_client else False
            },
            "redis": {
                "available": REDIS_AVAILABLE,
                "connected": redis_memory_store.is_redis_connected if redis_memory_store else False
            },
            "postgres": {
                "available": POSTGRES_AVAILABLE,
                "connected": hitl_service.is_connected if hitl_service else False
            }
        }

        # Overall status
        all_healthy = (
            health_status["langfuse"]["enabled"] and
            health_status["redis"]["connected"] and
            health_status["postgres"]["connected"]
        )

        return jsonify({
            "status": "healthy" if all_healthy else "degraded",
            "components": health_status
        }), 200 if all_healthy else 503

    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@observability_bp.get("/stats")
def get_stats():
    """
    Get system statistics and performance metrics.

    Returns:
        JSON object with statistics
    """
    try:
        stats = {
            "llmops_enabled": LANGFUSE_AVAILABLE and langfuse_client.is_enabled if langfuse_client else False,
            "redis_enabled": REDIS_AVAILABLE and redis_memory_store.is_redis_connected if redis_memory_store else False,
            "hitl_enabled": POSTGRES_AVAILABLE and hitl_service.is_connected if hitl_service else False
        }

        # Get HITL stats if available
        if stats["hitl_enabled"]:
            try:
                pending_reviews = hitl_service.get_pending_reviews(limit=1000)
                stats["hitl_pending_count"] = len(pending_reviews)
            except Exception as e:
                logger.warning(f"Failed to get HITL stats: {e}")
                stats["hitl_pending_count"] = 0

        return jsonify({
            "status": "success",
            "stats": stats
        }), 200

    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@observability_bp.get("/traces")
def list_traces():
    """
    List recent Langfuse traces.

    Note: This is a placeholder endpoint. In production, you would
    query the Langfuse API or database directly for trace information.

    Returns:
        JSON with trace information
    """
    try:
        if not LANGFUSE_AVAILABLE or not langfuse_client.is_enabled:
            return jsonify({
                "status": "unavailable",
                "message": "Langfuse is not enabled or configured"
            }), 503

        # Note: Langfuse Python SDK doesn't provide direct trace listing
        # In production, you would use the Langfuse REST API or web UI
        # For now, return a helpful message
        return jsonify({
            "status": "success",
            "message": "Please use the Langfuse web UI to view traces",
            "langfuse_host": langfuse_client.client.host if langfuse_client and langfuse_client.client else None,
            "note": "Traces are automatically sent to Langfuse with each chat request. Check the trace_id in chat responses."
        }), 200

    except Exception as e:
        logger.error(f"Error in list traces endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@observability_bp.get("/traces/<trace_id>")
def get_trace(trace_id: str):
    """
    Get details of a specific trace.

    Path parameters:
        trace_id: Trace identifier

    Returns:
        JSON with trace details
    """
    try:
        if not LANGFUSE_AVAILABLE or not langfuse_client.is_enabled:
            return jsonify({
                "status": "unavailable",
                "message": "Langfuse is not enabled or configured"
            }), 503

        # Note: Use Langfuse web UI or REST API for full trace details
        return jsonify({
            "status": "success",
            "trace_id": trace_id,
            "message": "Please use the Langfuse web UI to view trace details",
            "langfuse_url": f"{langfuse_client.client.host}/trace/{trace_id}" if langfuse_client and langfuse_client.client else None
        }), 200

    except Exception as e:
        logger.error(f"Error in get trace endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@observability_bp.post("/flush")
def flush_traces():
    """
    Manually flush pending traces to Langfuse.

    This forces immediate sending of any buffered traces.

    Returns:
        Success status
    """
    try:
        if not LANGFUSE_AVAILABLE or not langfuse_client.is_enabled:
            return jsonify({
                "status": "unavailable",
                "message": "Langfuse is not enabled or configured"
            }), 503

        langfuse_client.flush()

        return jsonify({
            "status": "success",
            "message": "Traces flushed to Langfuse"
        }), 200

    except Exception as e:
        logger.error(f"Error in flush endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500
