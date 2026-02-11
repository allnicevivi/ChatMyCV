"""
HITL (Human-In-The-Loop) Routes

API endpoints for managing HITL review workflow:
- GET /hitl/pending - List pending reviews
- POST /hitl/{id}/approve - Approve a review with answer
- POST /hitl/{id}/reject - Reject a review
- GET /hitl/{id} - Get review details
"""

import sys
sys.path.append("./")
sys.path.append("../")

from flask import Blueprint, request, jsonify
from services.hitl_serv import hitl_service
from utils.app_logger import LoggerSetup

logger = LoggerSetup("HITL_Routes").logger

hitl_bp = Blueprint("hitl", __name__)


@hitl_bp.get("/pending")
def get_pending_reviews():
    """
    List all pending HITL reviews.

    Query parameters:
        limit: Maximum number of reviews to return (default 50)

    Returns:
        JSON array of pending reviews
    """
    try:
        limit = request.args.get("limit", 50, type=int)

        if limit < 1 or limit > 200:
            return jsonify({
                "status": "failed",
                "error": "Limit must be between 1 and 200"
            }), 400

        reviews = hitl_service.get_pending_reviews(limit=limit)

        return jsonify({
            "status": "success",
            "count": len(reviews),
            "reviews": reviews
        }), 200

    except Exception as e:
        logger.error(f"Error in get pending reviews endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@hitl_bp.get("/<int:review_id>")
def get_review(review_id: int):
    """
    Get details of a specific HITL review.

    Path parameters:
        review_id: Review ID

    Returns:
        JSON object with review details
    """
    try:
        review = hitl_service.get_review_by_id(review_id)

        if not review:
            return jsonify({
                "status": "failed",
                "error": f"Review {review_id} not found"
            }), 404

        return jsonify({
            "status": "success",
            "review": review
        }), 200

    except Exception as e:
        logger.error(f"Error in get review endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@hitl_bp.post("/<int:review_id>/approve")
def approve_review(review_id: int):
    """
    Approve a HITL review and provide the final answer.

    Path parameters:
        review_id: Review ID to approve

    Expected JSON body:
    {
        "answer": "The approved answer to send to the user",
        "reviewer": "reviewer_name" (optional, defaults to "human_reviewer")
    }

    Returns:
        Success status
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "failed",
                "error": "No JSON data provided"
            }), 400

        answer = data.get("answer")
        reviewer = data.get("reviewer", "human_reviewer")

        if not answer:
            return jsonify({
                "status": "failed",
                "error": "Answer is required"
            }), 400

        # Check if review exists
        review = hitl_service.get_review_by_id(review_id)
        if not review:
            return jsonify({
                "status": "failed",
                "error": f"Review {review_id} not found"
            }), 404

        # Approve the review
        success = hitl_service.approve_review(
            review_id=review_id,
            final_answer=answer,
            reviewer=reviewer
        )

        if not success:
            return jsonify({
                "status": "failed",
                "error": "Failed to approve review"
            }), 500

        logger.info(f"Review {review_id} approved by {reviewer}")

        return jsonify({
            "status": "success",
            "review_id": review_id,
            "message": "Review approved successfully"
        }), 200

    except Exception as e:
        logger.error(f"Error in approve review endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@hitl_bp.post("/<int:review_id>/reject")
def reject_review(review_id: int):
    """
    Reject a HITL review.

    Path parameters:
        review_id: Review ID to reject

    Expected JSON body:
    {
        "reason": "Rejection reason" (optional),
        "reviewer": "reviewer_name" (optional, defaults to "human_reviewer")
    }

    Returns:
        Success status
    """
    try:
        data = request.get_json() or {}

        reviewer = data.get("reviewer", "human_reviewer")
        reason = data.get("reason", "Rejected by reviewer")

        # Check if review exists
        review = hitl_service.get_review_by_id(review_id)
        if not review:
            return jsonify({
                "status": "failed",
                "error": f"Review {review_id} not found"
            }), 404

        # Reject the review
        success = hitl_service.reject_review(
            review_id=review_id,
            reviewer=reviewer,
            reason=reason
        )

        if not success:
            return jsonify({
                "status": "failed",
                "error": "Failed to reject review"
            }), 500

        logger.info(f"Review {review_id} rejected by {reviewer}")

        return jsonify({
            "status": "success",
            "review_id": review_id,
            "message": "Review rejected successfully"
        }), 200

    except Exception as e:
        logger.error(f"Error in reject review endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@hitl_bp.get("/stats")
def get_stats():
    """
    Get HITL statistics.

    Returns:
        JSON object with statistics (pending count, etc.)
    """
    try:
        # Get counts of reviews by status
        # This is a simple implementation - could be expanded
        pending = hitl_service.get_pending_reviews(limit=1000)

        return jsonify({
            "status": "success",
            "stats": {
                "pending_count": len(pending),
                "connected": hitl_service.is_connected
            }
        }), 200

    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500
