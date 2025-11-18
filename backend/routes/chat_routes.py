#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

from flask import Blueprint, request, jsonify, Response, stream_with_context
from services import chat_service
from services.chat_serv import chroma_usage_en, chroma_usage_zhtw
from utils.app_logger import LoggerSetup

logger = LoggerSetup("Chat_Rte").logger

chat_bp = Blueprint("chat", __name__)


@chat_bp.post("/")
def chat():
    """
    Handle chat query with RAG.
    
    Expected JSON body:
    {
        "lang": "en" | "zhtw",
        "query": "user question",
        "character": "hr" | "engineer" (optional, interviewer character),
        "session_id": "optional session id",
        "conversation_history": [optional list of messages],
        "k": 5 (optional, number of docs to retrieve),
        "temperature": 0.7 (optional),
        "max_tokens": null (optional),
        "system_prompt": null (optional, overrides character),
        "model": null (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "failed",
                "error": "No JSON data provided"
            }), 400
        
        lang = data.get("lang", "en")
        query = data.get("query")
        
        if not query:
            return jsonify({
                "status": "failed",
                "error": "Query is required"
            }), 400
        
        if lang not in ["en", "zhtw"]:
            return jsonify({
                "status": "failed",
                "error": "lang must be 'en' or 'zhtw'"
            }), 400
        
        # Extract optional parameters
        character = data.get("character")
        session_id = data.get("session_id")
        conversation_history = data.get("conversation_history")
        k = data.get("k", 5)
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens")
        system_prompt = data.get("system_prompt")
        model = data.get("model")
        
        # Validate character if provided
        if character is not None and character.lower() not in ["hr", "engineer", "engineering", "eng"]:
            return jsonify({
                "status": "failed",
                "error": "character must be 'hr' or 'engineer'"
            }), 400
        
        # Get or create session_id (use last session if recent, otherwise create new)
        session_id = chat_service.get_or_create_session_id(session_id=session_id, timeout_seconds=180)
        
        logger.info(f"Chat request - lang: {lang}, character: {character}, query: {query[:50]}..., session_id: {session_id}")
        
        # Call chat service
        response = chat_service.chat(
            lang=lang,
            query=query,
            conversation_history=conversation_history,
            session_id=session_id,
            k=k,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            character=character,
            model=model
        )
        
        return jsonify({
            "status": "success",
            "response": response["content"],
            "session_id": session_id,
            "character": character,
            "usage": response.get("usage", {}),
            "retrieved_docs_count": response.get("retrieved_docs_count", 0),
            "context_used": response.get("context_used", False)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@chat_bp.post("/stream")
def stream_chat():
    """
    Handle streaming chat query with RAG.
    
    Expected JSON body:
    {
        "lang": "en" | "zhtw",
        "query": "user question",
        "character": "hr" | "engineer" (optional, interviewer character),
        "session_id": "optional session id",
        "conversation_history": [optional list of messages],
        "k": 5 (optional),
        "temperature": 0.7 (optional),
        "max_tokens": null (optional),
        "system_prompt": null (optional, overrides character),
        "model": null (optional)
    }
    
    Returns: Server-Sent Events (SSE) stream
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "failed",
                "error": "No JSON data provided"
            }), 400
        
        lang = data.get("lang", "en")
        query = data.get("query")
        
        if not query:
            return jsonify({
                "status": "failed",
                "error": "Query is required"
            }), 400
        
        if lang not in ["en", "zhtw"]:
            return jsonify({
                "status": "failed",
                "error": "lang must be 'en' or 'zhtw'"
            }), 400
        
        # Extract optional parameters
        character = data.get("character")
        session_id = data.get("session_id")
        conversation_history = data.get("conversation_history")
        k = data.get("k", 5)
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens")
        system_prompt = data.get("system_prompt")
        model = data.get("model")
        
        # Validate character if provided
        if character is not None and character.lower() not in ["hr", "engineer", "engineering", "eng"]:
            return jsonify({
                "status": "failed",
                "error": "character must be 'hr' or 'engineer'"
            }), 400
        
        # Get or create session_id (use last session if recent, otherwise create new)
        session_id = chat_service.get_or_create_session_id(session_id=session_id, timeout_seconds=180)
        
        logger.info(f"Stream chat request - lang: {lang}, character: {character}, query: {query[:50]}..., session_id: {session_id}")
        
        def generate():
            try:
                # Send session_id as first message
                yield f"data: [SESSION_ID] {session_id}\n\n"
                
                # Set the vectorstore based on lang parameter
                # (stream_chat doesn't currently support lang parameter)
                if lang == "zhtw":
                    chat_service.vectorstore = chroma_usage_zhtw
                elif lang == "en":
                    chat_service.vectorstore = chroma_usage_en
                
                for chunk in chat_service.stream_chat(
                    query=query,
                    conversation_history=conversation_history,
                    session_id=session_id,
                    k=k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    character=character,
                    model=model
                ):
                    yield f"data: {chunk}\n\n"
                
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in stream generation: {e}", exc_info=True)
                yield f"data: [ERROR] {str(e)}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in stream chat endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@chat_bp.post("/clear")
def clear_history():
    """
    Clear conversation history for a specific session.
    
    Expected JSON body:
    {
        "session_id": "session id to clear"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "failed",
                "error": "No JSON data provided"
            }), 400
        
        session_id = data.get("session_id")
        
        if not session_id:
            return jsonify({
                "status": "failed",
                "error": "session_id is required"
            }), 400
        
        logger.info(f"Clearing history for session: {session_id}")
        
        cleared = chat_service.clear_history(session_id)
        
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "cleared": cleared
        }), 200
        
    except Exception as e:
        logger.error(f"Error in clear history endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


@chat_bp.post("/clear_all")
def clear_all_histories():
    """
    Clear all conversation histories.
    
    No body required.
    """
    try:
        logger.info("Clearing all conversation histories")
        
        count = chat_service.clear_all_histories()
        
        return jsonify({
            "status": "success",
            "sessions_cleared": count
        }), 200
        
    except Exception as e:
        logger.error(f"Error in clear all histories endpoint: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500

