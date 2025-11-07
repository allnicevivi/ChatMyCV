# from typing import Optional

# from flask import Blueprint, request, jsonify

# from backend.llm_providers.azure_client import AzureClient
# from backend.llm_providers.gemini_client import GeminiClient


# chat_bp = Blueprint("chat", __name__)


# def _select_provider(requested: Optional[str]) -> str:
#     if requested in {"azure", "gemini"}:
#         return requested
#     import os
#     if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_API_BASE"):
#         return "azure"
#     if os.getenv("GOOGLE_API_KEY"):
#         return "gemini"
#     return "echo"


# @chat_bp.post("/completion")
# def chat_completion():
#     body = request.get_json(silent=True) or {}
#     messages = body.get("messages") or []
#     if not isinstance(messages, list) or not messages:
#         return jsonify({"error": "messages cannot be empty"}), 400

#     provider_name = _select_provider(body.get("provider"))
#     model = body.get("model")
#     temperature = body.get("temperature", 0.7)
#     max_tokens = body.get("max_tokens")
#     extra = body.get("extra") or {}

#     # Normalize messages to expected format {role, content}
#     norm_messages = []
#     for m in messages:
#         role = (m.get("role") or "user").strip()
#         content = m.get("content") or ""
#         norm_messages.append({"role": role, "content": content})

#     if provider_name == "azure":
#         client = AzureClient()
#         content = client.get_chat_completion(
#             messages=norm_messages,
#             model=model or "gpt-35-turbo",
#             temperature=temperature,
#             max_tokens=max_tokens,
#             **extra,
#         )
#         return jsonify({
#             "provider": provider_name,
#             "model": model or "gpt-35-turbo",
#             "content": content,
#         })

#     if provider_name == "gemini":
#         client = GeminiClient()
#         content = client.get_chat_completion(
#             messages=norm_messages,
#             model=model or "gemini-1.5-flash",
#             temperature=temperature,
#             max_tokens=max_tokens,
#             **extra,
#         )
#         return jsonify({
#             "provider": provider_name,
#             "model": model or "gemini-1.5-flash",
#             "content": content,
#         })

#     last_user = next((m["content"] for m in reversed(norm_messages) if m.get("role") == "user"), "")
#     return jsonify({"provider": "echo", "model": None, "content": f"[echo] {last_user}"})


