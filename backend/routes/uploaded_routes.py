from flask import Blueprint, request, jsonify


upload_bp = Blueprint("upload", __name__)


@upload_bp.post("/file")
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    data = file.read()

    try:
        preview = data.decode("utf-8", errors="ignore")[:200]
    except Exception:
        preview = ""

    return jsonify({
        "filename": getattr(file, "filename", None),
        "content_type": getattr(file, "content_type", None) or file.mimetype,
        "size_bytes": len(data),
        "preview": preview,
    })


