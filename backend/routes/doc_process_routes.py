import sys

import os
from pathlib import Path
from flask import Blueprint, request, jsonify
from services import doc_processor
from utils.app_logger import LoggerSetup

logger = LoggerSetup("DocProcess_Rte").logger

process_bp = Blueprint("process", __name__)


@process_bp.post("/process_file")
def process_file():

    directory_path = Path("/Users/viviliu/Documents/10_Vivi/ChatMyCV/backend/data")
    files = list(directory_path.iterdir())

    if not files:
        return jsonify({
            "status": "failed", 
            "error": "No file provided"
        }), 400

    try:
        nodes = doc_processor.run(files)
    except Exception as e:
        logger.error(str(e), exc_info=True)

    return jsonify({
        "status": "success"
    }), 200


