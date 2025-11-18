#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

from pathlib import Path
from flask import Blueprint, request, jsonify
from services import DocProcessor
from vectorstores.chroma_vectordb import ChromaUsage
from utils.app_logger import LoggerSetup

logger = LoggerSetup("DocProcess_Rte").logger

process_bp = Blueprint("process", __name__)


@process_bp.post("/process_file")
def process_file():

    data = request.get_json(silent=True) or {}
    lang = data.get("lang")

    allowed_langs = ["en", "zhtw"]

    if lang is not None:
        if lang not in allowed_langs:
            return jsonify({
                "status": "failed",
                "error": "lang must be 'en' or 'zhtw'"
            }), 400
        langs_to_process = [lang]
    else:
        langs_to_process = allowed_langs

    directory_path = Path("/Users/viviliu/Documents/10_Vivi/ChatMyCV/backend/data")

    lang_files: dict[str, list[Path]] = {}
    for lang_code in langs_to_process:
        folder = directory_path / lang_code
        if not folder.exists() or not folder.is_dir():
            logger.warning(f'Skipping missing directory for lang "{lang_code}"')
            continue
        files = [f for f in folder.iterdir() if f.is_file()]
        if files:
            files = [f for f in files if f.name != ".DS_Store"]
            lang_files[lang_code] = files

    if not lang_files:
        return jsonify({
            "status": "failed", 
            "error": "No file provided for requested languages"
        }), 400

    try:
        for lang_code, files in lang_files.items():
            logger.info(f'Processing files in {lang_code}')
            doc_processor = DocProcessor(lang=lang_code)
            nodes = doc_processor.run(files)
    except Exception as e:
        logger.error(str(e), exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500

    return jsonify({
        "status": "success",
        "processed_langs": list(lang_files.keys())
    }), 200


@process_bp.delete("/collection")
def delete_collection():
    data = request.get_json()

    if not data:
        return jsonify({
            "status": "failed",
            "error": "No JSON data provided"
        }), 400

    lang = data.get("lang")
    collection_name = data.get("collection_name")

    if lang not in ["en", "zhtw"]:
        return jsonify({
            "status": "failed",
            "error": "lang must be 'en' or 'zhtw'"
        }), 400

    target_collection = collection_name or f"chat_cv_{lang}"

    try:
        chroma_usage = ChromaUsage(collection_name=target_collection, auto_create=False)

        if not chroma_usage.collection:
            return jsonify({
                "status": "failed",
                "error": f'Collection "{target_collection}" does not exist'
            }), 404

        deleted = chroma_usage.delete_collection(collection_name=target_collection)

        if not deleted:
            return jsonify({
                "status": "failed",
                "error": f'Unable to delete collection "{target_collection}"'
            }), 500

        return jsonify({
            "status": "success",
            "collection": target_collection
        }), 200
    except Exception as e:
        logger.error(f"Error deleting collection {target_collection}: {e}", exc_info=True)
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    directory_path = Path("/Users/viviliu/Documents/10_Vivi/ChatMyCV/backend/data")
    files = list(directory_path.iterdir())
    # Get the last folder name from the directory path
    last_folder_name = directory_path.parts[-1] if directory_path.parts else None
    logger.info(f"Last folder name: {last_folder_name}")