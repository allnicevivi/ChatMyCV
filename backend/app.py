from flask import Flask, jsonify
from flask_cors import CORS

from routes.chat_routes import chat_bp
# from routes.uploaded_routes import upload_bp
from routes.doc_process_routes import process_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["APP_NAME"] = "ChatMyCV API"
    app.config["APP_VERSION"] = "0.1.0"

    # CORS (allow all origins by default; tighten for production)
    CORS(app, resources={r"*": {"origins": "*"}})

    @app.get("/")
    def root():
        return jsonify({"message": "ChatMyCV backend is running"})

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    # Blueprints
    app.register_blueprint(chat_bp, url_prefix="/chat")
    # app.register_blueprint(upload_bp, url_prefix="/upload")
    app.register_blueprint(process_bp, url_prefix="/process")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


