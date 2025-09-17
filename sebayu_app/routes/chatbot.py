import json
from flask import render_template, request, jsonify
# Impor relatif dari package routes
from . import chatbot_bp
# Impor relatif dari package sebayu_app
from ..database import get_db, now_str
from ..utils import handle_chat_message

# --- Web Chatbot UI + API ---
@chatbot_bp.route("/chat")
def chat_page():
    return render_template("chat.html")

@chatbot_bp.post("/api/chat")
def api_chat():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    user = (data.get("username") or "web-user").strip()
    reply = handle_chat_message(text)

    lowered = text.lower()
    if lowered.startswith("request ") or lowered.startswith("req "):
        payload = text.split(" ", 1)[1] if " " in text else ""
        if payload:
            with get_db() as db:
                db.execute(
                    "INSERT INTO requests(username, platform, message, status, created_at) VALUES(?,?,?,?,?)",
                    (user, "web", payload, "baru", now_str()),
                )
                db.commit() # Commit perubahan
            reply += "\n\n✅ Request kamu sudah tercatat. Terima kasih!"
    return jsonify({"reply": reply})