
from flask import Flask, jsonify, render_template, request

from askwise_api import database
from askwise_api.ollama_client import (
    best_model_for_ram,
    installed_models,
    ollama_reachable,
)
from askwise_api.pdf_reader import extract_text
from askwise_api.rag import run_and_store
from askwise_api.search_engine import SearchError

app = Flask(__name__)

# Initialize the database.
database.init_db()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    try:
        reachable = ollama_reachable()
        models = installed_models() if reachable else []

        return jsonify({
            "status": "ok",
            "ollama_connected": reachable,
            "models": models,
            "recommended_model": best_model_for_ram(),
        })

    except Exception:
        app.logger.exception("Health check failed")
        return jsonify({
            "status": "error",
            "message": "Could not check Ollama status.",
        }), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    question = data.get("message", data.get("question", ""))

    if not isinstance(question, str) or not question.strip():
        return jsonify({
            "error": "Please enter a question."
        }), 400

    question = question.strip()

    try:
        # Get the current conversation ID.
        conversation_id = data.get("conversationId")

        # Create a new conversation if no ID was provided.
        if conversation_id is None or conversation_id == "":
            conversation_id = database.create_conversation(
                question[:60]
            )
        else:
            try:
                conversation_id = int(conversation_id)
            except (TypeError, ValueError):
                return jsonify({
                    "error": "Invalid conversation ID."
                }), 400

            # Verify that this conversation exists.
            conversations = database.list_conversations()
            valid_ids = {
                int(item["id"]) for item in conversations
            }

            if conversation_id not in valid_ids:
                return jsonify({
                    "error": "Conversation not found. Please start a new chat."
                }), 404

        # Read options sent by the frontend.
        use_web = data.get("useWeb", True)
        use_model = data.get("useModel", True)
        document_ids = data.get("documentIds", [])

        if not isinstance(use_web, bool):
            use_web = True

        if not isinstance(use_model, bool):
            use_model = True

        if not isinstance(document_ids, list):
            document_ids = []

        try:
            document_ids = [
                int(doc_id)
                for doc_id in document_ids
                if not isinstance(doc_id, bool)
                and str(doc_id).isdigit()
            ]
        except (TypeError, ValueError):
            document_ids = []

        # Generate and save the answer.
        result = run_and_store(
            question,
            conversation_id=conversation_id,
            use_web=use_web,
            use_model=use_model,
            document_ids=document_ids,
        )

        # Return both naming formats for compatibility.
        result["conversationId"] = conversation_id
        result["conversation_id"] = conversation_id

        return jsonify(result)

    except SearchError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except Exception:
        app.logger.exception("Chat request failed")
        return jsonify({
            "error": "Something went wrong while generating the answer."
        }), 500


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({
            "error": "No file was uploaded."
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "error": "Please select a file."
        }), 400

    try:
        text = extract_text(file)

        if not text or not text.strip():
            return jsonify({
                "error": "No readable text found in the file."
            }), 400

        # Associate the document with the active conversation if possible.
        raw_conversation_id = request.form.get("conversationId")
        conversation_id = None

        if raw_conversation_id:
            try:
                candidate_id = int(raw_conversation_id)
                valid_ids = {
                    int(item["id"])
                    for item in database.list_conversations()
                }

                if candidate_id in valid_ids:
                    conversation_id = candidate_id
            except (TypeError, ValueError):
                conversation_id = None

        document_id = database.save_document(
            conversation_id,
            file.filename,
            text,
        )

        return jsonify({
            "filename": file.filename,
            "text": text,
            "documentId": document_id,
            "conversationId": conversation_id,
        })

    except Exception:
        app.logger.exception("File upload failed")
        return jsonify({
            "error": "Could not read the uploaded file."
        }), 500


@app.route("/api/conversations", methods=["GET"])
def conversations():
    try:
        return jsonify({
            "conversations": database.list_conversations()
        })
    except Exception:
        app.logger.exception("Could not list conversations")
        return jsonify({
            "error": "Could not load conversations."
        }), 500


@app.route("/api/conversations/<int:conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    try:
        conversations_list = database.list_conversations()
        valid_ids = {
            int(item["id"]) for item in conversations_list
        }

        if conversation_id not in valid_ids:
            return jsonify({
                "error": "Conversation not found."
            }), 404

        return jsonify({
            "conversationId": conversation_id,
            "messages": database.conversation_messages(
                conversation_id
            ),
        })

    except Exception:
        app.logger.exception("Could not load conversation")
        return jsonify({
            "error": "Could not load this conversation."
        }), 500


@app.route("/api/conversations/<int:conversation_id>", methods=["DELETE"])
def remove_conversation(conversation_id):
    try:
        deleted = database.delete_conversation(conversation_id)

        if not deleted:
            return jsonify({
                "error": "Conversation not found."
            }), 404

        return jsonify({
            "success": True,
            "conversationId": conversation_id,
        })

    except Exception:
        app.logger.exception("Could not delete conversation")
        return jsonify({
            "error": "Could not delete this conversation."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
