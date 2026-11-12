
import os
import time
import requests
import mlflow

from flask import Flask, request, render_template_string

app = Flask(__name__)

# Configuration
MODEL_NAME = "qwen2.5:0.5b"

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://host.minikube.internal:11434/api/generate"
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.minikube.internal:5001"
)

MLFLOW_EXPERIMENT = "LLM-Chatbot-Tracking"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Chatbot GUI
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>My LLM Chatbot</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #101827;
            color: white;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: 90%;
            max-width: 750px;
            background: #1e293b;
            padding: 30px;
            border-radius: 18px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.4);
        }

        h1 {
            text-align: center;
            color: #38bdf8;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 30px;
        }

        textarea {
            width: 100%;
            height: 110px;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #475569;
            background: #0f172a;
            color: white;
            font-size: 16px;
            resize: vertical;
        }

        button {
            width: 100%;
            padding: 14px;
            margin-top: 15px;
            border: none;
            border-radius: 10px;
            background: #0284c7;
            color: white;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #0369a1;
        }

        .response {
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            background: #0f172a;
            line-height: 1.7;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }

        .response h3 {
            color: #38bdf8;
        }

        .error {
            color: #f87171;
        }

        .status {
            text-align: center;
            margin-top: 20px;
            color: #94a3b8;
            font-size: 13px;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>My LLM Chatbot</h1>

    <p class="subtitle">
        Powered by Ollama, Qwen and Kubernetes
    </p>

    <form method="POST">

        <textarea
            name="question"
            placeholder="Ask me anything..."
            required
        >{{ question }}</textarea>

        <button type="submit">
            Ask AI
        </button>

    </form>

    {% if answer %}
    <div class="response">
        <h3>AI Response</h3>
        {{ answer }}
    </div>
    {% endif %}

    {% if error %}
    <div class="response error">
        <h3>Error</h3>
        {{ error }}
    </div>
    {% endif %}

    <div class="status">
        Model: Qwen 2.5 | Tracking: MLflow
    </div>

</div>

</body>
</html>
"""


# Chatbot route
@app.route("/", methods=["GET", "POST"])
def home():

    answer = ""
    error = ""
    question = ""

    if request.method == "POST":

        question = request.form.get("question", "").strip()

        if question:

            start_time = time.perf_counter()
            success = False
            failure_type = "None"

            try:

                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL_NAME,
                        "prompt": question,
                        "stream": False
                    },
                    timeout=180
                )

                response.raise_for_status()

                answer = response.json()["response"]
                success = True

            except (requests.RequestException, ValueError, KeyError) as exc:

                error = str(exc)
                failure_type = type(exc).__name__

            finally:

                response_time = time.perf_counter() - start_time

                # Log chatbot interaction in MLflow
                try:

                    mlflow.set_experiment(MLFLOW_EXPERIMENT)

                    with mlflow.start_run():

                        mlflow.log_param("model", MODEL_NAME)
                        mlflow.log_param("framework", "Ollama")
                        mlflow.log_param("question_length", len(question))

                        mlflow.log_metric(
                            "response_time_seconds",
                            response_time
                        )

                        mlflow.log_metric(
                            "success",
                            int(success)
                        )

                        mlflow.set_tag(
                            "application",
                            "LLM Chatbot"
                        )

                        mlflow.set_tag(
                            "status",
                            "success" if success else "failure"
                        )

                        mlflow.set_tag(
                            "failure_type",
                            failure_type
                        )

                        # Conversation content is stored as MLflow artifacts
                        mlflow.log_text(
                            question,
                            "conversation/user_question.txt"
                        )

                        mlflow.log_text(
                            answer if success else error,
                            "conversation/ai_response.txt"
                        )

                except Exception:

                    app.logger.exception(
                        "MLflow logging failed"
                    )

    return render_template_string(
        HTML,
        question=question,
        answer=answer,
        error=error
    )


# Health check
@app.route("/health")
def health():

    return {"status": "healthy"}


# Run Flask
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )