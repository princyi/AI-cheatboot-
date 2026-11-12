import os
import requests
from flask import Flask, request, render_template_string

app = Flask(__name__)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://host.minikube.internal:11434/api/generate"
)

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>My LLM Chatbot</title>

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Arial, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #101827, #1e293b);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            color: white;
        }

        .chat-container {
            width: 100%;
            max-width: 850px;
            height: 85vh;
            background: #172033;
            border: 1px solid #334155;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        }

        .header {
            padding: 22px;
            background: #1e293b;
            border-bottom: 1px solid #334155;
            text-align: center;
        }

        .header h1 {
            font-size: 25px;
            color: #38bdf8;
        }

        .header p {
            color: #94a3b8;
            font-size: 13px;
            margin-top: 7px;
        }

        .status {
            color: #4ade80;
            font-size: 12px;
            margin-top: 10px;
        }

        .chat-box {
            flex: 1;
            padding: 25px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .welcome {
            text-align: center;
            margin: auto;
            color: #cbd5e1;
        }

        .welcome h2 {
            font-size: 27px;
            margin-bottom: 12px;
            color: white;
        }

        .welcome p {
            color: #94a3b8;
        }

        .message {
            max-width: 80%;
            padding: 15px 18px;
            border-radius: 15px;
            line-height: 1.6;
            overflow-wrap: anywhere;
        }

        .user {
            align-self: flex-end;
            background: #2563eb;
            color: white;
            border-bottom-right-radius: 4px;
        }

        .ai {
            align-self: flex-start;
            background: #334155;
            color: #e2e8f0;
            border-bottom-left-radius: 4px;
        }

        .error {
            background: #7f1d1d;
            color: #fecaca;
            padding: 14px;
            border-radius: 10px;
        }

        .input-area {
            padding: 20px;
            background: #1e293b;
            border-top: 1px solid #334155;
        }

        .input-form {
            display: flex;
            gap: 12px;
        }

        input {
            flex: 1;
            padding: 16px;
            border: 1px solid #475569;
            outline: none;
            border-radius: 12px;
            background: #0f172a;
            color: white;
            font-size: 15px;
        }

        input:focus {
            border-color: #38bdf8;
        }

        button {
            padding: 14px 25px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(37,99,235,0.4);
        }

        .footer {
            text-align: center;
            color: #64748b;
            font-size: 11px;
            margin-top: 12px;
        }

        @media(max-width: 600px) {
            .chat-container {
                height: 90vh;
            }

            .message {
                max-width: 95%;
            }

            .header h1 {
                font-size: 20px;
            }

            button {
                padding: 12px;
            }
        }
    </style>
</head>

<body>

    <div class="chat-container">

        <div class="header">
            <h1>🤖 My LLM Chatbot</h1>
            <p>Powered by AI | Kubernetes Deployment</p>
            <div class="status">● AI Assistant</div>
        </div>

        <div class="chat-box" id="chatBox">

            {% if not answer and not error %}
            <div class="welcome">
                <h2>Welcome! 👋</h2>
                <p>I'm your AI assistant. Ask me anything!</p>
            </div>
            {% endif %}

            {% if question %}
            <div class="message user">
                <strong>You:</strong><br>
                {{ question }}
            </div>
            {% endif %}

            {% if answer %}
            <div class="message ai">
                <strong>🤖 AI Assistant:</strong><br>
                {{ answer }}
            </div>
            {% endif %}

            {% if error %}
            <div class="error">
                <strong>⚠ Error:</strong><br>
                {{ error }}
            </div>
            {% endif %}

        </div>

        <div class="input-area">

            <form method="POST" class="input-form">
                <input
                    type="text"
                    name="question"
                    placeholder="Type your message here..."
                    required
                    autocomplete="off"
                >

                <button type="submit">Send ➤</button>
            </form>

            <div class="footer">
                My LLM Chatbot | Built with Python, Flask, Docker & Kubernetes
            </div>

        </div>

    </div>

    <script>
        const chatBox = document.getElementById("chatBox");
        chatBox.scrollTop = chatBox.scrollHeight;
    </script>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    error = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": "qwen2.5:0.5b",
                    "prompt": question,
                    "stream": False
                },
                timeout=180
            )
            response.raise_for_status()
            answer = response.json()["response"]

        except requests.RequestException as e:
            error = str(e)

    return render_template_string(
        HTML, answer=answer, error=error
    )

@app.route("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
