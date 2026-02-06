from flask import Flask, render_template, request, jsonify
from groq import Groq
from deep_translator import GoogleTranslator

app = Flask(__name__)

API_KEY = "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"
client = Groq(api_key=API_KEY)


def translate_text(text, target_language):
    try:
        return GoogleTranslator(source="auto", target=target_language).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


@app.route("/")
def home():
    return render_template("s2.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    target_language = data.get("language", "en")

    try:
        user_message_en = translate_text(user_message, "en")

        

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": user_message_en}],
            model="llama-3.3-70b-versatile"
        )

        bot_response_en = response.choices[0].message.content
        bot_response_translated = translate_text(bot_response_en, target_language)

        return jsonify({"response": bot_response_translated})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "Sorry, there was an error. Try again later!"})

if __name__ == "__main__":
    app.run(debug=True)
