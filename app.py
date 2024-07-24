from flask import Flask, request, jsonify, render_template, send_file
import requests
import os
from gtts import gTTS
import io
import base64

app = Flask(__name__)


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "Your_API_Key")
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY", "Your_API_Key")


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_ai():
    data = request.json
    user_query = data.get('query')
    mode = data.get('mode', 'text')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    if mode == 'text' or mode == 'speech':
        text_response = query_groq(user_query)
        if mode == 'speech':
            return text_to_speech(text_response['response'])
        return jsonify(text_response)
    elif mode == 'image':
        return generate_image(user_query)
    else:
        return jsonify({"error": "Invalid mode"}), 400

def query_groq(user_query):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": user_query}],
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        groq_response = response.json()
        
        return {
            "query": user_query,
            "response": groq_response['choices'][0]['message']['content']
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Error communicating with Groq API: {str(e)}"}

def generate_image(prompt):
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30,
    }

    try:
        response = requests.post(STABILITY_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        image_base64 = data['artifacts'][0]['base64']
        return jsonify({
            "query": prompt,
            "image": image_base64
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error communicating with Stability AI API: {str(e)}"}), 500

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        return send_file(mp3_fp, mimetype="audio/mpeg", as_attachment=True, download_name="response.mp3")

    except Exception as e:
        return jsonify({"error": f"Error generating speech: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

