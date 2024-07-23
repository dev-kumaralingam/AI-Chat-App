from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_groq():
    user_query = request.json.get('query')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

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
        
        return jsonify({
            "query": user_query,
            "response": groq_response['choices'][0]['message']['content']
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error communicating with Groq API: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)