from flask import Flask, request, jsonify, render_template
import joblib
import sys
sys.path.insert(0, './src')
from utils import clean_text

app        = Flask(__name__)
model      = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json()
    text    = data.get('text', '')
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]

    return jsonify({
        "label":      "REAL" if pred == 1 else "FAKE",
        "confidence": round(float(max(proba)) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)