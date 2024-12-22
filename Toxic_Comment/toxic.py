from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from lime.lime_text import LimeTextExplainer


app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'distilbert-base-uncased'
try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    model.load_state_dict(torch.load('C://Users//HP//Desktop//Gitdemo//Toxic_Comment//distilbert_toxicity_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise

explainer = LimeTextExplainer(class_names=["Non-Toxic", "Toxic"])

def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

def predict_toxicity(text):
    
    prob = predict_proba([text])[0][1]  
    response = {"label": "Non-toxic comment", "probability": f"{prob:.2f}"}

    if prob > 0.5:
        response["label"] = "Toxic comment detected!"
        exp = explainer.explain_instance(text, predict_proba, num_features=5)
        response["explanation"] = exp.as_list()

    return response

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input. Please provide 'text' in JSON format."}), 400

    text = data['text']
    try:
        prediction = predict_toxicity(text)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
