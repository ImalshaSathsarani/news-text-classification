from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

#Load the trained model
model = joblib.load('models/best_text_classification_model.pkl')

#Define target names
classes = ['comp.graphics', 'rec.sport.hockey','sci.space','talk.politics.mideast']

@app.route('/')
def home():
    return "Text Classification API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text','')

        if not text:
            return jsonify({'error':'No text provided'}), 400
        
        #predict
        prediction_idx = model.predict([text])[0]
        prediction_label = classes[prediction_idx]

        return jsonify({
            'text': text,
            'prediction':prediction_label
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)