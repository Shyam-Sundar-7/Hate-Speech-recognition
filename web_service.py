from flask import Flask
from flask import Flask
from flask import jsonify,request
from inference import HatePredictor
from waitress import serve
import torch



model = HatePredictor("models/hate_model.ckpt")

app = Flask('__name__')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    y_pred = model.predict(client["sentence"])
    toxic_score = y_pred[1]['score']


    
    return jsonify({'hate_score': float(toxic_score),
                    'speech': "Toxic Speech" if toxic_score >= 0.5 else "Normal Speech"})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)   
