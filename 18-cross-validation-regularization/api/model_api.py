import json
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd
import traceback

app = Flask(__name__)

COLUMNS = ['Income','Limit','Cards','Age','Education',
		   'Gender','Student','Married','Balance']

@app.route('/predict', methods=['POST'])
def predict():

	try:
		query = pd.DataFrame(request.json)
		query = query[COLUMNS]
		prediction = clf.predict(query)
		return jsonify({
			'data': query.to_dict(),
			'prediction': list(prediction)
			})

	except Exception as e:
		return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    clf = joblib.load('../models/cross-validation-challenge-model.pkl')
    app.run(port=8080)