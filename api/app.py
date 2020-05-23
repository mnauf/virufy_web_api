from flask import Flask, jsonify, request, make_response
import text_api
import pandas as pd
import numpy as np
from flask_cors import CORS



app = Flask(__name__)
CORS(app)
# /api/virufy -> all data will be posted here and we will return the response

@app.route('/api/predict', methods=['POST'])
def post():
    age = request.form.get('age')
    gender = request.form.get('gender')
    smoker = request.form.get('smoker')
    symptoms = request.form.getlist('reported_symptoms')
    medical_history = request.form.getlist('medical_history')
    response = {"age": int(age), "gender": gender,
     "smoker": smoker, "patient_reported_symptoms": symptoms,
     "medical_history": medical_history}
    print(response)
    symptoms = ",".join(symptoms)
    medical_history = ",".join(medical_history)
    response = {"age": int(age), "gender": gender,
     "smoker": smoker, "patient_reported_symptoms": symptoms,
     "medical_history": medical_history}
    print(response)
    df1 = pd.DataFrame(response,index=[0])
    df1 = df1.replace('NaN',np.NaN)
    prediction = text_api.predict(df1,"textual_model83.sav")
    print("prediction is: ",prediction)
    return make_response(jsonify({"data":round(prediction,3)}), 200)
    # message = {"data": "Hello World"}
    # return jsonify(message)

@app.route('/api', methods=['GET'])
def get():
    message = { 'message': 'This api just has endpoints for POST request' }
    return make_response(jsonify(message), 404)

app.run()
# if __name__ == "__main__":
#     app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
