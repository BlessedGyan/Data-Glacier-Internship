from flask import Flask, send_file, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the model
model_path = r"C:\Users\bless\titanic_model.pkl"  # Correct path to the model file
model = joblib.load(model_path)

# Serve the HTML page
@app.route('/')
def home():
    return send_file(r"C:\Users\bless\Downloads\titanic.html")  # Correct path to the HTML file

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = [[
        data['Pclass'],
        data['Sex'],  # 0 for female, 1 for male
        data['Age'],
        data['SibSp'],
        data['Parch'],
        data['Fare']
    ]]
    prediction = model.predict(input_data)
    return jsonify({'Survived': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
