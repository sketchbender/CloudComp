from flask import Flask, render_template, request
import numpy as np
from sklearn import linear_model

app = Flask(__name__)

# ML Model Setup
height = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0]]
weight = [8, 10, 12, 14, 16, 18, 20, 22, 24]
reg = linear_model.LinearRegression()
reg.fit(height, weight)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        height_input = float(request.form['height'])
        prediction = reg.predict([[height_input]])[0]
        return render_template('index.html', prediction_text=f"Predicted Weight for height {height_input}: {prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
