from flask import Flask, render_template, request
from ml_model import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Make prediction using the ML model
        crop = predict(N, P, K, temperature, humidity, ph, rainfall)

        return render_template('result.html', crop=crop)

if __name__ == '__main__':
    app.run(debug=True)
