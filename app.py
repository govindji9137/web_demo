from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
poly = pickle.load(open('poly.pkl','rb'))
# Load the trained model
model_path = 'polylogi.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


from sklearn.preprocessing import PolynomialFeatures



@app.route('/predict', methods=['POST'])
def predict():
    # features = [float(x) for x in request.form.values()]
    # features_array = np.array(features).reshape(1, -1)
    x1 = float(request.form['x1'])
    x2 = float(request.form['x2'])

    features_array = np.array([[x1, x2]])

    # Apply same transformation
    final_features = poly.transform(features_array)

    prediction = model.predict(final_features)

    output = 'True' if prediction[0] == 1 else 'False'

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True) 