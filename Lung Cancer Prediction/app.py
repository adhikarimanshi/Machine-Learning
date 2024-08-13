from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load the saved lung cancer prediction model
model_path = 'D:\\Lung cancer Prediction\\ML_MODEL\\random_forest_model.pkl'
model = joblib.load(model_path)

# Define the expected columns manually based on the training data
expected_columns = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
    'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING',
    'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN',
    'ANXYELFIN'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username == "admin" and password == "password":
        return redirect(url_for('form_page'))
    else:
        error = "Invalid username or password"
        return render_template('home.html', error=error)

@app.route('/form', methods=['GET', 'POST'])
def form_page():
    if request.method == 'POST':
        patient_name = request.form['patient_name']
        age = int(request.form['age'])
        gender = 1 if request.form['gender'] == 'Male' else 0
        smoking = 1 if request.form['smoking'] == 'Yes' else 0
        yellow_fingers = 1 if request.form['yellow_fingers'] == 'Yes' else 0
        anxiety = 1 if request.form['anxiety'] == 'Yes' else 0
        peer_pressure = 1 if request.form['peer_pressure'] == 'Yes' else 0
        chronic_disease = 1 if request.form['chronic_disease'] == 'Yes' else 0
        fatigue = 1 if request.form['fatigue'] == 'Yes' else 0
        allergy = 1 if request.form['allergy'] == 'Yes' else 0
        wheezing = 1 if request.form['wheezing'] == 'Yes' else 0
        alcohol_consuming = 1 if request.form['alcohol_consuming'] == 'Yes' else 0
        coughing = 1 if request.form['coughing'] == 'Yes' else 0
        shortness_of_breath = 1 if request.form['shortness_of_breath'] == 'Yes' else 0
        swallowing_difficulty = 1 if request.form['swallowing_difficulty'] == 'Yes' else 0
        chest_pain = 1 if request.form['chest_pain'] == 'Yes' else 0

        # Calculate ANXYELFIN feature
        anxyelfin = anxiety * yellow_fingers

        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({
            'GENDER': [gender],
            'AGE': [age],
            'SMOKING': [smoking],
            'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety],
            'PEER_PRESSURE': [peer_pressure],
            'CHRONIC DISEASE': [chronic_disease],
            'FATIGUE ': [fatigue],
            'ALLERGY ': [allergy],
            'WHEEZING': [wheezing],
            'ALCOHOL CONSUMING': [alcohol_consuming],
            'COUGHING': [coughing],
            'SHORTNESS OF BREATH': [shortness_of_breath],
            'SWALLOWING DIFFICULTY': [swallowing_difficulty],
            'CHEST PAIN': [chest_pain],
            'ANXYELFIN': [anxyelfin]
        })

        # Ensure columns are in the same order as during model training
        input_data = input_data[expected_columns]

        # Prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        # Plotting
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        sns.barplot(x=['No Lung Cancer', 'Lung Cancer'], y=[1 - probability, probability], ax=axes, palette=['green', 'red'])
        axes.set_title('Lung Cancer Probability')
        axes.set_ylabel('Probability')

        # Save plot to a BytesIO object and encode it to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Render result template
        result = 'Lung Cancer' if prediction[0] == 1 else 'No Lung Cancer'
        return render_template('result.html', name=patient_name, result=result, probability=probability, plot_url=plot_url)
   


    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
