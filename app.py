from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from werkzeug.utils import secure_filename

app=Flask(__name__)

df1=pd.read_csv('heart.csv')
df1=df1.drop(index=14)
x1=df1.drop(columns='target',axis=1)
y1=df1['target']
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,stratify=y1,random_state=43)
model1=LogisticRegression()
model1.fit(x1_train,y1_train)


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
df.drop('id', axis=1, inplace=True)
gender_encoder = LabelEncoder()
ever_married_encoder = LabelEncoder()
work_type_encoder = LabelEncoder()
residence_type_encoder = LabelEncoder()
smoking_status_encoder = LabelEncoder()
df['gender'] = gender_encoder.fit_transform(df['gender'])
df['ever_married'] = ever_married_encoder.fit_transform(df['ever_married'])
df['work_type'] = work_type_encoder.fit_transform(df['work_type'])
df['Residence_type'] = residence_type_encoder.fit_transform(df['Residence_type'])
df['smoking_status'] = smoking_status_encoder.fit_transform(df['smoking_status'])
X = df.drop(columns='stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

model3 = RandomForestClassifier(n_estimators=100, random_state=43)
model3.fit(X_train, y_train)

def preprocess_input(data):
    data['gender'] = gender_encoder.transform([data['gender']])[0]
    data['ever_married'] = ever_married_encoder.transform([data['ever_married']])[0]
    data['work_type'] = work_type_encoder.transform([data['work_type']])[0]
    data['Residence_type'] = residence_type_encoder.transform([data['Residence_type']])[0]
    data['smoking_status'] = smoking_status_encoder.transform([data['smoking_status']])[0]
    return pd.DataFrame([data])



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/stroke')
def stroke():
    return render_template('stroke.html')

@app.route('/predict_stroke', methods=['POST'])
def predict_stroke():

    data = {
        'gender': request.form['gender'],
        'age': float(request.form['age']),
        'hypertension': int(request.form['hypertension']),
        'heart_disease': int(request.form['heartDisease']),
        'ever_married': request.form['everMarried'],
        'work_type': request.form['workType'],
        'Residence_type': request.form['residenceType'],
        'avg_glucose_level': float(request.form['avgGlucoseLevel']),
        'bmi': float(request.form['bmi']),
        'smoking_status': request.form['smokingStatus']
    }

    input_data = preprocess_input(data)
    prediction = model3.predict(input_data)
    result = "Your risk of stroke is high. This could be due to factors such as elevated blood pressure, heart disease, or unhealthy lifestyle choices." if prediction[0] == 1 else "Your risk of stroke is currently low. This suggests good management of risk factors such as blood pressure, glucose levels, and lifestyle habits."
    return render_template('stroke_result.html', result=result)

from tensorflow import keras
from keras.preprocessing import image
import os

model = keras.models.load_model('lung_cancer_cnn_model.h5')
class_labels = ['Lung Adenocarcinomas', 'Lung Normal ', 'Lung Squamous Cell Carcinomas']

if not os.path.exists('uploads'):
    os.makedirs('uploads')

def preprocess_and_predict(img_path):
    img_height, img_width = 128, 128
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])  # Get the index of the highest score
    return class_labels[predicted_class]

@app.route('/lung')
def lung():
    return render_template('lung.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)


        result = preprocess_and_predict(file_path)

        os.remove(file_path)

        return jsonify({'prediction': result})

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    input_method = request.form.get('input_method')

    if input_method == 'file' and 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            return "Unsupported file format. Please upload a CSV or Excel file."
        
        input_data1 = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                            'ca', 'thal']].values
        
        predictions = model1.predict(input_data1)
        results = ["You don't have any Symptoms of Heart Disease" if pred == 0 else "You have Symptoms of Heart Disease. Please consult a doctor." for pred in predictions]
        return render_template('heart_result1.html', results=results)
    

    elif input_method == 'form':
        
        age = int(request.form['age'])
        sex_input = request.form['sex'].strip().lower()
        sex = 1 if sex_input == 'male' else 0

        cp_input = request.form['cp'].strip().lower()
        cp = {'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3}.get(cp_input, -1)

        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])

        fbs_input = int(request.form['fbs'])
        fbs = 1 if fbs_input > 120 else 0

        restecg_input = request.form['restecg'].strip().lower()
        restecg = {'normal': 0, 'st-t wave abnormality': 1, 'left ventricular hypertrophy': 2}.get(restecg_input, -1)

        thalach = int(request.form['thalach'])

        exang_input = request.form['exang'].strip().lower()
        exang = 1 if exang_input == 'yes' else 0

        oldpeak = float(request.form['oldpeak'])

        slope_input = request.form['slope'].strip().lower()
        slope = {'upsloping': 0, 'flat': 1, 'downsloping': 2}.get(slope_input, -1)

        ca = int(request.form['ca'])

        thal_input = request.form['thal'].strip().lower()
        thal = {'normal': 1, 'fixed defect': 2, 'reversible defect': 3}.get(thal_input, -1)

        input_data1 = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        prediction = model1.predict(input_data1)
        
        result = "You don't have any Symptoms of Heart Disease" if prediction[0] == 0 else "You have Symptoms of getting Heart Disease. Please consult a doctor."

        return render_template('heart_result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
