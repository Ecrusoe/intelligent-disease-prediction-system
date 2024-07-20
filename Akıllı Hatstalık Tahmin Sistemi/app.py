import os
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

def predict(values, dic):
   #diyabet
    if len(values) == 8:
        dic2 = {'NewBMI_Obesity 1': 0, 'NewBMI_Obesity 2': 0, 'NewBMI_Obesity 3': 0, 'NewBMI_Overweight': 0,
                'NewBMI_Underweight': 0, 'NewInsulinScore_Normal': 0, 'NewGlucose_Low': 0,
                'NewGlucose_Normal': 0, 'NewGlucose_Overweight': 0, 'NewGlucose_Secret': 0}

        if dic['BMI'] <= 18.5:
            dic2['NewBMI_Underweight'] = 1
        elif 18.5 < dic['BMI'] <= 24.9:
            pass
        elif 24.9 < dic['BMI'] <= 29.9:
            dic2['NewBMI_Overweight'] = 1
        elif 29.9 < dic['BMI'] <= 34.9:
            dic2['NewBMI_Obesity 1'] = 1
        elif 34.9 < dic['BMI'] <= 39.9:
            dic2['NewBMI_Obesity 2'] = 1
        elif dic['BMI'] > 39.9:
            dic2['NewBMI_Obesity 3'] = 1

        if 16 <= dic['Insulin'] <= 166:
            dic2['NewInsulinScore_Normal'] = 1

        if dic['Glucose'] <= 70:
            dic2['NewGlucose_Low'] = 1
        elif 70 < dic['Glucose'] <= 99:
            dic2['NewGlucose_Normal'] = 1
        elif 99 < dic['Glucose'] <= 126:
            dic2['NewGlucose_Overweight'] = 1
        elif dic['Glucose'] > 126:
            dic2['NewGlucose_Secret'] = 1

        dic.update(dic2)
        values2 = list(map(float, list(dic.values())))

        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values2)
        return model.predict(values.reshape(1, -1))[0]

    # göğüs_kanseri
    elif len(values) == 22:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # kalp
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    #  böbrek
    elif len(values) == 24:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # karaciğer
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')


    

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)

# Sıtma tahmini için rotayı tanımlayın
@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            # Yüklenen resmi kaydedin
            img = Image.open(request.files['image'])
            img.save("uploads/image.jpg")
            
            # Görüntüyü tahmin için hazırlayın
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # Modeli yükleyin
            model = tf.keras.models.load_model("models/malaria_model.h5")
            
            # Tahmin yap
            prediction = model.predict(img_array)
            
           # Sonucu belirleyin
            if prediction[0][0] > 0.5:
                result = "Sıtma Tespit Edilmemiştir."
            else:
                result = "Sıtma Olabilirsiniz"
            
            return render_template('malaria_predict.html', pred=result)
        
        except Exception as e:
            # İstisnaları veya hataları ele alın
            message = f"Error: {str(e)}"
            return render_template('malaria.html', message=message)
    
    return render_template('pneumonia.html')
@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            # Yüklenen resmi kaydedin
            img = Image.open(request.files['image'])
            img.save("uploads/image.jpeg")
            
            # Görüntüyü tahmin için hazırlayın
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpeg')
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # Modeli yükleyin
            model = tf.keras.models.load_model("models/pneumonia_model.h5")
            
            # Tahmin yap
            prediction = model.predict(img_array)
            
           # Sonucu belirleyin
            if prediction[0][0] > 0.5:
                result = "Zatürre Tespit Edilmemiştir."
            else:
                result = "Zatürre Olabilirsiniz"
            
            return render_template('pneumonia_predict.html', pred=result)
        
        except Exception as e:
            # İstisnaları veya hataları ele alın
            message = f"Error: {str(e)}"
            return render_template('pneumonia.html', message=message)
    
    return render_template('pneumonia.html')



if __name__ == '__main__':
    app.run(debug = True)