
from __future__ import division, print_function
import os
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_covid19GPU100e.h5'

# Load your trained model
model = load_model(MODEL_PATH)

doc_path='Doctors.csv'
hosp_path='Hospitals.csv'



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))   
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)   
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Person is Covid Negative"
    else:
        preds="The Person is Covid Positive"
        
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

@app.route('/submit1', methods=['GET', 'POST'])
def doc_rec():
    
    if request.method == 'POST':
        userinput=request.form["Source"]
        df_doc=pd.read_csv(doc_path)
        df_docfilter=df_doc[(df_doc['Status']=='Available') & (df_doc['Speciality']==userinput)]
        df_docrec=df_docfilter[['Doctor Name','Contact Number']]
        doc_rec=str((df_docrec.head(10).to_records(index=False)))
        return render_template('index.html', prediction_text1=f"10 recommended doctors in the format (Name, Contact Number) are: {doc_rec}")
        
    return None

@app.route('/submit2', methods=['GET', 'POST'])
def hosp_rec():
    
    if request.method == 'POST':
        userinput=request.form["Source"]
        if (userinput=='Yes'):
            df_hosp=pd.read_csv(hosp_path)
            df_hospfilter=df_hosp[(df_hosp['Availability Percentage'] >= 60) & (df_hosp['institution_type']=='Hospital')]
            df_hosprec=df_hospfilter[['ID','Available Beds']]
            hosp_rec=str(df_hosprec.head(10).to_records(index=False))
            return render_template('index.html', prediction_text2=f"10 recommended hospitals in the format (Hospital ID, No. of available beds) are: {hosp_rec}")
        else:
            return render_template('index.html', prediction_text2="Take the prescribed medicines, rest well at home and stay safe.")
        
    return None

if __name__ == '__main__':
    app.run(debug=True)




