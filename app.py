from flask import Flask,request,render_template
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Variance=float(request.form.get('Variance')),
            Standard_Deviation=float(request.form.get('Standard_Deviation')),
            Entropy=float(request.form.get('Entropy')),
            Skewness=float(request.form.get('Skewness')),
            Kurtosis=float(request.form.get('Kurtosis')),
            Contrast=float(request.form.get('Contrast')),
            Energy=float(request.form.get('Energy')),
            Homogeneity=float(request.form.get('Homogeneity')),
            Dissimilarity=float(request.form.get('Dissimilarity')),
            ASM=float(request.form.get('ASM')))
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=int(results[0]))
    
    

if __name__=="__main__":
    # app.run(host="0.0.0.0",port=8080)        
    app.run(debug=True,host='0.0.0.0', port=8080)        


