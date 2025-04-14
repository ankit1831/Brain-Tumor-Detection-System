import os
import tensorflow as tf
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix, accuracy_score
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import datetime
from src.components import data_ingestion, data_transformation

def train_model():
    

    obj = data_ingestion.DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transform = data_transformation.DataTransformation()
    train_arr, test_arr, _ = data_transform.initiate_data_transformation(train_data, test_data)

    X_train, y_train, X_test, y_test = (
        train_arr[:, :-1],
        train_arr[:, -1],
        test_arr[:, :-1],
        test_arr[:, -1]
    )

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # HL1
        Dense(32, activation='relu'),  # HL2
        Dense(1, activation='sigmoid')  # Output layer
    ])

    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    dl = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=100,
        callbacks=[tensorboard_callback, early_stopping_callback]
    )
    
    final_accuracy = dl.history['accuracy'][-1] * 100
    print(f"Final Training Accuracy: {final_accuracy:.4f}")

    

    return model 

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predicttdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Variance=float(request.form.get('Variance')),
            Standard_Deviation=float(request.form.get('Standard_Deviation')),
            Entropy=float(request.form.get('Entropy')),
            Skewness=float(request.form.get('Skewness')),
            Kurtosis=float(request.form.get('Kurtosis')),
            Contrast=float(request.form.get('Contrast')),
            Energy=float(request.form.get('Energy')),
            Homogeneity=float(request.form.get('Homogeneity')),
            Dissimilarity=float(request.form.get('Dissimilarity')),
            ASM=float(request.form.get('ASM'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=int(results[0]))

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        
        model = train_model()

    

    app.run(debug=True, host='0.0.0.0', port=8080)
    import os
    os.system('tensorboard --logdir=logs/fit')
