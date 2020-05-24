from flask import Flask,render_template,request
import joblib
import numpy as np

preds_args= [1, 10, 4, 2, 3,1000,2, 1998, 2, 3, 4]
preds_args_arr=np.array(preds_args)
preds_args_arr=preds_args_arr.reshape(1,-1)

regression_model_load=open('regression_model.pkl','rb')
regression_model=joblib.load(regression_model_load)
preds=regression_model.predict(preds_args_arr)
model_prediction=round(float(preds),2)
print(model_prediction)