from flask import Flask,render_template,request
import joblib
import numpy as np
regression_model_load=open('regression_model.pkl','rb')
regression_model=joblib.load(regression_model_load)
app=Flask(__name__)

@app.route('/')

def home():

    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])

def predict():
    if request.method == 'POST':
        try:
            Item_Identifier=float(request.form['Item_Identifier'])
            Item_Weight     =float(request.form['Item_Weight'])
            Item_Fat_Content=float(request.form['Item_Fat_Content'])
            Item_Visibility =float(request.form['Item_Visibility'])
            Item_Type       =float(request.form['Item_Type'])
            Item_MRP        =float(request.form['Item_MRP'])
            Outlet_Identifier=float(request.form['Outlet_Identifier'])
            Outlet_Establishment_Year=float(request.form['Outlet_Establishment_Year'])
            Outlet_Size=float(request.form['Outlet_Size'])
            Outlet_Location_Type=float(request.form['Outlet_Location_Type'])
            Outlet_Type=float(request.form['Outlet_Type'])
            preds_args=[Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,Outlet_Size,Outlet_Location_Type,Outlet_Type]
            preds_args_arr=np.array(preds_args)
            preds_args_arr=preds_args_arr.reshape(1,-1)
            
            preds=regression_model.predict(preds_args_arr)
            model_prediction=round(float(preds),2)
        except ValueError:
            return 'Check the values of the model'

    return render_template('predict.html',prediction=model_prediction)

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

if __name__ == "__main__":
    app.run()