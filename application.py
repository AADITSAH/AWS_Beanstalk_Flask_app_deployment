from flask import Flask,render_template,request, jsonify
import pickle
import sklearn
from sklearn import svm
#init
application=Flask(__name__)

# def fun(a):
# 	return a+5

# print("printing function :", fun(6))

#Route
@application.route('/')
def index():
	return render_template('index.html')

def predict_proba(sl,sw,pl,pw):
	data={'flower_type':'','probability': ''}
	sl=float(sl)
	sw=float(sw)
	pl=float(pl)
	pw=float(pw)

	ls=[[sl,sw,pl,pw]]
	with open('model_pkl' , 'rb') as f:
		loaded_model = pickle.load(f)

	res=loaded_model.predict(ls)[0]
	prob1=loaded_model.predict_proba(ls)
	prob2=prob1[0][res]*100

	if (res==0):
		data['flower_type']='Iris-setosa'
	elif (res==1):
		data['flower_type']='Iris-versicolor'
	else:
		data['flower_type']='Iris-virginica'
	data['probability']=prob2

	return data

@application.route('/predict',methods=['GET','POST'])
def predict():
	if request.method=='POST':
		sl=request.form['sl']
		sw=request.form['sw']
		pl=request.form['pl']
		pw=request.form['pw']

		result=predict_proba(sl,sw,pl,pw)
		return render_template('index.html',flower_type=result['flower_type'], probability=result['probability'])


if __name__ =='__main__':
	application.run(debug=False)
