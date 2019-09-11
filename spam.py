# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:41:48 2019

@author: Nayeemuddin
"""
from flask import Flask,request,render_template
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=pd.read_csv("spam.csv",encoding="latin-1")
    data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    ps=PorterStemmer()

    corpus=[]

    for i in range(0,len(data)):
        review=re.sub('[^a-zA-Z]',' ',data['message'][i])
        review=review.lower()
        review=review.split()
        review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
        review=' '.join(review)
        corpus.append(review)
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer(max_features=5000)
    X=cv.fit_transform(corpus).toarray()

    y=pd.get_dummies(data['message'])
    y=y.iloc[:,1].values

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.20,random_state=0)

    from sklearn.naive_bayes import MultinomialNB
    spam_detect_model=MultinomialNB().fit(X_train,y_train)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = spam_detect_model.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)



#spam_detect=pickle.load(open('spam.pkl','rb'))
#print(spam_detect(["Hi, My name is  Nayeem"]))