import pickle
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from flask import Flask, render_template, request

app = Flask(__name__)
vectModel = pickle.load(open(r'vectorizer.pkl', 'rb'))
vecRecSys = pickle.load(open(r'ContentBasedTfIdf.pkl' , 'rb'))
tfIdfDataset = pd.read_csv(r'tfidfSinopsis.csv')
novelDataset = pd.read_csv(r'Dataset Novel Image.csv')


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        judul = request.form['judul']
        sinopsis = [request.form['sinopsis']]
        model_type = request.form['model']
        
        print(model_type)
        open = True

        if judul == '' or sinopsis == '':
            open = False

        vectext = vectModel.transform(sinopsis).toarray()
        model = choose_model(model_type=model_type)
        proba = model.predict_proba(vectext)
        prediction = np.round(proba*100)
        # prediction = [[90,2,3,5]]

        prediction = probabilityClass(prediction=prediction[0])
        recommend = getImage(sinopsis, pred=proba)
        
        # print('recomendation : ', recommend)
        # recommend = []

        return render_template('index.html', open=open, percentage=prediction, recommend=recommend)
    else:
        return render_template('index.html')

def probabilityClass(prediction):
    genres = {'Romance': 0, 'Horror': 0, 'Fantasi': 0, 'Sejarah': 0}
    
    for percentage, genre  in zip(prediction, genres.keys()):
        genres[genre] = percentage

    i,j = 0, 2

    tups = list(genres.items())
    tups[i], tups[j] = tups[j], tups[i]
    res = dict(tups)
    
    print(res)

    return genres

def getImage(vecSinopsis, pred):
    i,j = 0,2
    pred[0][i], pred[0][j] = pred[0][j], pred[0][i]
    vecSinopsis = np.concatenate((vecRecSys.transform(vecSinopsis).toarray(), pred), axis=1)
    vecSinopsis = np.squeeze(vecSinopsis, axis=0)
    cosine_similarity = dot(tfIdfDataset.values, vecSinopsis)/(norm(tfIdfDataset.values)*norm(vecSinopsis))
    novelDataset['Similarity'] = pd.DataFrame(cosine_similarity)
    title = novelDataset.sort_values(by=['Similarity'], ascending=False).Judul.iloc[:].values
    sinopsis = novelDataset.sort_values(by=['Similarity'], ascending=False).Sinopsis.iloc[:].values
    image = novelDataset.sort_values(by=['Similarity'], ascending=False).Image.iloc[:].values

    recomendation = pd.DataFrame(list(zip(title,sinopsis,image)),
                                 columns=['Title', 'Sinopsis', 'Source'])
    return recomendation

def choose_model(model_type='SVM'):
    if model_type == 'SVM':
        model = pickle.load(open(r'SVM.pkl', 'rb'))
    elif model_type == 'LR':
        model = pickle.load(open(r'LogisticRegression.pkl', 'rb'))
    elif model_type == 'RFC':
        model = pickle.load(open(r'RandomForest.pkl', 'rb'))
    elif model_type == 'XGB':
        model = pickle.load(open(r'XgBoost.pkl', 'rb'))
    else:
        model = pickle.load(open(r'NaiveBayes.pkl', 'rb'))
        
    return model

if __name__ == '__main__':
    app.run(debug=True)