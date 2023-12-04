import pickle
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from src.config import Model
from src.config import VectorDataset

VECTORDATA = VectorDataset()
MODEL = Model()
vecRecSys = pickle.load(open(VECTORDATA.VECTORIZER_RECSYS_PATH , 'rb'))
tfIdfDataset = pd.read_csv(VECTORDATA.TFIDFDATASET_PATH)
novelDataset = pd.read_csv(VECTORDATA.NOVELDATASET_PATH)

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
        model = pickle.load(open(MODEL.SVM_MODEL_PATH, 'rb'))
    elif model_type == 'LR':
        model = pickle.load(open(MODEL.LR_MODEL_PATH, 'rb'))
    elif model_type == 'RFC':
        model = pickle.load(open(MODEL.RF_MODEL_PATH, 'rb'))
    elif model_type == 'XGB':
        model = pickle.load(open(MODEL.XGB_MODEL_PATH, 'rb'))
    else:
        model = pickle.load(open(MODEL.NBC_MODEL_PATH, 'rb'))
        
    return model