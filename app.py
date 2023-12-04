import pickle
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from flask import Flask, render_template, request
from src.post_process import getImage, choose_model, probabilityClass
from src.config import VectorDataset

app = Flask(__name__)
VECTORDATA = VectorDataset()
vectModel = pickle.load(open(VECTORDATA.VECTORIZER_MODEL_PATH, 'rb'))

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

        prediction = probabilityClass(prediction=prediction[0])
        recommend = getImage(sinopsis, pred=proba)
        
        return render_template('index.html', open=open, percentage=prediction, recommend=recommend)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)