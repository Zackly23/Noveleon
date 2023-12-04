import os
from os.path import join

ROOT_PATH = os.getcwd()

'''
Vectorizer and Dataset Path
'''
class VectorDataset:
    VECTORIZER_MODEL_PATH = join(ROOT_PATH,'vectorizer.pkl')
    VECTORIZER_RECSYS_PATH = join(ROOT_PATH,'ContentBasedTfIdf.pkl')
    TFIDFDATASET_PATH = join(ROOT_PATH,'tfidfSinopsis.csv')
    NOVELDATASET_PATH = join(ROOT_PATH,'Dataset Novel Image.csv')

'''
Model Classifier Path
'''
class Model:
    SVM_MODEL_PATH = join(ROOT_PATH, 'model/SVM.pkl')
    LR_MODEL_PATH = join(ROOT_PATH, 'model/LogisticRegression.pkl')
    DT_MODEL_PATH = join(ROOT_PATH, 'model/DecisionTree.pkl')
    NBC_MODEL_PATH = join(ROOT_PATH, 'model/NaiveBayes.pkl')
    RF_MODEL_PATH = join(ROOT_PATH, 'model/RandomForest.pkl')
    XGB_MODEL_PATH = join(ROOT_PATH, 'model/XgBoost.pkl')

'''

'''