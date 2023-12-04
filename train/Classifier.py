import pickle
from pathlib import Path
from Preprocessing import Preprocessing
from NovelGenreClassifier import NovelGenreClassifier

class Classifier:
    def __init__(self, sourcePath, stopwordPath, vectorizePath, model):
        self.sourcePath = sourcePath
        self.stopwordPath = stopwordPath
        self.vecPath = vectorizePath
        self.model = model
        self.classifier = None
        self.preprocessing = None
    
    def train(self):
        sourcePath = Path(self.sourcePath)
        stopwordPath = Path(self.stopwordPath)

        self.preprocessing = Preprocessing()
        data, vectorizer = self.preprocessing.process(sourcePath, stopwordPath)
        
        pickle.dump(vectorizer, open(self.vecPath / 'vectorizer.pkl', "wb"))
        print(f'vectorizer is saved in {self.vecPath}\ Vectorizer.pkl')

        X_train, X_test = data

        y_train = X_train.label
        X_train = X_train.drop(labels=['label'], axis=1)

        y_test = X_test.label
        X_test = X_test.drop(labels=['label'], axis=1)

        novelClassification = NovelGenreClassifier(model=self.model, vectorizer=vectorizer)

        novelClassification.fit(X_train, y_train)
        novelClassification.save_model(self.vecPath)
        print(f'Model Saved in {self.vecPath}\ NovelGenreClassifier.pkl')
              
        novelClassification.evalModel(X_test, y_test)

        self.classifier = novelClassification
    
    def predict(self, x):
        return self.classifier.predict(x)
