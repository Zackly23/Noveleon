import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from Preprocessing import Preprocessing

class NovelGenreClassifier:
    def __init__(self,
                 test_size: float = 0.2,
                 vectorizer=None,
                 model=None,
                 stopWordPath: Path='stopwordbahasa.csv'):
        
        self.transformer = Preprocessing()
        self.test_size = test_size
        self.model = model
        self.vec = vectorizer
        self.stopWordPath = stopWordPath

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame=None):          
        self.model.fit(X_train, y_train)

    
    def evalModel(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print('Model accuracy score with default hyperparameters: {0:0.3f}'.format(acc))
        
#         return acc

    def predict(self, X_test):
        test_data = pd.Series(X_test)
        cleanText = self.transformer.cleanText(test_data)
        stemmingText = self.transformer.stemmingText(cleanText)
        removeSW = self.transformer.removeStopword(stemmingText, self.stopWordPath)
        tfidf = self.vec.transform(removeSW).toarray()
        predict = self.model.predict(tfidf)
        
        return predict
    
    def save_model(self, path):
        pickle.dump(self.model, open(path /'NovelGenreClassifier.pkl', 'wb'))
        
