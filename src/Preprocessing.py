import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

class Preprocessing:
    def __init__(self, max_features=5000):
        self.df = None
        self.X = None
        self.y = None
        self.labels = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()
        self.stop_factory = StopWordRemoverFactory()
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def process(self, sourcePath, stopwordPath):
        self.df = pd.read_csv(sourcePath)
        self.splitDataset(self.df)
        
        self.X_train = self.processData(self.X_train, self.y_train, stopwordPath, 'train')
        self.X_test = self.processData(self.X_test, self.y_test, stopwordPath, 'test')
        
        return (self.X_train, self.X_test), self.vectorizer
        
    def processData(self, X, y, stopWordPath, phase=None):
        X = self.cleanText(X)
        X = self.stemmingText(X)
        X = self.removeStopword(X, stopWordPath)
        if phase == 'train':
            X = self.tfidf(X)
        elif phase == 'test':
            X = self._transform(X)
        X = self.transformDataframe(X, y)
        
        return X
        
    def cleanText(self, X):
        dataSinopsis = X.apply(lambda x: self._clean(x))
        
        return dataSinopsis
        
    def _clean(self, x):
        x = x.lower()
        x = re.sub(r'([->]+) *', ' ', str(x))
        x = re.sub(r'([".?!%-,]+) *', ' ', str(x))
        x = re.sub(' +', ' ', str(x))
        x = re.sub(r'\d+','',str(x))
        x = re.sub(r'(\w)\1(\1+)',r'\1',str(x))
        
        return x

    def stemmingText(self, x):
        hasil = list()
        for text in range(len(x)):
          result = self.stemmer.stem(x.values[text])
          hasil.append(result)
        
        return hasil
    
    def removeStopword(self, x, stopWordPath):
        more_stopword = list(pd.read_csv(stopWordPath).values.squeeze())
        data = self.stop_factory.get_stop_words() + more_stopword
        # stopword = stop_factory.create_stop_word_remover()
        dictionary = ArrayDictionary(data)
        remover = StopWordRemover(dictionary)
        cleanText = []

        for text in x:
            cleaningText = remover.remove(text)
            cleanText.append(cleaningText)
            
        return cleanText
    
    def tfidf(self, x):
        vecSinopsis = self.vectorizer.fit(x)
        xSinopsis = self._transform(x)
        
        return xSinopsis
    
    def _transform(self, x):
        xSinopsis = self.vectorizer.transform(x).toarray()
        
        return xSinopsis
    
    def transformDataframe(self, x, y):
        self.labels = {'Fantasi':0, 'Horor':1, 'Romance':2, 'Sejarah':3}
        dataset = pd.DataFrame(x)
        dataset['label'] = y.reset_index().drop(labels=['index'], axis=1)
        dataset['label'] = dataset.label.map(self.labels)
        
        return dataset
    
    def splitDataset(self, X: pd.DataFrame):
        dataValue = X.drop(labels=['Genre'], axis=1)
        targetValue = X.Genre

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataValue,
                                                                 targetValue,
                                                                 test_size = 0.2,
                                                                 random_state = 42)
        
        self.X_train = self.X_train.Sinopsis
        self.X_test = self.X_test.Sinopsis
