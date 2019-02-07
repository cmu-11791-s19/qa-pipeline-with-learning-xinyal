from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer # TfidfTransformer?


class TfidfFeaturizer(Featurizer):
    def getFeatureRepresentation(self, X_train, X_val):
        tfidf_vect = TfidfVectorizer()
        X_train_tfidf = tfidf_vect.fit_transform(X_train)
        X_val_tfidf = tfidf_vect.fit_transform(X_val) #.toarray() [[],[]]?
        return (X_train_tfidf, X_val_tfidf)






