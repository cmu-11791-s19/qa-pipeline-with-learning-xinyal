from Classifier import Classifier
from sklearn import svm


# This is a subclass that extends the abstract class Classifier.
class SvmClassifier(Classifier):

    # The abstract method from the base class is implemented here to return an SVM classifier
    def buildClassifier(self, X_features, Y_train):
        clf = svm.SVC(gamma='scale').fit(X_features, Y_train) # TODO: try 1e-5, and simple svm.SVC
        return (clf)

