import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# Iris Flower classifier
class IrisClassifier:

    def __init__(self):
        # Loads the iris dataset into a local variable
        self.irisDataset = load_iris()

    def printIrisFeatureNames(self):
        print self.irisDataset.feature_names

    def printIrisTargetNames(self):
        print self.irisDataset.target_names

    def printWholeIrisDataset(self):
        for i in range(len(self.irisDataset.target)):
            print "Example[%d]: Label = %s, features %s" % (i, self.irisDataset.target[i], self.irisDataset.data[i])

    def buildTrainingDataSet(self):
        # Training Data: Here we are removing 1 entry of each species of iris flower
        # and setting train target and train data to not have those entries
        self.testIndex = [0,50,100]
        self.trainTarget = np.delete(self.irisDataset.target, self.testIndex)
        self.trainData = np.delete(self.irisDataset.data, self.testIndex, axis=0)

    def buildTestingDataSet(self):
        # Here we build our test dataset from those entries removed
        # from the training dataset
        self.testTarget = self.irisDataset.target[self.testIndex]
        self.testData = self.irisDataset.data[self.testIndex]

    def trainClassifier(self):
        # Here we build a decision tree classifier and train it with
        # our training dataset
        self.dtc = tree.DecisionTreeClassifier()
        self.dtc = self.dtc.fit(self.trainData, self.trainTarget)

    def predictIrisFlower(self):
        # We predict the iris flower type of the test data,
        # assuming that our classifier was already trained
        return self.dtc.predict(self.testData)

    def printPredictionResult(self):
        # We print the prediction result
        prediction = self.predictIrisFlower()
        print "Test target: "
        print self.testTarget
        print "Result target: "
        print prediction

irisClassifier = IrisClassifier()       # Create a new instance of our classifier
irisClassifier.buildTrainingDataSet()   # Build the training data set
irisClassifier.buildTestingDataSet()    # Build the test data set
irisClassifier.trainClassifier()        # Train the classifier with the train data set
irisClassifier.printPredictionResult()  # Prints the result of using the test data set on our already trained classifier
irisClassifier.printIrisFeatureNames()
irisClassifier.printIrisTargetNames()
# Exports the decision tree tho a readable pdf format
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(irisClassifier.dtc,
                        out_file = dot_data,
                        feature_names = irisClassifier.irisDataset.feature_names,
                        class_names = irisClassifier.irisDataset.target_names,
                        filled = True,
                        rounded = True,
                        impurity = False )
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris_tree.pdf")
