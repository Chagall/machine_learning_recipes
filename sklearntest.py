from sklearn import tree

class AppleOrangeClassifier:

    def __init__(self):
        self.features = [[140, 1], [130, 1], [150, 0], [170, 0]]    # Features : [ Weigth in grams, texture: 0 == bumpy and 1 == smooth]
        self.labels = [0, 0, 1, 1]                                  # Possible labels: Apple == 0 and Orange == 1
        self.dtc = tree.DecisionTreeClassifier()                    # Builds an empty Decision Tree Classifier

    def trainClassifier(self):
        self.dtc = self.dtc.fit(self.features, self.labels)

    def predictFruit(self,weigth,texture):
        fruitFeats = [[weigth,texture]]
        fruitLabel = self.dtc.predict(fruitFeats)
        return fruitLabel

    def showPredictionAnswer(self,label):
            if(label == 0):
                print "Fruit Label: 0"
                print "I think the fruit is... an Apple! Am I right?"
            elif(label == 1):
                print "Fruit Label: 1"
                print "I think the fruit is... an Orange! Am I right?"


appleOrangeClassifier = AppleOrangeClassifier()             # Creates an instance of our classifier
appleOrangeClassifier.trainClassifier()                     # Trains it with our pre loaded data
predictedLabel = appleOrangeClassifier.predictFruit(50,0)   # Predicts the label based on the weigth and texture given
appleOrangeClassifier.showPredictionAnswer(predictedLabel)  # And prints the result
