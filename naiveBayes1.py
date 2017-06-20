# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """  
          
        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
        
        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]
            
        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter 
        that gives the best accuracy on the held-out validationData.
        
        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.
        
        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"
        '''n_label = len(self.legalLabels)
        n_feature = len(self.features)
        n_training = len(trainingLabels)

        self.prior_p = util.Counter()
        for label in self.legalLabels:
            #prior probability
            self.prior_p[label] = float(trainingLabels.count(label))/float(n_training)

        self.con_p = util.Counter()
        #Laplace smoothing

        #for very k value
        for k in kgrid:
            self.con_p[k] = util.Counter()

            # in each label
            for label in self.legalLabels:
                self.con_p[k][label] = util.Counter()

                # for each possible feature
                for feature in self.features:
                    self.con_p[k][label][feature] = util.Counter()
                    #find numbers of time pixel F take value f in all label y

                    time = 0
                    i = 0
                    sum_time =0
                    for image in trainingData:
                        if trainingLabels[i] == label:
                            time = time + image[feature]
                            i = i+1
                            sum_time = sum_time + image.totalCount()
                    prob = float(time + k) / float(sum_time + k * n_feature)
                    self.con_p[k][label][feature]["y"] = prob
                    self.con_p[k][label][feature]["n"] = 1-prob

        #tie break
        k_best = kgrid[0]
        current_best = 0
        n_data = len(validationData)
        for k in kgrid:
            correct = 0
            self.k_best = k
            guesses = self.classify(validationData)
            i=0
            for guess in guesses:
                if guess == validationLabels[i]:
                    correct = correct+1
                i = i+1
            accuracy = float(correct) / float(n_data)
            if accuracy > current_best:
                current_best = accuracy
                k_best = k
            if accuracy == current_best:
                if k < k_best:
                    k_best = k

                    
        self.k_best = k_best
        '''
        numTraining = len(trainingLabels)
   
        self.prior_p = util.Counter()

        for label in self.legalLabels:
            self.prior_p[label] = trainingLabels.count(label) / (numTraining + 0.0)
            
        
        self.count = util.Counter()
        self.totalCount = util.Counter()
        self.totalCount.incrementAll(self.legalLabels,-1)
        self.con_p = util.Counter()
        numFeatures = len(self.features)
        
        #Create CPT
        for k in kgrid:
            self.con_p[k] = util.Counter()
            for label in self.legalLabels:
                
                if(self.count[label] == 0):
                    self.count[label] = util.Counter()
                    self.count[label].incrementAll(self.features,-1)
            
                self.con_p[k][label] = util.Counter()
                for feature in self.features:

                    self.con_p[k][label][feature] = util.Counter()
                    #if self.count[label] == 0 or self.count[label][feature] == -1:
              
                    i = 0
                    count = 0
                    totalCount = 0
                    for datum in trainingData:
                        if trainingLabels[i] == label:
                            count = count + datum[feature]
                            #print "datum[feature]",datum[feature]
                            totalCount = totalCount + datum.totalCount()
                            i = i + 1
                        self.count[label][feature] = count
                        self.totalCount[label] = totalCount
                    else:
                        count = self.count[label][feature]
   
                        if self.totalCount[label]==-1:
                            totalCount=0
                            i = 0
                            for datum in trainingData:
                                if trainingLabels[i] == label:
                                    totalCount = totalCount + datum.totalCount()
                                i+=1
                            self.totalCount[label]=totalCount
                        else:
                            totalCount = self.totalCount[label]
                    prob = (count + k + 0.0) / (totalCount + k * numFeatures)
                    self.con_p[k][label][feature]["y"] = prob
                    self.con_p[k][label][feature]["n"] = 1-prob
        
        #Evaluate Accuracy        
        k_best = kgrid[0]
        bestAccuracy = 0
        numData = len(validationData)
        for k in kgrid:
            correct = 0
            self.k_best = k
            guesses = self.classify(validationData)
            index=0
            for guess in guesses:
                if guess == validationLabels[index]:
                    correct = correct+1
                index+=1
            accuracy = correct / (numData + 0.0)
            print accuracy
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                k_best = k
            if accuracy == bestAccuracy:
                k_best = min(k_best, k)
                
        self.k_best = k_best

        util.raiseNotDefined()
        
    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
          posterior = self.calculateLogJointProbabilities(datum)
          guesses.append(posterior.argMax())
          self.posteriors.append(posterior)
        return guesses
      
    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.    
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
        
        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """
        logJoint = util.Counter()
        
        "*** YOUR CODE HERE ***"
        for label in self.legalLabels:
            prior = self.prior_p[label]
            sum = 0
            for feature in self.features:
                if datum[feature] != 0:
                    if self.con_p[self.k_best][label][feature]==0:
                        raise NameError("positive broke: "+label+" "+feature)
                    sum = sum + math.log(self.con_p[self.k_best][label][feature]["y"])
                else:
                    if self.con_p[self.k_best][label][feature]==0:
                        raise NameError("negative broke: "+label+" "+feature)
                    sum = sum + math.log(self.con_p[self.k_best][label][feature]["n"])
            logJoint[label] = math.log(prior) + sum
        return logJoint
        util.raiseNotDefined()
        
        return logJoint
      
    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2) 
        
        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []
           
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresOdds
        

    
      
