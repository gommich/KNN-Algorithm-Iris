# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys

class KNN():
    
    def fit(self, trainx, trainy,testx,testy,ranges,k):
        self.trainx=trainx
        self.trainy=trainy
        self.testx=testx
        self.testy=testy
        self.ranges=ranges
        self.k=k
         
    def dist(self,A,B):
        sum = 0
        for i in range (4):
            sum=sum+((A[i]-B[i])**2)/((self.ranges[i])**2)
            
        d = np.sqrt(sum)
        
        return d
    
    def getAccuracy(self, predictions):
        count = 0
        
        for i in range(len(predictions)):
            if (predictions[i]==self.testy[i]):
                count=count+1
        
        return count/len(predictions)
        
    def classify(self):
        predictions=[]
        
        for instance in self.testx:
            species = self.n_nearest_species(instance,self.k)
            predictions.append(species)
        
        return predictions
    
    def n_nearest_species(self, test_instance, k):
        distances = []
        s1 = "Iris-setosa"
        s2 = "Iris-versicolor"
        s3 = "Iris-virginica"
        
        
        
        #Calculate the distances and associate with class
        for i in range(len(self.trainx)):
            train_instance = self.trainx[i]
            actual_species = self.trainy[i]
            
            d=self.dist(train_instance,test_instance)
            T = (d, actual_species) #tuple of distance and associated species
            
            distances.append(T)
            
        distances = sorted(distances)
        
        
        n_nearest_distances = distances[0:k]
        
        species_count = {}
        species_count[s1]=0
        species_count[s2]=0
        species_count[s3]=0
        maxcount = 0
        maxspecies = ""
        
        for x in n_nearest_distances:
            
            if (x[1]==s1):
                species_count[s1]=species_count[s1]+1
                if (species_count[s1]>maxcount):
                    maxcount=species_count[s1]
                    maxspecies=s1
                
            elif (x[1]==s2):
                species_count[s2]=species_count[s2]+1
                if (species_count[s2]>maxcount):
                    maxcount=species_count[s2]
                    maxspecies=s2
            elif (x[1]==s3):
                species_count[s3]=species_count[s3]+1
                if (species_count[s3]>maxcount):
                    maxcount=species_count[s3]
                    maxspecies=s3
            
        return maxspecies
                    
    
                    
                    
 


if __name__ == "__main__":
    #loading the dataset
    
    names = ['s_len','s_width','p_len','p_width','class']
    training_data = pd.read_csv(sys.argv[1],delim_whitespace=True,names=names)
    test_data = pd.read_csv(sys.argv[2],delim_whitespace=True,names=names)
    
    #Splitting the train and test sets
    trainx=training_data.iloc[:,0:4].values
    trainy = training_data.iloc[:,4]
    testx = test_data.iloc[:,0:4].values
    testy = test_data.iloc[:,4]
    
    #Range of each feature in training set
    ranges = []
    
    for i in range (4):
       max_val = np.amax(trainx[:,i])
       min_val = np.amin(trainx[:,i])
       
       feature_range = max_val-min_val
       
       ranges.append(feature_range)
       
    
    
    newKNN = KNN()
    
    newKNN.fit(trainx,trainy,testx,testy,ranges,int(sys.argv[3]))
    
    predictions = newKNN.classify()
    
    accuracy = newKNN.getAccuracy(predictions)
    
    print("---- K =",sys.argv[3],"----")
    print("Accuracy:", accuracy)


    

    
    

    
    



        
        

        
        
            



