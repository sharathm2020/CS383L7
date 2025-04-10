import numpy as np
import pandas as pd
import os
import math
from PIL import Image
import matplotlib.pyplot as plt
import random

#Naive Bayes Probabilistic Classifier
class NaiveBayesClassifier:
    #Initialize the classifier as well as the class priors, conditionals, classes, feature means, feature count, and binary flag
    def __init__(self):
        self.class_priors = {}
        self.conditionals = {}
        self.classes = None
        self.feature_means = None
        self.feature_count = None
        self.is_binary = None
    
    #Train function for the classifier
    def fit(self, X_train, y_train, is_binary=True):
        self.is_binary = is_binary
        self.classes = np.unique(y_train)
        self.feature_count = X_train.shape[1]
        
        #Calculate the class priors P(y)
        total_samples = len(y_train)
        for cls in self.classes:
            self.class_priors[cls] = np.sum(y_train == cls) / total_samples
        
        #For the CTG dataset, we have binary features.
        #For these features, we want to calculate P(x_i=1|y) for each feature and class
        if is_binary:
            self.conditionals = {}
            for cls in self.classes:
                class_samples = X_train[y_train == cls]
                #Compute the prob that each feature is 1 given the class
                self.conditionals[cls] = {}
                for feature_idx in range(self.feature_count):
                    #Add Laplace smoothing to avoid zero probabilities
                    feature_true_count = np.sum(class_samples[:, feature_idx]) + 1
                    feature_count = len(class_samples) + 2  # add 2 for smoothing (true & false)
                    self.conditionals[cls][feature_idx] = feature_true_count / feature_count
        else:
            #For the Yale Faces dataset, we have continuous features
            self.conditionals = {}
            for cls in self.classes:
                class_samples = X_train[y_train == cls]
                self.conditionals[cls] = {}
                
                #Add laplace smoothing to avoid zero probabilities
                for feature_idx in range(self.feature_count):
                    value_counts = np.zeros(256)
                    for value in range(256):
                        count = np.sum(class_samples[:, feature_idx] == value) + 1
                        value_counts[value] = count
                    
                    #Normalize our values to get probabilities
                    total = np.sum(value_counts)
                    self.conditionals[cls][feature_idx] = value_counts / total
    

    #Predict function using the log-probability trick
    def predict(self, X):
        predictions = []
        
        for sample in X:
            #Calculate log probabilities for each class
            log_probs = {}
            for cls in self.classes:
                log_prob = math.log(self.class_priors[cls])
                
                #Add log likelihoods for each feature
                if self.is_binary:
                    for feature_idx in range(self.feature_count):
                        feature_val = sample[feature_idx]
                        #P(x_i|y) if x_i=1, or 1-P(x_i|y) if x_i=0
                        prob = self.conditionals[cls][feature_idx] if feature_val == 1 else 1 - self.conditionals[cls][feature_idx]
                        #Compute log probabilities
                        log_prob += math.log(max(prob, 1e-10))
                else:
                    for feature_idx in range(self.feature_count):
                        feature_val = int(sample[feature_idx])
                        prob = self.conditionals[cls][feature_idx][feature_val]
                        log_prob += math.log(max(prob, 1e-10))
                
                log_probs[cls] = log_prob
            
            #Choose the highest log probability
            predicted_class = max(log_probs, key=log_probs.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)

    #Function to evaluate the classifier
    def evaluate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        accuracy = np.mean(y_pred == y_val)
        
        #Create a confusion matrix
        confusion_matrix = np.zeros((len(self.classes), len(self.classes)))
        for i in range(len(y_val)):
            true_idx = np.where(self.classes == y_val[i])[0][0]
            pred_idx = np.where(self.classes == y_pred[i])[0][0]
            confusion_matrix[true_idx, pred_idx] += 1
        
        return accuracy, confusion_matrix

#Function to standardize the data for the CTG dataset
def preprocess_ctg_data(X, feature_means=None):
    if feature_means is None:
        feature_means = np.mean(X, axis=0)
    
    #Convert features to binary (1 if above mean, 0 otherwise)
    X_binary = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_binary[:, i] = (X[:, i] > feature_means[i]).astype(int)
    
    return X_binary, feature_means

#Function to load the CTG dataset
def load_ctg_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    X = df.iloc[:, :-2].values  
    X = df.iloc[:, :-2].values  
    y = df.iloc[:, -1].values   
    return X, y

#Function to load the Yale Faces Data
def load_yale_faces(directory):
    images = []
    labels = []
    
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.startswith("subject"):
            try:
                #Extract the subject number from the filename and convert it to an integer
                subject_info = filename.split('.')[0]
                subject_id = int(subject_info.replace('subject', ''))
                
                # Load and resize the image
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).convert('L')
                img = img.resize((40, 40))
                
                # Flatten the image into a feature vector
                img_array = np.array(img).flatten()
                
                images.append(img_array)
                labels.append(subject_id)
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    
    return np.array(images), np.array(labels)

#Shuffle and split the data into training and validation sets
def split_data(X, y, train_ratio=2/3):
    indices = list(range(len(y)))
    random.shuffle(indices)
    split = int(train_ratio * len(indices))
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    return X_train, X_val, y_train, y_val

#Shuffle and split the data into training and validation sets for the Yale Faces dataset
def split_data_by_subject(X, y, train_ratio=2/3):
    X_train, X_val = [], []
    y_train, y_val = [], []
    
    subjects = np.unique(y)
    
    for subject in subjects:
        subject_indices = np.where(y == subject)[0]
        np.random.shuffle(subject_indices)
        split = int(train_ratio * len(subject_indices))
        
        X_train.extend(X[subject_indices[:split]])
        X_val.extend(X[subject_indices[split:]])
        y_train.extend(y[subject_indices[:split]])
        y_val.extend(y[subject_indices[split:]])
    
    return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

def main():
    #Load, split and process the CTG dataset
    #Train and evaluate the Naive Bayes classifier
    #Get accuracy and confusion matrix
    #Print results

    X, y = load_ctg_data('CTG.csv')
    
    X_train, X_val, y_train, y_val = split_data(X, y)
    
    X_train_binary, feature_means = preprocess_ctg_data(X_train)
    X_val_binary, _ = preprocess_ctg_data(X_val, feature_means)
    
    #Train and evaluate the Naive Bayes classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train_binary, y_train)
    
    accuracy, confusion_matrix = nb_classifier.evaluate(X_val_binary, y_val)
    
    print(f"CTG Dataset - Validation Accuracy: {accuracy:.4f}")
    print('Confusion Matrix:')
    print(confusion_matrix)
    
    #Load, split and process the yale faces dataset
    #Train and evaluate the Naive Bayes classifier
    #Get accuracy and confusion matrix
    #Print Results

    X_faces, y_faces = load_yale_faces('yalefaces')
    
    X_train_faces, X_val_faces, y_train_faces, y_val_faces = split_data_by_subject(X_faces, y_faces)
    
    nb_faces = NaiveBayesClassifier()
    nb_faces.fit(X_train_faces, y_train_faces, is_binary=False)
    
    accuracy_faces, confusion_matrix_faces = nb_faces.evaluate(X_val_faces, y_val_faces)
    
    print(f"Yalefaces Dataset - Validation Accuracy: {accuracy_faces:.4f}")
    print('Confusion Matrix:')
    print(confusion_matrix_faces)

if __name__ == "__main__":
    main()
