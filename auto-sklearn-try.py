import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# trying it on german data first since its smaller
trainingData = pd.read_csv("/home/mackenzie/Downloads/EnglishCleanedTrainingData.csv") # change this with new csv!
testingData = pd.read_csv("/home/mackenzie/Downloads/EnglishCleanedTestingData.csv")

# converting pandas dataframe to numpy array
trainingData = trainingData.as_matrix()
testingData = testingData.as_matrix()

# label encoder
labelencoder = LabelEncoder()
trainingData[0:, 0] = labelencoder.fit_transform(trainingData[0:, 0])
testingData[0:, 0] = labelencoder.fit_transform(testingData[0:, 0])

# our x is tweets and our y is the different classes
X_train = trainingData[0:, 0]
y_train = trainingData[0:, 1]
X_test = testingData[0:, 0]
y_test = testingData[0:, 1]

#For testing purposes
print(X_train[0]) # prints the first tweet from train
print(len(X_train)) # total num of tweets from train
print(X_test[1]) # prints the first tweet from test
print(len(X_test)) # total num of tweets in test


automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train, dataset_name='german_data') # feat_type=classification is another param we might use?
y_hat = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
automl.cv_results_
automl.sprint_statistics()
automl.show_models()
