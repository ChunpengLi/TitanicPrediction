# Import necessary Modules
import pandas as pd
import scipy # This is a required module in "pandas". Also need to install this module before using "panda".
from sklearn.tree import DecisionTreeRegressor
import csv

print("\n...starting...")
# Import data of training
titanic_file_path = "train_quantified.csv"
titanic_data = pd.read_csv(titanic_file_path)

# Extract result and features
y = titanic_data.Survived  # This is the result for training

titanic_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']  # This is selected features
print("\n\nFeature", titanic_features, "are selected.")
X = titanic_data[titanic_features]  # Build a training set.

print("------First 5 samples in the training set:------")
print(X.head())  # Show first 5 samples with selected features
print("\nStatistic information of the training set:")
print(X.describe(),"\n")  # Analyse the statistic information of the training set.

# Build model
titanic_model = DecisionTreeRegressor(random_state=1)
titanic_model.fit(X, y)

# Self-prediction for Validating the ML model
selfPrediction_result_fractional = titanic_model.predict(X)
selfPrediction_result_Boolean = (selfPrediction_result_fractional >= 0.5) * 1

# Put "self-prediction result" & "actual result" into a List
selfPrediction_result_inList = []
for x in selfPrediction_result_Boolean:
    selfPrediction_result_inList.append(x)

actual_Survived_data = titanic_data['Survived']
actual_result_inList = []
for x in titanic_data['Survived']:
    actual_result_inList.append(x)


# Print "actual result", "predicted result" & "self prediction accuracy"
print("------ML model validation:------")
print("         Actual result:", actual_result_inList)  # Print "actual result"
print("Self-prediction result:", selfPrediction_result_inList, "\n")  # Print "self-prediction accuracy"
nTotal = len(actual_result_inList)  # Total number of samples
nCorrect = 0  # counter for correct prediction
for i in range(0, nTotal):
    if selfPrediction_result_inList[i] == actual_result_inList[i]:
        nCorrect += 1  # counter +1 if "predict = actual"
accuracy_precent = nCorrect/nTotal*100  # accuracy percentage of self-prediction
print("    Correct self-prediction:", nCorrect)
print("Quantity of self-prediction:", nTotal)
print("                   Accuracy: "+"%.4f" % accuracy_precent+" %\n")

# Predict data of testing set
titanic_testfile_path = "test_quantified.csv"
titanic_testdata = pd.read_csv(titanic_testfile_path)
X_test = titanic_testdata[titanic_features]  # Build a testing set.
print("------Statistic imformation of testing set:------")
print(X_test.describe(), "\n")
testPrediction_result_fractional = titanic_model.predict(X_test)
testPrediction_result_Boolean = (testPrediction_result_fractional >= 0.5) * 1

# Put "testPassengerId" & "testPrediction" into a list
testPrediction_result_inList = []
testPassengerId_inList = []
for i in range (0, len(testPrediction_result_Boolean)):
    testPassengerId_inList.append(titanic_testdata['PassengerId'][i])
    testPrediction_result_inList.append(testPrediction_result_Boolean[i])
print("------Test Prediction Result:------")
print("      PassengerId in testing set:", testPassengerId_inList)
print("Prediction result of testing set:", testPrediction_result_inList)

# Generate test result file
print("\n...Creating Output File...")
with open("gender_submission_CL.csv",'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['PassengerId','Survived'])
    for i in range (0, len(testPrediction_result_inList)):
        thewriter.writerow([testPassengerId_inList[i], testPrediction_result_inList[i]])


