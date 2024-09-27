from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split  # Add this line
import numpy as np
import pandas as pd
# Custom file holds my classifier. You should use one of your own
from Custom import Classifier 
#from Custom import Classifier1 as Classifier
#from Custom import Classifier2 as Classifier
#from Custom import Classifier3 as Classifier




def compute_final_score(classifier_f1):
    classifier_f1 = round(100 * classifier_f1)
    if (classifier_f1 >= 90):
        return classifier_f1 + 4
    elif classifier_f1 >= 80:
        return classifier_f1 + 3
    elif classifier_f1 >= 70:
        return classifier_f1 + 2
    return classifier_f1

if __name__ == '__main__':

    data = pd.read_csv('data.csv')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    #save train and test data to csv files
    #df = pd.read_csv("C:\\Afeka\\Teaching\\intro_to_ML\\Final_project\\train_data.csv")
    df = train_data #pd.read_csv("train_data.csv")
    X_train = np.array(df.iloc[:, :-1])  # Extracting all rows and all columns except the last one
    y_train = np.array(df.iloc[:, -1])
    classifier = Classifier()

    # preprocess
    X_train = classifier.preprocess(X_train)
    classifier.fit(X_train, y_train)

    # df = pd.read_csv("C:\\Afeka\\Teaching\\intro_to_ML\\Final_project\\test_data.csv")
    df = test_data #pd.read_csv("test_data.csv")
    X_test = np.array(df.iloc[:, :-1])  # Extracting all rows and all columns except the last one
    Y_test = np.array(df.iloc[:, -1])

    # preprocess
    X_test = classifier.preprocess(X_test)
    classifier_predictions = classifier.predict(X_test)
    classifier_f1 = f1_score(classifier_predictions, Y_test, average='weighted')
    print(f"classifier f1: {classifier_f1:.2f}")

    final_score = compute_final_score(classifier_f1)
    print("Final project grade = " + str(final_score))

    #Additional info
    classifier_accuracy = accuracy_score(classifier_predictions, Y_test)
    print(f"classifier Accuracy: {classifier_accuracy:.2f}")
    print(classification_report(Y_test, classifier_predictions))