import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, classification_report
import pandas as pd


class CustomRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trees=50, sample_size=0.9, class_weight='balanced', criterion='gini', splitter='best', 
                 max_depth=18, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=8, min_impurity_decrease=0.0, random_state=None, max_leaf_nodes=None, ccp_alpha=0.0):
        """
        Initialize the CustomRandomForestClassifier class.

        Parameters:
        - n_trees: Number of decision trees in the random forest.
        - sample_size: Fraction of the training data used to train each tree.
        - class_weight: Weights associated with classes.
        - criterion: Function to measure the quality of a split ('gini' or 'entropy').
        - splitter: Strategy used to split at each node ('best' or 'random').
        - max_depth: Maximum depth of each tree.
        - min_samples_split: Minimum number of samples required to split an internal node.
        - min_samples_leaf: Minimum number of samples required to be at a leaf node.
        - min_weight_fraction_leaf: Minimum weighted fraction of samples required to be at a leaf node.
        - max_features: Number of features to consider when looking for the best split.
        - min_impurity_decrease: Minimum decrease in impurity required to split a node.
        - random_state: Random seed for reproducibility.
        - max_leaf_nodes: Maximum number of leaf nodes.
        - ccp_alpha: Complexity parameter used for cost-complexity pruning.
        """
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.class_weight = class_weight
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.ccp_alpha = ccp_alpha
        self.trees = []
        self.classes_ = None

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        This is necessary for the custom classifier to be used in scikit-learn pipelines and models like StackingClassifier.
        """
        return {
            'n_trees': self.n_trees,
            'sample_size': self.sample_size,
            'class_weight': self.class_weight,
            'criterion': self.criterion,
            'splitter': self.splitter,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'min_impurity_decrease': self.min_impurity_decrease,
            'random_state': self.random_state,
            'max_leaf_nodes': self.max_leaf_nodes,
            'ccp_alpha': self.ccp_alpha
        }

    def set_params(self, **params):
        """
        Set parameters for this estimator.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """
        Train the random forest by creating multiple decision trees with random subsets of the data.

        Parameters:
        - X: The training data features.
        - y: The training data labels.

        Returns:
        - self: The fitted model.
        """

        #convert X and y to a pandas DataFrame
        print()
        print()
        print("Inishalizing Custom Random Forest Classifier training...")
        print()
        X = pd.DataFrame(X)
        y = pd.Series(y)
        self.trees = []  # Reset trees
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), size=int(len(X) * self.sample_size), replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]

            # Create and train the decision tree
            tree = DecisionTreeClassifier(
                class_weight=self.class_weight,
                criterion=self.criterion,
                splitter=self.splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes,
                ccp_alpha=self.ccp_alpha
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        # Save the classes
        self.classes_ = np.unique(y)
        print("Custom Random Forest Classifier training completed")
        return self

    def predict(self, X):
        """
        Predict the class labels for the test data using majority voting across all trees.

        Parameters:
        - X: The test data features.

        Returns:
        - y_pred_final: The predicted class labels based on majority voting.
        """
        print("Predicting test data using Custom Random Forest Classifier...")
        predictions = []

        # Collect predictions from each tree
        for tree in self.trees:
            pred = tree.predict(X)
            predictions.append(pred)

        # Ensure that predictions are not empty
        if len(predictions) == 0:
            return np.array([])
            raise ValueError("No predictions were made by any of the trees in the forest.")

        # Convert predictions to a numpy array
        predictions = np.array(predictions)

        # Initialize an empty list to hold the final predicted class for each sample
        y_pred_final = []
        num_classes = len(self.classes_)  # Number of classes
        # Perform majority voting
        for i in range(predictions.shape[1]):  # Iterate over each sample
            votes = [0] * num_classes  # Initialize votes array for each class (assuming 6 classes: 0 to 5)
            for j in range(predictions.shape[0]):  # Iterate over predictions from each tree
                votes[predictions[j][i]] += 1  # Increment the vote for the predicted class
            y_pred_final.append(votes.index(max(votes)))  # Select the class with the most votes

        print("Prediction using Custom Random Forest Classifier completed.")
        return y_pred_final



    # def predict(self, X):
    #     """
    #     Predict the class labels for the test data using majority voting across all trees.

    #     Parameters:
    #     - X: The test data features.

    #     Returns:
    #     - y_pred: The predicted class labels.
    #     """
    #     #check if x is all integer if not print a worning
    #     if not np.issubdtype(X.dtype, np.integer):
    #         print("Warning: X is not integers")
    #         print("found type: ", X.dtype)
    #         print("u can see the values in the next line")
    #         print(X)
    #         print("###########################################################################################")
    #     print()
    #     print("Predicting test data using Custom Random Forest Classifier...")
    #     predictions = []
    #     for tree in self.trees:
    #         pred = tree.predict(X)
    #         predictions.append(pred)
    #         #check if predictions is integer or if it empty if not print a worning
    #         if not np.issubdtype(pred.dtype, np.integer) or len(pred)==0:
    #             print("Warning: predictions are not integers")
    #             print("###########################################################################################")
    #             #
    #         #ch
    #         #predictions.append(tree.predict(X))

        

    #     # Convert predictions to a numpy array and take the majority vote
    #     predictions = np.array(predictions)
        
        
    #     #y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    #     y_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions)
    #     print("Prediction using Custom Random Forest Classifier completed")
    #     print()
    #     print()

    #     return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the test data using the proportion of votes for each class across all trees.

        Parameters:
        - X: The test data features.

        Returns:
        - proba: The predicted class probabilities (n_samples, n_classes).
        """

        # Collect predictions from all trees
        predictions = []
        for tree in self.trees:
            pred = tree.predict(X)
            predictions.append(pred)
            #check if predictions is integer if not print a worning
       # Convert predictions to numpy array for easier processing
        predictions = np.array(predictions)

        # Number of classes (assuming classes are numbered from 0 to n_classes-1)
        n_classes = 6# len(np.unique(predictions))

        # Create a zero matrix to store probabilities
        proba = np.zeros((X.shape[0], n_classes))

        # For each sample, count the votes for each class
        for i in range(X.shape[0]):  # loop over samples
            class_counts = np.bincount(predictions[:, i], minlength=n_classes)  # count votes for each class
            proba[i] = class_counts / self.n_trees  # normalize to get probabilities (votes / number of trees)

        return proba
























#use this if to check the model saperately
def main():
    # Load the training and test data
    data= pd.read_csv("train_data.csv")
    X_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]
    data = pd.read_csv("test_data.csv")
    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]
    y_pred = []
    classifier = CustomRandomForestClassifier()
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_micro = f1_score(y_test, y_pred, average='weighted')
    print(f"Micro-Averaged F1 Score: {f1_micro:.4f}")
    print(classification_report(y_test, y_pred))
    print("Done")
    print(classifier.predict_proba(X_test))

if __name__ == '__main__':
    main()
    