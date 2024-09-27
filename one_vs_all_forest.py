import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class Forest1VsAllClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classes=[0, 1, 2, 3, 4, 5], n_trees=100, thresholds=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],weights=[0.8,1.0,1.0,1.0,1.0,1.0]):
        self.classes_ = classes
        self.classes = classes
        self.n_trees = n_trees
        self.thresholds = thresholds
        self.weights = weights 
        self.models = []
        
        # Get default parameters from DecisionTreeClassifier
        default_params = DecisionTreeClassifier().get_params()

        # Define all parameter names we want to include
        self.param_names = ['criterion', 'splitter', 'max_features', 'min_samples_split', 'min_samples_leaf', 'max_depth', 'random_state']

        # Create the matrix of parameters
        self.class_params_matrix = np.array([
        #'criterion', 'splitter','max_features', 'min_samples_split',   'min_samples_leaf',                   'max_depth',                  'random_state'
            ['entropy', 'random', 8, default_params['min_samples_split'], default_params['min_samples_leaf'], default_params['max_depth'], default_params['random_state']],
            ['entropy', 'random', 8, default_params['min_samples_split'], default_params['min_samples_leaf'], default_params['max_depth'], default_params['random_state']],
            ['entropy', 'random', 8, default_params['min_samples_split'], default_params['min_samples_leaf'], default_params['max_depth'], default_params['random_state']],
            ['entropy', 'random', 8, default_params['min_samples_split'], default_params['min_samples_leaf'], default_params['max_depth'], default_params['random_state']],
            ['entropy', 'random', 8, default_params['min_samples_split'], default_params['min_samples_leaf'], default_params['max_depth'], default_params['random_state']],
            ['entropy', 'random', 8, default_params['min_samples_split'], default_params['min_samples_leaf'], default_params['max_depth'], default_params['random_state']]
        ], dtype=object)

        # Create a dictionary to map classes to their parameters
        self.class_model_params = {
            cls: dict(zip(self.param_names, params))
            for cls, params in zip(self.classes, self.class_params_matrix)
        }

    def train_forest_for_single_class(self, target_class, train_data):
        """
        Train a forest of decision trees for the one-vs-all classification problem.
        """
        
        train_data = self.prepare_one_vs_all_equal_sampling(train_data, target_class)
        x_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]

        # Use the specific parameters for this target class
        params = self.class_model_params[target_class]
        # the  model params for each class defined in the constructor
        models = [DecisionTreeClassifier(**params) for _ in range(self.n_trees)]

        for i in range(self.n_trees):
            models[i].fit(x_train, y_train)

        return models  

    def train_multiple_forests(self, train_data):
        """
        Train multiple "One vs All" forests, one for each class. and return the models
        models is a list of lists, each list contains the models for one class
        """
        print()
        print("Know that u cane adjust parameters for each classifir in the constructor.")
        print()
        print("Training multiple One-vs-All forests. One for each class... ")
        models = []
        for cls in self.classes:
            print(f"Training model for class {cls}...")
            model = self.train_forest_for_single_class(cls, train_data)
            models.append(model)

        print("One-vs-All forests training completed.")
        print()
        return models

    def forest_prediction_for_single_class(self, models, x_test, target_class):
       """
       Predict class labels using the trained forest and calculate probabilities.
       my probability = number of votes for the target class / total number of trees
       """
       y_pred = [model.predict(x_test) for model in models]
       final_predictions = []
       for i in range(len(y_pred[0])):
           votes = [0, 0]
           for j in range(self.n_trees):
               if y_pred[j][i] == target_class:
                   votes[1] += 1
               else:
                   votes[0] += 1
           final_class = target_class if votes[1] >= votes[0] else -1
           probability = votes[1] / self.n_trees
           final_predictions.append((final_class, probability))
       return final_predictions

    def apply_threshold_on_single_class_predictions(self, predictions, threshold):
        """
        Apply a threshold to the predictions to determine the final class.
        """
        final_classes = []
        for pred, prob in predictions:
            if pred != -1 and prob >= threshold:
                final_classes.append((pred, prob))
            else:
                final_classes.append((-1, 0))
        return final_classes

    def get_prediction_from_all_forests(self, models, x_test):
        """
        Predict class labels for the test set using multiple forests.
        """
        all_predictions = []
        class_models_info = {}

        print("Predicting with One-vs-All different forests for each class AND applying thresholds...")
        print("Current thresholds are: ", self.thresholds)
        for i, model in enumerate(models):
            print(f"Predicting model for class {self.classes[i]}...")
            # Get the predictions for the current class
            predictions = self.forest_prediction_for_single_class(model, x_test, self.classes[i])
            # Apply the threshold to the predictions for the current class
            predictions = self.apply_threshold_on_single_class_predictions(predictions, self.thresholds[i])
            # final class predictions after applying the threshold without the probability
            predicted_classes = [pred for pred, _ in predictions]

            #concider to add the predicted_classes instade of predictions
            all_predictions.append(predictions)
            # all_predictions contains the predictions for all tets samples for all classes saparatedly
        print("Single class One-vs-All forests prediction completed.")
        print("search me and uncomment the print to get more info about each class model")
        print()
        #print(class_models_info)

        return all_predictions

    def get_final_prediction(self, all_predictions):
        """
        Make final predictions by combining the results of multiple forests.
        here we use the weights to give more importance to some classes
        """
        final_decisions = []
        num_samples = len(all_predictions[0])
        probabilities = np.zeros((num_samples, len(all_predictions)))

        
        print("applying weights and combining the results of all One-vs-All forests...")
        print("Current weights are: ", self.weights)
        print()
        for i in range(num_samples):
            class_votes = [0] * len(all_predictions)
            class_probabilities = [0.0] * len(all_predictions)

            for j in range(len(all_predictions)):
                predicted_class, probability = all_predictions[j][i]
                if predicted_class != -1:
                    class_votes[j] += 1
                    class_probabilities[j] = max(class_probabilities[j], probability * self.weights[j])
                probabilities[i][j] = class_probabilities[j]
                    

            if sum(class_votes) == 1:
                final_class = class_votes.index(1)
                
                
            elif sum(class_votes) > 1:
                final_class = class_probabilities.index(max(class_probabilities))
                
                
                
            else:
                final_class = -1
                

            final_decisions.append(final_class)
            

        print("One-vs-All Classifiers Decision Completed.")
        return final_decisions ,probabilities

    def fit(self, X_train, y_train):
        """
        Fit the model with training data.
        """

        y_train_named = pd.Series(y_train, name='label')
        train_data = pd.concat([pd.DataFrame(X_train), pd.Series(y_train_named)], axis=1)

        print()
        
        print("Initializing one-vs-all forests...")
        self.models = self.train_multiple_forests(train_data)

        print()
        return self
    # def predict(self, X_test, y_test):
    def predict(self, X_test):
        """
        Predict the class labels for test data.
        """
        print()
        print("One-vs-ALL predictions for test data...")
        print()
        # all_predictions,_ = self.get_prediction_from_all_forests(self.models, X_test, y_test)
        all_predictions = self.get_prediction_from_all_forests(self.models, X_test)
        final_decisions,_ = self.get_final_prediction(all_predictions)

        return final_decisions
    
    def predict_proba(self, X_test):
        """
        Compute and return the probabilities for each class in the test set.
        """
        print()
        print("One-vs-ALL probability predictions for test data...")
        print()

        # Get predictions (including probabilities) from all forests
        all_predictions = self.get_prediction_from_all_forests(self.models, X_test)

        # Extract probabilities from the final predictions
        _, final_probabilities = self.get_final_prediction(all_predictions)

        return final_probabilities

    #function to prepare a balanced dataset for one-vs-all classification using a two-step sampling process
    def prepare_one_vs_all_equal_sampling(self, df, target_class):
        """
        Prepare a balanced dataset for one-vs-all classification using a two-step sampling process.
        it will take the same number of target class samples from remaining classes tring to keep the same number of samples for each class as possible
        will return a new dataframe with label target_class and -1 for all other classes
        """
        target_df = df[df['label'] == target_class]
        num_target_samples = len(target_df)

        remaining_classes = df[df['label'] != target_class]['label'].unique()

        initial_samples_per_class = num_target_samples // len(remaining_classes)

        sampled_dfs = [target_df]
        remaining_samples_needed = num_target_samples
        extra_samples_classes = []

        for cls in remaining_classes:
            class_df = df[df['label'] == cls]
            num_samples_to_take = min(len(class_df), initial_samples_per_class)
            sampled_class_df = class_df.sample(n=num_samples_to_take, random_state=42)
            sampled_dfs.append(sampled_class_df)
            remaining_samples_needed -= num_samples_to_take

            if len(class_df) > initial_samples_per_class:
                extra_samples_classes.append(cls)

        if remaining_samples_needed > 0 and extra_samples_classes:
            extra_samples_per_class = remaining_samples_needed // len(extra_samples_classes)
            for cls in extra_samples_classes:
                class_df = df[df['label'] == cls]
                additional_needed = min(len(class_df) - initial_samples_per_class, extra_samples_per_class)
                if additional_needed > 0:
                    additional_samples_df = class_df.sample(n=additional_needed, random_state=42)
                    sampled_dfs.append(additional_samples_df)
                    remaining_samples_needed -= additional_needed

        final_df = pd.concat(sampled_dfs)
        final_df['label'] = np.where(final_df['label'] == target_class, target_class, -1)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        ##########################################################################################################
        #drope nan values
        if final_df.isnull().values.any():
            print("dropping nan values")
            final_df.dropna(inplace=True)
        ##########################################################################################################

        return final_df
    



























from sklearn.preprocessing import PolynomialFeatures, StandardScaler

 ##   Main function to orchestrate the training and prediction processes
def main():
    # Load your training and test data
    data = pd.read_csv('train_data.csv')
    x_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]
    # scaler and poly
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    poly = PolynomialFeatures(degree=3)
    x_train = poly.fit_transform(x_scaled)
    x_train = x_scaled


    

    # Initialize the Forest1VsAllClassifier
    classifier = Forest1VsAllClassifier()

    # Fit the model with training data
    classifier.fit(x_train, y_train)
    
    # Load your test data
    data = pd.read_csv('test_data.csv')
    x_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]
    # scaler and poly
    x_scaled = scaler.transform(x_test)
    x_test = x_scaled
    #x_test = poly.transform(x_scaled)
    # Make predictions for test data
    final_decisions= classifier.predict(x_test)
    #print total F1 score for all clases combine
    print(f"Total F1 Score: {f1_score(y_test, final_decisions, average='weighted' , zero_division=0):.2f}")
    # class report without class -1
    report = classification_report(y_test, final_decisions, zero_division=0)
    #print without class -1
    report = report.replace(' -1       ', '         ')
    print ("classification report:")
    print(report)

if __name__ == '__main__':
    main()