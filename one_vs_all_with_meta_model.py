import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier





class Forest1VsAllClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,cv= 5, classes=[0, 1, 2, 3, 4, 5], n_trees=100, thresholds=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],weights=[0.8,1.0,1.0,1.0,1.0,1.0]):
        self.classes_ = classes
        self.scaler = StandardScaler()
        self.cv = cv
        self.flag = 1
        self.classes = classes
        self.n_trees = n_trees
        self.thresholds = thresholds
        self.weights = weights 
        self.models = []
        #self.final_model = DecisionTreeClassifier(random_state=42)
        self.final_model = LogisticRegression(random_state=42 , max_iter=1000)
        #self.final_model = RandomForestClassifier(random_state=42)
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
       #y pred is a list of arrays, each array contains the predictions of one tree
       #  a decision tree returns an array of predictions looks like [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for example
       final_predictions = []
       #len y_pred[0] is the number of samples in the test set ! for all samples
       for i in range(len(y_pred[0])):
           votes = [0, 0]
           #for all trees
           for j in range(self.n_trees):
               #y_pred[j][i] is the prediction of the jth tree for the ith sample
               if y_pred[j][i] == target_class:
                   votes[1] += 1
               else:
                   votes[0] += 1

           ##################################################################################################################
           final_class = target_class if votes[1] >= 1 else -1 # now even if one tree votes for the target class, it will be chosen
           ##################################################################################################################
           probability = votes[1] / self.n_trees
           final_predictions.append((final_class, probability))

           #final_predictions[sample_number][0] will return the class for sample_number
           #final_predictions[sample_number][1] will return the probability for sample_number
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
        #for each one-vs-all model
        for i, model in enumerate(models):
            print(f"Predicting model for class {self.classes[i]}...")
            # Get the predictions for the current class
            predictions = self.forest_prediction_for_single_class(model, x_test, self.classes[i])

            all_predictions.append(predictions)
            # all_predictions contains the predictions for all tets samples for all classes saparatedly
            # all_predictions[class_number][sample_number] will return the class and the probability for the sample_number
            #the length of all_predictions is the number of classes * the number of samples in the test set
        print("Single class One-vs-All forests prediction completed.")
        print("search me and uncomment the print to get more info about each class model")
        print()
        #print(class_models_info)
        # all_predictions[class_number][sample_number] will return the class and the probability for the sample_number
        return all_predictions

    def get_final_prediction(self, all_predictions):
        """
        Make final predictions by combining the results of multiple forests.
        here we use the weights to give more importance to some classes
        """
        final_decisions = []
        fina_probabilities = []
        num_samples = len(all_predictions[0])
        print ("len of all_predictions[0] is: ", num_samples)
        probabilities = np.zeros((num_samples, len(all_predictions)))

        
        print("applying weights and combining the results of all One-vs-All forests...")
        print("Current weights are: ", self.weights)
        print()
        #for each sample
        for i in range(num_samples):
            class_votes = [0] * len(all_predictions)
            class_probabilities = [0.0] * len(all_predictions)
            #for each class
            for j in range(len(all_predictions)):
                predicted_class, probability = all_predictions[j][i]
                if predicted_class != -1:
                    class_votes[j] += 1
                    #if 2 models vote for different classes, the class with the highest probability will be chosen
                    #store the probability from the model that voted for the class
                    class_probabilities[j] = probability * self.weights[j]

                probabilities[i][j] = class_probabilities[j]
                #probabilities[sample_number][class_number] = the probability of the class_number for the sample_number
                # the size of probabilities is the number of samples in the test set * the number of classes
                    
            #ch
            if sum(class_votes) == 1:
                final_class = class_votes.index(1)
                final_prob = class_probabilities[final_class]
                
                
            elif sum(class_votes) > 1:
                final_class = class_probabilities.index(max(class_probabilities))
                final_prob = max(class_probabilities)
                
                
                
            else:
                final_class = -1
                final_prob = 0.0

                

            final_decisions.append(final_class)
            fina_probabilities.append(final_prob)# the probability of the final class
         

            #final_decisions[sample_number] will return the final class for the sample_number
            #fina_probabilities[sample_number] will return the probability of the final class for the sample_number
            #probabilities[sample_number] will return the probabilities of all classes for the sample_number

        print("One-vs-All Classifiers Decision Completed.")
        return final_decisions , fina_probabilities , probabilities
    
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
        _, final_probabilities , probabilities = self.get_final_prediction(all_predictions)



        return final_probabilities , probabilities
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
        return final_df




    def fit(self, X_train, y_train):
        """
        Train the One-vs-All forests and the final decision tree with cross-validation, using pandas DataFrame.
        """

        if self.cv == 0:
            # No cross-validation, train on full data
            print("Training on full dataset...")
            y_train_named = pd.Series(y_train, name='label')

            # Concatenate X_train and y_train to form the full training data
            train_data = pd.concat([X_train.reset_index(drop=True), y_train_named.reset_index(drop=True)], axis=1)

            # Train One-vs-All models on full dataset
            self.models = self.train_multiple_forests(train_data)

            # Generate meta-features from the base models' predictions on the training data
            meta_features = self.generate_meta_features(X_train)

            # Train the final decision tree on the meta-features
            self.final_model.fit(meta_features, y_train)
            return self

        # Cross-validation with StratifiedKFold
        print(f"Training with {self.cv}-fold cross-validation...")
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        meta_features_list = []  # List to collect meta-features from each fold
        y_train_meta = []  # List to collect the labels from each fold

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"Training fold {fold_idx + 1}...")

            # Train and validation sets for current fold, using .iloc for row indexing
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Concatenate X_fold_train and y_fold_train into a single DataFrame for training
            fold_train_data = pd.concat([X_fold_train.reset_index(drop=True), pd.Series(y_fold_train, name='label').reset_index(drop=True)], axis=1)

            # Train the One-vs-All models on current fold
            self.models = self.train_multiple_forests(fold_train_data)

            # Generate meta-features from the base models' predictions on the validation data
            meta_features_val = self.generate_meta_features(X_fold_val)
            meta_features_list.append(pd.DataFrame(meta_features_val))  # Store meta-features from this fold as DataFrame
            y_train_meta.append(pd.Series(y_fold_val))  # Store the labels from this fold

        # After all folds, combine the meta-features and labels
        meta_features_combined = pd.concat(meta_features_list, axis=0).reset_index(drop=True)
        y_train_meta_combined = pd.concat(y_train_meta, axis=0).reset_index(drop=True)

        # Train the final decision tree on the combined meta-features
        print("Training final decision tree on all meta-features...")
        self.final_model.fit(meta_features_combined, y_train_meta_combined)

        print("Training completed.")
        return self




























    def predict(self, X_test):
        """
        Predict the class labels using the final decision tree.
        """
        print("Predicting test data using the trained One-vs-All forests...")




        # Generate meta-features from the base models' predictions on the test data
        meta_features = self.generate_meta_features(X_test)

        # Predict the final labels using the decision tree trained on the meta-features

        
        final_decisions = self.final_model.predict(meta_features)

        return final_decisions

    def generate_meta_features(self, X):
        """
        Generate meta-features by predicting from all base models.
        Each column in meta_features corresponds to the predictions of one One-vs-All forest.
        """
        finsl_classes , _ , probabilities = self.get_final_prediction(self.get_prediction_from_all_forests(self.models, X))
        # the meta features are as follows:
        #f0- the final class for the sample 
        #f1 -f6 the probabilities of the classes for the sample
        meta_features = np.zeros((len(X), len(self.classes) + 2))
        for i in range (len(X)):
            meta_features[i][0] = (finsl_classes[i]/5)
            for j in range(len(self.classes)):
                meta_features[i][j+1] = probabilities[i][j]



        #add the original features to the meta_features
       # meta_features = np.concatenate((X, meta_features), axis=1)

        if self.flag == 1:
            meta_features = self.scaler.fit_transform(meta_features)
            self.flag = 0
        else:
            meta_features = self.scaler.transform(meta_features)

        #add polynomial features
        # poly = PolynomialFeatures(2)
        # meta_features = poly.fit_transform(meta_features)

        
        #print example of meta_features
        print("Example of meta_features:")
        print(meta_features[:1])
        return meta_features














































def main():
    # Load your data (assuming data is a CSV file)
    data = pd.read_csv("data.csv")
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Labels

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize your One-vs-All Forest classifier
    one_vs_all_classifier = Forest1VsAllClassifier(classes=[0, 1, 2, 3, 4, 5], n_trees=100 , thresholds=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],weights=[1.0,1.0,1.0,1.0,1.0,1.0])

    # Fit the classifier on the training data
    print("Training the One-vs-All classifier...")
    one_vs_all_classifier.fit(X_train, y_train)

    # Predict on the test data
    print("Predicting the test data...")
    y_pred = one_vs_all_classifier.predict(X_test)# , y_train)

    # Evaluate the performance
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    


    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"Weighted F1 Score: {f1_weighted:.4f}")

if __name__ == "__main__":
    main()
