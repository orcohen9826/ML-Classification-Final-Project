import warnings
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from Custom_Random_Forest import CustomRandomForestClassifier
from one_vs_all_forest import Forest1VsAllClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, ClassifierMixin



#######################################################################################################################################                          
######################################-------------------Classifier-------------------##################################################
########################################################################################################################################

'''
2 classes are defined in this section: CustomStackingClassifier and Classifier.
The CustomStackingClassifier class is a custom stacking classifier that uses cross-validation to train 
the base models and the final estimator. 
The Classifier class is a custom classifier that uses the CustomStackingClassifier as the model. 
'''

#-------------------Custom Stacking Classifier-------------------#
'''
The CustomStackingClassifier class is a custom stacking classifier that uses cross-validation to train the base 
models and the final estimator. The base models are trained on the training data, and the predictions from the base
models are used as meta-features to train the final estimator. The final estimator is trained on the meta-features

'''
class CustomStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, final_estimator, cv):
        self.cv = cv
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.base_models = [model for _, model, _ in estimators]
        self.preprocess = [preprocess for _, _, preprocess in estimators]

    def fit(self, X, y):
        """
        Fit the base models and the final estimator using stacking with cross-validation.
        """
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        meta_features_list = []
        y_meta_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Training fold {fold_idx + 1}...")

            X_train1, X_val1 = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            X_train = X_train1.copy()
            X_val = X_val1.copy()

            fold_meta_features = []

            for (model_name, model, preprocess), preprocess_steps in zip(self.estimators, self.preprocess):
                print(f"Training base model {model_name}...")
                
                X_train_copy = X_train.copy()
                X_val_copy = X_val.copy()
                # Apply preprocessing for each model
                if preprocess_steps[0]:
                    X_train_copy = preprocess_steps[0].fit_transform(X_train_copy)
                    X_val_copy = preprocess_steps[0].transform(X_val_copy)
                
                if preprocess_steps[1]:
                    X_train_copy = preprocess_steps[1].fit_transform(X_train_copy)
                    X_val_copy = preprocess_steps[1].transform(X_val_copy)
                print("prepering data for model", model_name)
                print("X_train", X_train_copy.shape)
                # Train the model on training data
                model.fit(X_train_copy, y_train)
                print("Model", model_name, "fitted")

                # Generate predictions for the validation set (meta-features)
                val_preds =model.predict_proba(X_val_copy) if hasattr(model, 'predict_proba') else model.predict(X_val_copy)
                fold_meta_features.append(pd.DataFrame(val_preds))

            # Combine meta-features for all models
            meta_features_list.append(pd.concat(fold_meta_features, axis=1))
            y_meta_list.append(pd.Series(y_val))

        # After collecting meta-features from all folds, combine them
        meta_features_combined = pd.concat(meta_features_list).reset_index(drop=True)
        y_meta_combined = pd.concat(y_meta_list).reset_index(drop=True)

        # Fit the final estimator on the combined meta-features
        print("Training final estimator on meta-features...")
        self.final_estimator.fit(meta_features_combined, y_meta_combined)

        return self
   
    def generate_meta_features(self, X):
        """
        Generate meta-features for the final estimator using the trained base models.
        """
        meta_features = []
    
        for (model_name, model, preprocess), preprocess_steps in zip(self.estimators, self.preprocess):
            print(f"Generating meta-features with base model {model_name}...")
    
            # Apply preprocessing for each model
            X_transformed = X.copy()
            if preprocess_steps[0]:
                X_transformed = preprocess_steps[0].transform(X_transformed)
            
            if preprocess_steps[1]:
                X_transformed = preprocess_steps[1].transform(X_transformed)
    
            # Generate predictions (meta-features)
            model_preds = model.predict_proba(X_transformed) if hasattr(model, 'predict_proba') else model.predict(X_transformed)
            meta_features.append(pd.DataFrame(model_preds))
    
        # Combine meta-features for all base models
        meta_features_combined = pd.concat(meta_features, axis=1).reset_index(drop=True)
        return meta_features_combined
    
    def predict(self, X):
        """
        Generate predictions using the stacked models. 
        First, generate meta-features from base models, then use the final estimator for predictions.
        """
        # Generate meta-features from the base models
        print("Generating meta-features for prediction...")
        meta_features = self.generate_meta_features(X)

        # Use the final estimator to predict based on the meta-features
        print("Predicting with final estimator...")
        predictions = self.final_estimator.predict(meta_features)

        return predictions


#------------------- Custom Classifier -------------------#

'''
The Custom_classifier class is a custom classifier that uses a custom stacking classifier as the model. 
The base models are:
       1: CustomRandomForestClassifier - more details in Custom_Random_Forest.py
       2: Forest1VsAllClassifier - more details in one_vs_all_forest.py
       3: KNeighborsClassifier - KNN as base model with weird parameters chosen by me witout deep analysis
       4: LogisticRegression - Logistic

The final estimator is a CustomRandomForestClassifier. The model is trained on the training data and used to predict the test data.
F1 score: 0.94
'''
class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):     
        # Base models
        #self.custom_rf = CustomRandomForestClassifier(n_trees=50, sample_size=0.9, max_depth=18, max_features=8)
        self.custom_rf = CustomRandomForestClassifier(n_trees=100, sample_size=0.9, max_depth=18, max_features=0.8)
        self.forest_1vsall = Forest1VsAllClassifier(classes=[0, 1, 2, 3, 4, 5], n_trees=100)
        self.knn = KNeighborsClassifier(n_neighbors=50, weights='distance', p=1 , n_jobs=-1)  # KNN as base model
        self.log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)  # Added Logistic Regression
        
        #self.decision_tree = DecisionTreeClassifier(max_depth=10, random_state=42)  # Decision Tree as final estimator
        self.final_estimat = CustomRandomForestClassifier(n_trees=100, sample_size=0.9, max_depth=18, max_features= 0.9)
        self.preprocess_params = [(StandardScaler(), None), (StandardScaler(), None ), (StandardScaler(), PolynomialFeatures(2)), (StandardScaler(),PolynomialFeatures(3))]
        # Stacking model with added KNN and Logistic Regression, final decision by Decision Tree
        self.model = CustomStackingClassifier(
            estimators=[
                ('custom_rf', self.custom_rf ,self.preprocess_params[0]),  
                ('forest_1vsall', self.forest_1vsall ,self.preprocess_params[1]),  
                ('knn', self.knn ,self.preprocess_params[2]),  
                ('log_reg', self.log_reg ,self.preprocess_params[3])
      
            ],
            final_estimator=self.final_estimat,  #custom random forest
            cv=5
        )

    def preprocess(self, X):
        return X
    
    def fit(self, X, y):
        # Fit the model
        #convert x,y to dataframe keeping the column names f0 -f9 , Lable
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        y = pd.Series(y, name='Label')
        

        self.model.fit(X, y)
        print("Model fitted")
        return self
    
    def predict(self, X):
        # Predict
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        return self.model.predict(X)
    
    def predict_proba(self, X):
        # Predict probabilities
        return self.model.predict_proba(X)
    

########################################################################################################################################



















## The following classifiers are not under the project's guidelines, but they are provided for comparison purposes. ##






####################################################################################################################################################
###########################################   ----  Classifier1  ----   ############################################################################
####################################################################################################################################################
'''
The Classifier1 class is a custom classifier that uses a stacking classifier with the following base models:
    1: CustomRandomForestClassifier - more details in Custom_Random_Forest.py
    2: Forest1VsAllClassifier - more details in one_vs_all_forest.py
    3: KNeighborsClassifier - KNN as base model with weird parameters chosen by me witout deep analysis
    4: LogisticRegression - Logistic

The final estimator is a CustomRandomForestClassifier. The model is trained on the training data and used to predict the test data.
F1 score: 0.95
'''

from sklearn.ensemble import StackingClassifier
class Classifier1(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        
        # Base models
        #self.custom_rf = CustomRandomForestClassifier(n_trees=50, sample_size=0.9, max_depth=18, max_features=8)
        self.custom_rf = CustomRandomForestClassifier(n_trees=100, sample_size=0.9, max_depth=18, max_features=0.8)
        self.forest_1vsall = Forest1VsAllClassifier(classes=[0, 1, 2, 3, 4, 5], n_trees=100)
        self.knn = KNeighborsClassifier(n_neighbors=50, weights='distance', p=1 , n_jobs=2)  # KNN as base model
        self.log_reg = LogisticRegression(max_iter=1000, solver='liblinear', class_weight={0: 1, 1: 1, 2: 2.5, 3: 2.5, 5: 1}, C=10)  # Added Logistic Regression
        
        #self.decision_tree = DecisionTreeClassifier(max_depth=10, random_state=42)  # Decision Tree as final estimator
        self.final_estimat = CustomRandomForestClassifier(n_trees=100, sample_size=0.9, max_depth=18, max_features= 0.9)
        # Stacking model with added KNN and Logistic Regression, final decision by Decision Tree
        self.model = StackingClassifier(
            estimators=[
                ('custom_rf', self.custom_rf),
                ('forest_1vsall', self.forest_1vsall),
                ('knn', self.knn),  # KNN as base model
                ('log_reg', self.log_reg)  # Logistic Regression as base model
            ],
            final_estimator=self.final_estimat,  # Decision Tree as final decision model
            cv=5
        )

    def preprocess(self, X):
        return X  # Return X without modification

    def fit(self, X, y):
        print("Preprocessing training data...")
        X_scaled = self.scaler.fit_transform(X)
        X_poly = self.poly.fit_transform(X_scaled)
        
        print("Training the model, this may take a while plz wait and follow the console")
        print(" it will use cross validation to train the model so u will see a circular printing just follow the first one")
        self.model.fit(X_poly, y)
        return self

    def predict(self, X):
        print("\nPreprocessing test data...")
        X_scaled = self.scaler.transform(X)
        X_poly = self.poly.transform(X_scaled)
        
        print("Predicting...")
        predictions = self.model.predict(X_poly)
        print("\n" + "="*80)
        print("These results were achieved using custom models and models permitted according to the project's guidelines.\n"
              "The sklearn.ensemble.StackingClassifier was used, but if this is considered an external function,\n"
              "you can modify the final_project_test file to import Classifier2, which has a built-in function\n"
              "that works even better.")
        print("="*80 + "\n")

        return predictions

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_poly = self.poly.transform(X_scaled)
        return self.model.predict_proba(X_poly)
####################################################################################################################################################












####################################################################################################################################################
###########################################   ----  Classifier2  ----   ############################################################################
####################################################################################################################################################
'''
same as classifier1 but with polynomial features and using pipeline for the models
The Classifier2 class is a custom classifier that uses a staking classifier with the following base models:
    1: CustomRandomForestClassifier - more details in Custom_Random_Forest.py
    2: Forest1VsAllClassifier - more details in one_vs_all_forest.py
    3: KNeighborsClassifier - KNN as base model with weird parameters chosen by me witout deep analysis
    4: LogisticRegression - Logistic

The final estimator is a CustomRandomForestClassifier. The model is trained on the training data and used to predict the test data.
F1 score: 0.95

'''

from sklearn.pipeline import Pipeline
class Classifier2(BaseEstimator, ClassifierMixin):
    def __init__(self):
        # Scaler for all models
        self.scaler = StandardScaler()
        
        # Base models
        self.custom_rf = CustomRandomForestClassifier(n_trees=100, sample_size=0.9, max_depth=18, max_features=0.8)
        self.forest_1vsall = Forest1VsAllClassifier(classes=[0, 1, 2, 3, 4, 5], n_trees=100)
        
        # KNN with polynomial transformation (degree 2)
        self.knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Apply standard scaling
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Polynomial transformation of degree 2
            ('knn', KNeighborsClassifier(n_neighbors=50, weights='distance', p=1, n_jobs=2))  # KNN model
        ])
        
        # Logistic Regression with polynomial transformation (degree 3)
        self.log_reg_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Apply standard scaling
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),  # Polynomial transformation of degree 3
            ('logreg', LogisticRegression(max_iter=1000, solver='liblinear', 
                                          class_weight={0: 1, 1: 1, 2: 2.5, 3: 2.5, 5: 1}, C=10))  # Logistic Regression
        ])
        
        # Final model is still RandomForest
        self.final_estimat = CustomRandomForestClassifier(n_trees=100, sample_size=0.9, max_depth=18, max_features=0.9)
        
        # Stacking model with all base models
        self.model = StackingClassifier(
            estimators=[
                ('custom_rf', self.custom_rf),
                ('forest_1vsall', self.forest_1vsall),
                ('knn', self.knn_pipeline),  # KNN with polynomial features
                ('log_reg', self.log_reg_pipeline)  # Logistic Regression with polynomial features
            ],
            final_estimator=self.final_estimat,
            cv=5
        )

    def preprocess(self, X):
        return X  # Return X without modification for Random Forest models

    def fit(self, X, y):
        print("Training the model, this may take a while...")
        
        # Fit the stacking model
        self.model.fit(X, y)
        return self

    def predict(self, X):
        print("Predicting...")
        predictions = self.model.predict(X)
        
        print("\n" + "="*80)
        print("These results were achieved using custom models and models permitted according to the project's guidelines.")
        print("="*80 + "\n")

        return predictions

    def predict_proba(self, X):
        # Predict probabilities
        return self.model.predict_proba(X)

####################################################################################################################################################










####################################################################################################################################################
###########################################   ----  Classifier3  ----   ############################################################################
####################################################################################################################################################
'''
The Classifier3 class is a custom classifier that uses a stacking classifier with the following base models:
    1: RandomForestClassifier - Random Forest
    2: XGBClassifier - XGBoost
    3: MLPClassifier - Multi-layer Perceptron

The final estimator is a MLPClassifier. The model is trained on the training data and used to predict the test data.
F1 score: 0.97

'''
from xgboost import XGBClassifier
class Classifier3(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        # Base models
        self.rf = RandomForestClassifier(n_estimators=300, random_state=42)
        self.xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=0.9,
            random_state=42
        )
        self.mlp_base = MLPClassifier(max_iter=1000, random_state=42)  # Base MLP
        
        # Stacking model with final MLP classifier
        self.mlp_final = MLPClassifier(max_iter=1000, random_state=42)  # Final estimator
        self.model = StackingClassifier(
            estimators=[
                ('rf', self.rf),
                ('xgb', self.xgb),
                ('mlp_base', self.mlp_base)
            ],
            final_estimator=self.mlp_final,  # MLP as final decision model
            cv=5
        )

    def preprocess(self, X):
        return X  # Return X without modification

    def fit(self, X, y):
        print("Preprocessing training data... will apply standard scaler to normalize the data")
        print("Trainning the base models: Random Forest, XGBoost and MLP. also it will use MLP as a final estimator ")
        print()
        print("The MLP has 1000 iterations, and the Random Forest and XGBoost have 300 estimators. also we use 5 fold cross validation. so it will take a while plz wait ")
        X_train = self.scaler.fit_transform(X)
        self.model.fit(X_train, y)
        print("Training completed")
        return self

    def predict(self, X):
        print("\nPreprocessing test data... will apply standard scaler to normalize the data")
        X_train = self.scaler.transform(X)       
        print("Predicting...")
        predictions = self.model.predict(X_train)
        print("completed")
        print("\n" + "="*80)
        return predictions

    def predict_proba(self, X):
        X_train= self.scaler.transform(X)
        return self.model.predict_proba(X_train)