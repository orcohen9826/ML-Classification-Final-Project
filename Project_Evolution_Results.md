# Comprehensive Model Comparison - Corrected

## KNN with Polynomial Features
```
precision    recall  f1-score   support

           0       0.52      0.63      0.57       326
           1       0.83      0.73      0.78      2593
           2       0.58      0.61      0.59      3157
           3       0.63      0.55      0.59      2242
           4       0.00      0.00      0.00        18
           5       0.74      0.81      0.77      3664

    accuracy                           0.69     12000
   macro avg       0.55      0.56      0.55     12000
weighted avg       0.69      0.69      0.68     12000

Calculated Weighted F1 Score: 0.6849
```

## KNN without Polynomial Features
```
              precision    recall  f1-score   support

           0       0.96      0.15      0.26       326
           1       0.77      0.80      0.78      2593
           2       0.59      0.55      0.57      3157
           3       0.63      0.54      0.58      2242
           4       0.00      0.00      0.00        18
           5       0.71      0.85      0.77      3664

    accuracy                           0.68     12000
   macro avg       0.61      0.48      0.50     12000
weighted avg       0.68      0.68      0.67     12000

Calculated Weighted F1 Score: 0.6750
```

## Logistic Regression without Polynomial Features
```
precision    recall  f1-score   support

           0       0.00      0.00      0.00       326
           1       0.70      0.46      0.56      2593
           2       0.32      0.54      0.40      3157
           3       0.35      0.33      0.34      2242
           4       0.00      0.00      0.00        18
           5       0.70      0.55      0.61      3664

    accuracy                           0.47     12000
   macro avg       0.35      0.31      0.32     12000
weighted avg       0.52      0.47      0.48     12000

Calculated Weighted F1 Score: 0.4942
```

## Logistic Regression with Polynomial Features (Degree 2)
```
              precision    recall  f1-score   support

           0       0.84      0.54      0.66       326
           1       0.80      0.67      0.73      2593
           2       0.47      0.57      0.52      3157
           3       0.49      0.58      0.53      2242
           4       0.40      0.11      0.17        18
           5       0.80      0.68      0.73      3664

    accuracy                           0.63     12000
   macro avg       0.63      0.53      0.56     12000
weighted avg       0.66      0.63      0.64     12000

Calculated Weighted F1 Score: 0.6449
```

## Logistic Regression with Polynomial Features (Degree 3)
```
              precision    recall  f1-score   support

           0       0.78      0.56      0.65       326
           1       0.97      0.84      0.90      2593
           2       0.62      0.72      0.67      3157
           3       0.63      0.78      0.69      2242
           4       0.12      0.11      0.11        18
           5       0.94      0.78      0.85      3664

    accuracy                           0.77     12000
   macro avg       0.68      0.63      0.65     12000
weighted avg       0.80      0.77      0.78     12000

f1 score:  0.7783
```

## Logistic Regression with Polynomial Features (Degree 4)
```
              precision    recall  f1-score   support

           0       0.71      0.67      0.69       326
           1       0.94      0.88      0.91      2593
           2       0.74      0.81      0.77      3157
           3       0.73      0.80      0.76      2242
           4       0.00      0.00      0.00        18
           5       0.91      0.84      0.87      3664

    accuracy                           0.82     12000
   macro avg       0.67      0.66      0.67     12000
weighted avg       0.83      0.82      0.83     12000

f1 score:  0.827
```

## Decision Tree without Over-sampling and Under-sampling
```
              precision    recall  f1-score   support

           0       0.87      0.82      0.85       215
           1       0.70      0.69      0.70      1715
           2       0.59      0.58      0.58      2119
           3       0.54      0.56      0.55      1506
           4       0.23      0.40      0.29        15
           5       0.72      0.72      0.72      2430

    accuracy                           0.65      8000
   macro avg       0.61      0.63      0.62      8000
weighted avg       0.65      0.65      0.65      8000

Calculated Weighted F1 Score: 0.6500
```

## Decision Tree with Over-sampling and Under-sampling
```
              precision    recall  f1-score   support

           0       0.92      0.90      0.91       215
           1       0.70      0.71      0.70      1715
           2       0.59      0.60      0.59      2119
           3       0.56      0.57      0.56      1506
           4       0.33      0.40      0.36        15
           5       0.75      0.71      0.73      2430

    accuracy                           0.66      8000
   macro avg       0.64      0.65      0.64      8000
weighted avg       0.66      0.66      0.66      8000

Calculated Weighted F1 Score: 0.6600
```

## Custom Random Forest (50 trees) without Over-sampling and Under-sampling
```
              precision    recall  f1-score   support

           0       0.81      0.99      0.89       272
           1       0.81      0.80      0.80      2143
           2       0.68      0.71      0.69      2628
           3       0.74      0.62      0.68      1866
           4       0.15      0.60      0.24        15
           5       0.80      0.83      0.81      3076

    accuracy                           0.76     10000
   macro avg       0.66      0.76      0.69     10000
weighted avg       0.76      0.76      0.75     10000

F1 Score: 0.7549
```

## Custom Random Forest (50 trees) with Over-sampling and Under-sampling
```
precision    recall  f1-score   support

           0       0.84      0.99      0.91       215
           1       0.79      0.81      0.80      1715
           2       0.68      0.72      0.70      2119
           3       0.71      0.64      0.68      1506
           4       0.12      0.40      0.18        15
           5       0.84      0.81      0.82      2430

    accuracy                           0.76      8000
   macro avg       0.66      0.73      0.68      8000
weighted avg       0.76      0.76      0.76      8000

F1 Score: 0.7579
```

## Classifier (Custom Stacking Model)
```
classifier f1: 0.95
classifier Accuracy: 0.95
              precision    recall  f1-score   support
           0       0.98      0.98      0.98       215
           1       0.95      0.96      0.96      1715
           2       0.95      0.93      0.94      2119
           3       0.92      0.92      0.92      1506
           4       0.44      0.47      0.45        15
           5       0.96      0.96      0.96      2430
    accuracy                           0.95      8000
   macro avg       0.87      0.87      0.87      8000
weighted avg       0.95      0.95      0.95      8000
```

## Classifier 1 (Built-in Stacking Ensemble)
```
classifier f1: 0.94
Final project grade = 98
classifier Accuracy: 0.94
              precision    recall  f1-score   support
           0       0.98      1.00      0.99       215
           1       0.96      0.95      0.96      1715
           2       0.94      0.94      0.94      2119
           3       0.92      0.93      0.92      1506
           4       0.62      0.67      0.65        15
           5       0.95      0.95      0.95      2430
    accuracy                           0.94      8000
   macro avg       0.90      0.91      0.90      8000
weighted avg       0.94      0.94      0.94      8000
```

## Classifier 2 (Stacking Ensemble with Polynomial Features)
```
classifier f1: 0.95
Final project grade = 99
classifier Accuracy: 0.95
              precision    recall  f1-score   support
           0       0.98      0.98      0.98       215
           1       0.95      0.96      0.95      1715
           2       0.95      0.93      0.94      2119
           3       0.92      0.92      0.92      1506
           4       0.53      0.67      0.59        15
           5       0.96      0.96      0.96      2430
    accuracy                           0.95      8000
   macro avg       0.88      0.90      0.89      8000
weighted avg       0.95      0.95      0.95      8000
```

## Classifier 3 (Stacking Ensemble with XGBoost and MLP)
```
classifier f1: 0.97
Final project grade = 101
classifier Accuracy: 0.97
              precision    recall  f1-score   support
           0       0.97      0.99      0.98       215
           1       0.97      0.98      0.98      1715
           2       0.96      0.97      0.96      2119
           3       0.97      0.94      0.95      1506
           4       0.76      0.87      0.81        15
           5       0.97      0.97      0.97      2430
    accuracy                           0.97      8000
   macro avg       0.93      0.95      0.94      8000
weighted avg       0.97      0.97      0.97      8000
```

