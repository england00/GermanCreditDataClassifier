import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, BaggingClassifier
import time
from colorama import Fore


############################################### MODELS SELECTION #######################################################

def models():
    models_list = [
        # Logistic Regression
        LogisticRegression(solver='saga', class_weight='balanced'),

        # K-Nearest Neighbors
        KNeighborsClassifier(weights='distance'),

        # Decision Tree
        DecisionTreeClassifier(class_weight='balanced'),

        # Support Vector Classifier
        SVC(class_weight='balanced')
    ]

    models_names = [
        'Logistic Regression',
        'K-Nearest Neighbors',
        'Decision Tree',
        'Support Vector Classifier'
    ]

    models_hyperparameters = [
        # Logistic Regression --> "C" is the hyperparameter for regularization
        {'penalty': ['l1', 'l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 0.01, 0.05, 0.07, 0.08, 0.09, 0.1, 0.5, 1]},

        # K-Nearest Neighbors
        {'n_neighbors': list(range(1, 20, 2))},

        # Decision Tree
        {'criterion': ['gini', 'entropy']},

        # Support Vector Classifier
        {'C': [400, 500, 600], 'gamma': [0.005, 0.004, 0.003, 0.002, 0.001, 0.0005],
         'kernel': ['linear', 'rbf']}
    ]

    return models_list, models_names, models_hyperparameters


##################################################### GRID SEARCH ######################################################

def grid_search(models_list, models_names, models_hyperparameters, X_training, y_training):
    chosen_hyperparameters = []
    estimators = []

    # searching the best hyperparameters for each model
    for model, model_name, hparameters in zip(models_list, models_names, models_hyperparameters):
        print('\n' + model_name.upper() + ':')
        clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='accuracy', cv=5, n_jobs=-1)  # 'f1_weighted'
        clf.fit(X_training, y_training)

        # collecting chosen hyperparameters and estimators for ensembles
        chosen_hyperparameters.append(clf.best_params_)
        estimators.append((model_name, clf))
        print('Accuracy:  ', Fore.GREEN, clf.best_score_, Fore.RESET)

        for hparam in hparameters:
            print(f'\t--> best value for hyperparameter "{hparam}": ',
                  Fore.YELLOW, clf.best_params_.get(hparam), Fore.RESET)

    return estimators


################################################# CROSS-VALIDATION #####################################################

def cross_validation_model_assessment(estimators, X_training, y_training):
    # SINGLE MODELS
    # LOGISTIC REGRESSION
    logr_model = LogisticRegression(solver='saga', class_weight='balanced', penalty='l1', C=0.09)
    scores = cross_validate(logr_model, X_training, y_training, cv=5, scoring=('f1_weighted', 'accuracy'))
    print('\nLOGISTIC REGRESSION:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)

    # K-NEAREST NEIGHBORS
    knn_model = KNeighborsClassifier(n_neighbors=13)
    scores = cross_validate(knn_model, X_training, y_training, cv=5, scoring=('f1_weighted', 'accuracy'))
    print('\nK-NEAREST NEIGHBORS:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)

    # DECISION TREE
    dt_model = DecisionTreeClassifier(class_weight='balanced', criterion='gini')
    scores = cross_validate(dt_model, X_training, y_training, cv=5, scoring=('f1_weighted', 'accuracy'))
    print('\nDECISION TREE:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)

    # SUPPORT VECTOR CLASSIFIER
    svc_model = SVC(class_weight='balanced', C=100.0, gamma=0.004, kernel='rbf')
    scores = cross_validate(svc_model, X_training, y_training, cv=5, scoring=('f1_weighted', 'accuracy'))
    print('\nSUPPORT VECTOR CLASSIFIER:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)

    # ENSEMBLES
    '''
    # STACKING CLASSIFIER with Logistic Regression, KNN and DT
    set_t0 = time.time()
    sf1_estimators = estimators.copy()
    sf1_estimators.pop(3)  # removes Support Vector Classifier position from estimator list
    clf_stack1 = StackingClassifier(estimators=sf1_estimators, final_estimator=LogisticRegression())
    scores = cross_validate(clf_stack1, X_training, y_training, cv=5, scoring=('f1_weighted', 'accuracy'))
    print('\nSTACKING CLASSIFIER with Logistic Regression, KNN and DT:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\nValidation took {time.time() - set_t0} sec')
    '''

    # STACKING CLASSIFIER with Logistic Regression, KNN and SVC
    set_t0 = time.time()
    sf2_estimators = estimators.copy()
    sf2_estimators.pop(2)  # removes Decision Tree position from estimator list
    clf_stack2 = StackingClassifier(estimators=sf2_estimators, final_estimator=LogisticRegression())
    scores = cross_validate(clf_stack2, X_training, y_training, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nSTACKING CLASSIFIER with Logistic Regression, KNN and SVC:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\nValidation took {time.time() - set_t0} sec')

    # BAGGING CLASSIFIER with Decision Tree
    '''
    set_t0 = time.time()
    clf_bagging2 = BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight='balanced',
                                                                           criterion='gini'), n_estimators=11)
    scores = cross_validate(clf_bagging2, X_training, y_training, cv=5, scoring=('f1_weighted', 'accuracy'))
    print('\nRANDOM FOREST CLASSIFIER:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\nValidation took {time.time() - set_t0} sec')
    '''

    # final model
    return clf_stack2
