import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, BaggingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
import time
from colorama import Fore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scikitplot.metrics import plot_roc_curve, plot_confusion_matrix

################################################### OPTIONS ############################################################

PLOT = False

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

pd.set_option('display.max_columns', None)
plt.rcParams.update({'figure.max_open_warning': 0})


def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(60, '#'), Fore.RESET)


############################################### LOADING DATASET ########################################################

def loading_dataset(path):
    # acquiring data
    names = ['Status of existing checking account', 'Duration in month', 'Credit history', 'Purpose',
             'Credit amount', 'Savings account/bonds', 'Present employment since',
             'Installment rate in percentage of disposable income', 'Personal status and sex',
             'Other debtors / guarantors', 'Present residence since', 'Property', 'Age in years',
             'Other installment plans', 'Housing', 'Number of existing credits at this bank', 'Job',
             'Number of people being liable to provide maintenance for', 'Telephone', 'Foreign worker', 'y']
    df = pd.read_csv(path, delimiter=' ', header=None, names=names)

    # splitting the dataset in training and testing
    df_tr, df_ts = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df['y'])

    return df_tr, df_ts, names


########################################### EXPLORATORY DATA ANALYSIS ##################################################

def exploratory_data_analysis(df, preprocessing, X=0, y=0):
    if not preprocessing:

        # showing some details about the dataset
        print('\nINFORMATIONS ABOUT THE DATASET:')
        df.info()
        print('\n\nDATA PREVIEW:\n', df.head(3))

        # selecting features and target variable
        y = df['y']
        X = df.drop(columns='y')

        return X, y

    else:
        if PLOT:
            # plotting the correlation matrix
            correlation_df = pd.concat([X, y], axis=1, sort=False)
            plt.figure(figsize=(12, 10), dpi=80)
            sns.heatmap(correlation_df.corr(),
                        xticklabels=correlation_df.corr().columns,
                        yticklabels=correlation_df.corr().columns,
                        center=0, annot=True, fmt='.2f', square=True, linewidths=.5)
            plt.title('Correlation Matrix', fontsize=22)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.show()

            # showing potentially correlated features
            for features in [['Credit amount', 'Duration in month'],
                             ['Telephone', 'Job'],
                             ['Credit history', 'Number of existing credits at this bank']]:
                correlation_features = [features[0], features[1], 'y']
                sns.pairplot(correlation_df[correlation_features], hue='y', corner=True)
                plt.show()

            # showing relation between target value "y" (1 == GOOD, 2 == BAD) and "Status of existing checking account"
            sns.catplot(x='Status of existing checking account', y='y', data=correlation_df, orient='h', height=5, aspect=1,
                        palette='tab10',
                        kind='violin', dodge=True, cut=0, bw=.2)
            plt.show()


############################################ FEATURES PREPROCESSING ####################################################

def features_preprocessing(X, test=False):
    # checking possible values problems
    print('\n\nNAN VALUES:\n', X.isnull().any())
    print('\n\nMISSING VALUES:\n', (X == ' ').any())
    print('\n\nDUPLICATED VALUES:\n', X.duplicated(keep='first'))

    # transforming categorical features in discrete and showing their values frequency
    categorical_features = [feature for feature in X.columns if X[feature].dtype == 'object']
    X_categorical = X[categorical_features].copy()
    for feature in categorical_features:
        X_categorical[feature], _ = pd.factorize(X_categorical[feature], sort=True)

        # plotting values frequency of categorical features
        if PLOT and test == False:
            sns.catplot(x=feature, kind='count', data=X_categorical).set(title='Values frequency')
            plt.show()

    if test == False:
        print('\n\nOBTAINED DISCRETE DATA:\n', X_categorical.head(3))

    numerical_features = [feature for feature in X.columns if X[feature].dtype != 'object']

    # showing values distribution of numerical features
    if PLOT and test == False:
        for feature in numerical_features:
            # plotting values distribution of discrete features
            sns.distplot(X[feature], color='green').set(title='Values distribution in {}'.format(feature))
            plt.show()

    print('\n\nNUMERICAL DATA:\n', X[numerical_features].head(3))

    return X_categorical, categorical_features, numerical_features


########################################## NUMERICAL FEATURES SCALING ##################################################

def numerical_features_scaling(X, X_categorical, numerical_features, test=False):
    # numerical features scaling
    scaler = StandardScaler().fit(X[numerical_features].astype('float64'))
    X_Z_score = pd.DataFrame(scaler.transform(X[numerical_features].astype('float64')),
                                          columns=numerical_features,
                                          index=X[numerical_features].index)


    # showing new distribution compared to previous
    if PLOT and test == False:
        for feature in numerical_features:
            sns.pairplot(pd.DataFrame({'{} Z-score'.format(feature): X_Z_score[feature],
                                       '{}'.format(feature): X[feature]}), corner=True)
            plt.show()

    # obtaining final features
    X = pd.concat([X_categorical, X_Z_score], axis=1, sort=False)

    return X


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
        {'C': [1e-4, 1e-2, 1, 1e1, 50, 1e2], 'gamma': [0.005, 0.004, 0.003, 0.002, 0.001, 0.0005],
         'kernel': ['linear', 'rbf']}
    ]

    return models_list, models_names, models_hyperparameters


##################################################### GRID SEARCH ######################################################

def grid_search(models_list, models_names, models_hyperparameters,
                                     X_training, y_training):
    chosen_hyperparameters = []
    estimators = []

    # searching the best hyperparameters for each model
    for model, model_name, hparameters in zip(models_list, models_names, models_hyperparameters):
        print('\n' + model_name.upper() + ':')
        clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='accuracy', cv=5)  # 'f1_weighted'
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
    scores = cross_validate(clf_stack2, X_training, y_training, cv=5, scoring=('f1_weighted', 'accuracy'))
    print('\nSTACKING CLASSIFIER with Logistic Regression, KNN and SVC:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\nValidation took {time.time() - set_t0} sec')
    '''

    '''
    # BAGGING CLASSIFIER with Decision Tree
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
    return clf_stack1
    # return clf_stack2
    # return clf_bagging2


################################################ FEATURES SELECTION ####################################################

def wrapper(final_model, X_training, y_training, feature_names):
    # setting initial time
    set_t0 = time.time()

    sfs = SequentialFeatureSelector(final_model, cv=2)
    sfs.fit(X_training, y_training)
    print('Selected features in ', Fore.GREEN, 'green', Fore.RESET, ':\n')
    for i in range(len(feature_names) - 1):
        if sfs.get_support()[i]:
            print(Fore.GREEN, feature_names[i].upper(), Fore.RESET)
        else:
            print(feature_names[i])

    print('\nFeature selected: ', Fore.GREEN, sfs.get_support()[sfs.get_support()].size, '/20', Fore.RESET)
    X_training = sfs.transform(X_training)

    # ending timer
    print(f'\nFeature selection took {time.time() - set_t0} sec')

    return X_training, sfs


####################################################### TRAINING #######################################################

def training(final_model, X_training, y_training):
    # setting initial time
    set_t0 = time.time()

    # training the final model
    final_model.fit(X_training, y_training)

    # ending timer
    print(f'\nFinal model training took {time.time() - set_t0} sec')

    return final_model


####################################################### TESTING ########################################################

def testing(final_model, testing_set, sfs=None):
    # showing testing set info and splitting in design matrix and target vector
    title('EXPLORATORY DATA ANALYSIS')
    X_testing, y_testing = exploratory_data_analysis(testing_set, 0)

    # managing categorical features
    title('FEATURES PREPROCESSING')
    X_testing_categorical, categorical_features, numerical_features = features_preprocessing(X_testing, True)

    # scaling numerical features and obtaining final testing set
    title("NUMERICAL FEATURES SCALING")
    X_testing = numerical_features_scaling(X_testing, X_testing_categorical, numerical_features, True)

    # X_testing = sfs.transform(X_testing) # uncomment if using FEATURE SELECTION
    y_prediction = final_model.predict(X_testing)

    return X_testing, y_testing, y_prediction


############################################### FINAL TESTING RESULTS ##################################################

def results(final_model, X_testing, y_testing, y_prediction):
    # metrics
    print('Accuracy: ', accuracy_score(y_testing, y_prediction))
    print('Precision: ', precision_score(y_testing, y_prediction, average='weighted'))
    print('Recall: ', recall_score(y_testing, y_prediction, average='weighted'))
    print('F1-Score: ', f1_score(y_testing, y_prediction, average='weighted'))

    # performance plots
    # ROC curve
    # plot_roc_curve(final_model, X_testing, y_testing)
    # plt.show()
    # confusion matrix
    # plot_confusion_matrix(final_model, X_testing, y_testing)
    # plt.show()


if __name__ == "__main__":
    # DATA ACQUISITION
    title('DATA ACQUISITION')
    DATASET_PATH = '../data/german.data'
    training_set, testing_set, features_names = loading_dataset(DATASET_PATH)

    # EXPLORATORY DATA ANALYSIS with RAW DATA
    title('EXPLORATORY DATA ANALYSIS with RAW DATA')
    X_training, y_training = exploratory_data_analysis(training_set, False)

    # FEATURES PREPROCESSING
    title('FEATURES PREPROCESSING')
    X_training_categorical, categorical_features, numerical_features = features_preprocessing(X_training)

    # EXPLORATORY DATA ANALYSIS with CLEAN DATA
    title('EXPLORATORY DATA ANALYSIS with CLEAN DATA')
    exploratory_data_analysis(training_set,
                              True,
                              pd.concat([X_training_categorical, X_training[numerical_features]], axis=1, sort=False),
                              y_training)

    # NUMERICAL FEATURES SCALING
    title('NUMERICAL FEATURES SCALING')
    X_training = numerical_features_scaling(X_training, X_training_categorical, numerical_features)

    # MODELS SELECTION
    title('MODELS SELECTION')
    models_list, models_names, models_hyperparameters = models()

    # GRID SEARCH
    title('GRID SEARCH')
    estimators = grid_search(models_list, models_names, models_hyperparameters,
                                                  X_training, y_training)

    # CROSS-VALIDATION
    title('CROSS-VALIDATION')
    final_model = cross_validation_model_assessment(estimators, X_training, y_training)

    # FEATURES SELECTION
    # title('FEATURE SELECTION')
    # X_training, sequential_feature_selector = wrapper(final_model, X_training, y_training, features_names)

    # TRAINING
    title('TRAINING')
    final_model = training(final_model, X_training, y_training)

    # TESTING AND PREDICTION
    title('TESTING AND PREDICTION')
    # uncomment if using FEATURES SELECTION
    # X_testing, y_testing, y_prediction = testing(final_model, testing_set, sequential_feature_selector)
    # comment if not using FEATURES SELECTION
    X_testing, y_testing, y_prediction = testing(final_model, testing_set)

    # FINAL TESTING RESULTS
    title('FINAL TESTING RESULTS')
    results(final_model, X_testing, y_testing, y_prediction)
