from colorama import Fore
from module_exploratory_data_analysis import exploratory_data_analysis, features_preprocessing, \
    numerical_features_scaling


def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(60, '#'), Fore.RESET)


####################################################### TESTING ########################################################

def testing(final_model, testing_set, PLOT=True, sfs=None):
    # showing testing set info and splitting in design matrix and target vector
    title('EXPLORATORY DATA ANALYSIS')
    X_testing, y_testing = exploratory_data_analysis(testing_set, 0)

    # managing categorical features
    title('FEATURES PREPROCESSING')
    X_testing_categorical, categorical_features, numerical_features = features_preprocessing(X_testing, PLOT, True)

    # scaling numerical features and obtaining final testing set
    title("NUMERICAL FEATURES SCALING")
    X_testing = numerical_features_scaling(X_testing, X_testing_categorical, numerical_features, PLOT)

    # X_testing = sfs.transform(X_testing) # uncomment if using FEATURE SELECTION
    y_prediction = final_model.predict(X_testing)

    return X_testing, y_testing, y_prediction
