import pandas as pd
import matplotlib.pyplot as plt
import warnings
from colorama import Fore
from module_loading_dataset import loading_dataset
from module_exploratory_data_analysis import exploratory_data_analysis, features_preprocessing, \
    numerical_features_scaling
from module_model_selection import models, grid_search, cross_validation_model_assessment
from module_features_selection import wrapper
from module_training import training
from module_testing import testing
from module_final_testing_results import results


################################################### OPTIONS ############################################################

PLOT = False

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

pd.set_option('display.max_columns', None)
plt.rcParams.update({'figure.max_open_warning': 0})


def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(60, '#'), Fore.RESET)


#################################################### MAIN ##############################################################

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
    X_training_categorical, categorical_features, numerical_features = features_preprocessing(X_training, PLOT)

    # EXPLORATORY DATA ANALYSIS with CLEAN DATA
    title('EXPLORATORY DATA ANALYSIS with CLEAN DATA')
    exploratory_data_analysis(training_set, True,
                              pd.concat([X_training_categorical, X_training[numerical_features]], axis=1, sort=False),
                              y_training, PLOT)

    # NUMERICAL FEATURES SCALING
    title('NUMERICAL FEATURES SCALING')
    X_training = numerical_features_scaling(X_training, X_training_categorical, numerical_features, PLOT)

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
    # X_testing, y_testing, y_prediction = testing(final_model, testing_set, PLOT, sequential_feature_selector)
    # comment if not using FEATURES SELECTION
    X_testing, y_testing, y_prediction = testing(final_model, testing_set, PLOT)

    # FINAL TESTING RESULTS
    title('FINAL TESTING RESULTS')
    results(final_model, X_testing, y_testing, y_prediction)
