import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


########################################### EXPLORATORY DATA ANALYSIS ##################################################

def exploratory_data_analysis(df, preprocessing, X=0, y=0, PLOT=True):
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
            sns.catplot(x='Status of existing checking account', y='y', data=correlation_df, orient='h', height=5,
                        aspect=1,
                        palette='tab10',
                        kind='violin', dodge=True, cut=0, bw=.2)
            plt.show()


############################################ FEATURES PREPROCESSING ####################################################

def features_preprocessing(X, PLOT=True, test=False):
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
        if PLOT and not test:
            sns.catplot(x=feature, kind='count', data=X_categorical).set(title='Values frequency')
            plt.show()

    if not test:
        print('\n\nOBTAINED DISCRETE DATA:\n', X_categorical.head(3))

    numerical_features = [feature for feature in X.columns if X[feature].dtype != 'object']

    # showing values distribution of numerical features
    if PLOT and not test:
        for feature in numerical_features:
            # plotting values distribution of discrete features
            sns.distplot(X[feature], color='green').set(title='Values distribution in {}'.format(feature))
            plt.show()

    print('\n\nNUMERICAL DATA:\n', X[numerical_features].head(3))

    return X_categorical, categorical_features, numerical_features


########################################## NUMERICAL FEATURES SCALING ##################################################

def numerical_features_scaling(X, X_categorical, numerical_features, PLOT=True, test=False):
    # numerical features scaling
    scaler = StandardScaler().fit(X[numerical_features].astype('float64'))
    X_Z_score = pd.DataFrame(scaler.transform(X[numerical_features].astype('float64')),
                             columns=numerical_features,
                             index=X[numerical_features].index)

    # showing new distribution compared to previous
    if PLOT and not test:
        for feature in numerical_features:
            sns.pairplot(pd.DataFrame({'{} Z-score'.format(feature): X_Z_score[feature],
                                       '{}'.format(feature): X[feature]}), corner=True)
            plt.show()

    # obtaining final features
    X = pd.concat([X_categorical, X_Z_score], axis=1, sort=False)

    return X
