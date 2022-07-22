from sklearn.feature_selection import SequentialFeatureSelector
import time
from colorama import Fore


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
