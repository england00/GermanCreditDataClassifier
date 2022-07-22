import time


####################################################### TRAINING #######################################################

def training(final_model, X_training, y_training):
    # setting initial time
    set_t0 = time.time()

    # training the final model
    final_model.fit(X_training, y_training)

    # ending timer
    print(f'\nFinal model training took {time.time() - set_t0} sec')

    return final_model
