import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scikitplot.metrics import plot_roc_curve, plot_confusion_matrix


############################################### FINAL TESTING RESULTS ##################################################

def results(final_model, X_testing, y_testing, y_prediction):
    # metrics
    print('Accuracy: ', accuracy_score(y_testing, y_prediction))
    print('Precision: ', precision_score(y_testing, y_prediction, average='weighted'))
    print('Recall: ', recall_score(y_testing, y_prediction, average='weighted'))
    print('F1-Score: ', f1_score(y_testing, y_prediction, average='weighted'))

    # performance plots
    # ROC curve
    plot_roc_curve(final_model, X_testing, y_testing)
    plt.show()
    # confusion matrix
    plot_confusion_matrix(final_model, X_testing, y_testing)
    plt.show()
