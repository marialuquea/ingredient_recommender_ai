from itertools import cycle
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from surprise import KNNWithZScore, KNNWithMeans, KNNBasic, KNNBaseline, BaselineOnly, SVD, NMF, SVDpp
from surprise.model_selection import cross_validate, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from data.split_train_test import *
from algorithms.RandomForest import *
from algorithms.naive_bayes import *
from algorithms.SVM import *
from pyfiglet import Figlet
from numpy import interp
import joblib

def memory_based(X, model='baseline', similarity='cosine', method='als', cv=2, measures=['RMSE'], verbose=True):
    X_train = surprise_transform(X)
    sim_options = {'name': similarity, 'user_based': False}

    if model == 'baseline':
        if method == 'als':
            bsl_options = {'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5}
            algo = BaselineOnly(bsl_options=bsl_options)
        elif method == 'sgd':
            bsl_options = {'method': 'sgd', 'learning_rate': .00005}
            algo = BaselineOnly(bsl_options=bsl_options)

    elif model == 'knn_basic':
        algo = KNNBasic(sim_options=sim_options)

    elif model == 'knn_baseline':
        algo = KNNBaseline(sim_options=sim_options)

    elif model == 'knn_with_means':
        algo = KNNWithMeans(sim_options=sim_options)

    elif model == 'knn_with_z_score':
        algo = KNNWithZScore(sim_options=sim_options)
    else:
        raise Exception(f"Invalid model passed to function {model}")

    print(f"Cross validating algorithm with {cv} folds")

    return cross_validate(algo, X_train, measures=measures, cv=cv, verbose=verbose)

def model_based(X, model='svd', cv=2, measures=['RMSE'], verbose=True):
    X_train = surprise_transform(X)

    if model == 'svd':
        algo = SVD()
    elif model == 'svdpp':
        algo = SVDpp()
    # throws zero division error if all quantities are zero related to a item_id or store_id
    elif model == 'nmf':
        algo = NMF()
    else:
        raise Exception(f"Invalid model passed to function {model}")

    print(f"Cross validating algorithm with {cv} folds")
    return cross_validate(algo, X_train, measures=measures, cv=cv, verbose=verbose)

def random_forest(X_train, X_val, X_test, y_train, y_val, y_test, importances=False):
    rf = RandomForest(X_train, y_train, X_test, y_test)
    if importances:
        rf.get_importances()
    plot_true_Vs_predicted(y_val, rf.predict(X_val))
    print("Evaluating Random Forest algorithm")
    evaluate_model(X_train, y_train, rf.clf)
    print("Plotting roc curves for RF model")
    plot_roc_curve(X_train, y_train, X_test, y_test, RandomForestClassifier())
    get_metrics(rf.clf, X_val, y_val)
    # TODO: grid search
    return rf.clf

def naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test):
    nb = NaiveBayes(X_train, y_train, X_test, y_test)
    plot_true_Vs_predicted(y_val, nb.predict(X_val))
    print("Evaluating Naive Bayes algorithm")
    evaluate_model(X_train, y_train, nb.clf)
    get_metrics(nb.clf, X_val, y_val)
    # TODO: grid search
    return nb.clf

def svm(X_train, X_val, X_test, y_train, y_val, y_test):
    svm = SVM(X_train, y_train, X_test, y_test)
    plot_true_Vs_predicted(y_val, svm.predict(X_val))
    print("Evaluating SVM algorithm")
    evaluate_model(X_train, y_train, svm.clf)
    print("Plotting roc curves for SVM model")
    plot_roc_curve(X_train, y_train, X_test, y_test, SVC())
    get_metrics(svm.clf, X_val, y_val)
    #TODO: grid search
    return svm.clf

def get_metrics(model, X_val, y_val):
    print(f"Accuracy of testing set: {accuracy_score(y_val.values, model.predict(X_val))}")
    print(compute_confusion_matrix(y_val, model.predict(X_val)))

def compute_confusion_matrix(y_true, predictions):
    return confusion_matrix(y_true, predictions, labels=np.unique(y_true))

def plot_true_Vs_predicted(y_true, predictions):
    plt.scatter(y_true, predictions, c='#FF7AA6')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Precision of predicted outcomes')
    plt.plot(np.unique(y_true), np.poly1d(np.polyfit(y_true, predictions, 1))(np.unique(y_true)))
    plt.show()

def plot_roc_curve(X_train, y_train, X_test, y_test, model):
    y = label_binarize(y_train, classes=np.unique(y_train))
    y_test = label_binarize(y_test, classes=np.unique(y_train))
    n_classes = y.shape[1]
    model = OneVsRestClassifier(model).fit(X_train, y)
    y_score = model.predict(X_test)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic in multi-class RF model')
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.show()

def evaluate_model(xTrain, yTrain, model):
    kfold = KFold(n_splits=3, shuffle=True)
    count, avg_roc_auc, avg_accuracy, avg_precision, avg_recall, avg_f1score = 0, 0, 0, 0, 0, 0

    for train, test in kfold.split(xTrain):
        print(f"Test: {count + 1}")
        X_train, X_test = xTrain.iloc[train], xTrain.iloc[test]
        y_train, y_true = yTrain.iloc[train], yTrain.iloc[test]

        y_pred = model.predict(X_test)
        avg_accuracy += accuracy_score(y_true, y_pred)
        avg_precision += precision_score(y_true, y_pred, average='micro')
        avg_recall += recall_score(y_true, y_pred, average='micro')
        avg_f1score += f1_score(y_true, y_pred, average='micro')
        count += 1
    avg_accuracy /= count
    avg_precision /= count
    avg_recall /= count
    avg_f1score /= count
    print(f"\n{model} evaluation results")
    print("Average accuracy:", avg_accuracy)
    print("Average precision:", avg_precision)
    print("Average recall:", avg_recall)
    print("Average f1 score:", avg_f1score)
    return avg_accuracy, avg_precision, avg_recall, avg_f1score

def ask():
    print("Select an option:")
    print("[1] - Train ML model with data to predict what type of cuisine a recipe belongs to.")
    print("[2] - Use model based collaborative filtering to provide ingredient recommendation by developing a model of recipe ingredients")
    print("[3] - Use memory based collaborative filtering to provide ingredient recommendation by the same process as 2.")
    print("[4] - Exit program")

def choose_ML_model():
    print("Choose a model to work with.\n[1] - Random Forest\n[2] - Naive Bayes\n[3] - Support Vector Machine")
    model_option = input("Choose a number: ")
    option = input("Would you like to \n[1] - train a model\n[2] - load a pre-trained model\n(Make sure to create a folder called 'models' in this directory.)\nChoose a number: ")
    if option == "1":
        X_train, X_val, X_test, y_train, y_val, y_test = get_data()
        if model_option == "1":
            model = random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
            filename = "models/randomForest.sav"
        elif model_option == "2":
            model = naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test)
            filename = "models/naiveBayes.sav"
        elif model_option == "3":
            model = svm(X_train, X_val, X_test, y_train, y_val, y_test)
            filename = "models/svm.sav"
        else:
            print("Invalid input, try again.")
            return
        joblib.dump(model, filename)
        print(f"Model {model} saved correctly as {filename}!")
    elif option == "2":
        _, X_val, _, _, y_val, _ = get_data()
        if model_option == "1":
            model = joblib.load('models/randomForest.sav')
        elif model_option == "2":
            model = joblib.load('models/naiveBayes.sav')
        elif model_option == "3":
            model = joblib.load('models/svm.sav')
        else:
            print("Invalid input, try again.")
            return
        get_metrics(model, X_val, y_val)
        #TODO: Allow user to input ingredient names and predict cuisine (?) Totally not needed but it would be cool lol

if __name__ == '__main__':
    f = Figlet(font='slant')
    print(f.renderText('DME MiniProject'))

    finished = False
    while finished == False:
        ask()
        option = input("Choose a number: ")
        if option == "1":
            choose_ML_model()
        if option == "2":
            print("Model based")
        if option == "3":
            print("Memory based")
        if option == "4":
            finished = True



    # ------ Collborative filtering ------
    # X_train, X_test = get_data(y_val=False)
    # print(X_train.shape, X_test.shape)

    # ------ Memory based ------
    # models = ['baseline', 'knn_basic', 'knn_baseline', 'knn_with_means', 'knn_with_z_score']
    # baseline_methods = ['als', 'sgd']
    # similarities = ['cosine', 'msd', 'pearson']
    #
    # algorithm = memory_based(X_train, model='knn_with_z_score', similarity='pearson', method='sgd')
    # algorithm = model_based(X_train, model='svdpp')
    # print(f"algorithm: {algorithm}")
