from itertools import cycle
import argparse
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from surprise import KNNWithZScore, KNNWithMeans, KNNBasic, KNNBaseline, BaselineOnly, SVD, NMF, SVDpp
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
from data.split_train_test import *
from data.surprise_preprocess import *
from algorithms.RandomForest import *
from algorithms.naive_bayes import *
from algorithms.SVM import *
from pyfiglet import Figlet
from numpy import interp
import pickle
from tqdm import tqdm

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--render", type=str, default='DME MiniProject',
                        help="text to render")

    parser.add_argument("--verbose", type=int, default=0,
                        help="verbosity of the process")

    parser.add_argument("--task", type=str, default='cuisine',
                        choices=['cuisine', 'recommendation'],
                        help="task to perform")

    parser.add_argument("--cuisine_model", type=str, default='random_forest',
                        choices=['random_forest', 'naive_bayes', 'svm'],
                        help="model to use on the cuisine task")

    parser.add_argument("--load_or_train", type=str, default='train',
                        choices=['load', 'train'],
                        help="load a pre-trained model or train one")

    parser.add_argument("--recommendation_model", type=str, default='baseline',
                        choices=['baseline', 'knn_basic', 'knn_baseline', 'knn_with_means', 'knn_with_z_score', 'svd', 'svdpp', 'nmf'],
                        help="memory model to use on the recommendation task")

    parser.add_argument("--baseline_method", type=str, default='als',
                        choices=['als', 'sgd'],
                        help="method used by the baseline model (recommendation task)")

    parser.add_argument("--similarity", type=str, default='cosine',
                        choices=['cosine', 'msd', 'pearson'],
                        help="similarity metric used by the KNN models (recommendation task)")

    return parser

def memory_based(model='baseline', similarity='cosine', method='als'):
    sim_options = {'name': similarity, 'user_based': False}
    if model == 'baseline':
        if method == 'als':
            bsl_options = {'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5}
            algo = BaselineOnly(bsl_options=bsl_options)
        elif method == 'sgd':
            bsl_options = {'method': 'sgd', 'learning_rate': .00005}
            algo = BaselineOnly(bsl_options=bsl_options)
        else:
            raise Exception(f"Invalid method passed to function {method}")
    elif model == 'knn_basic':
        algo = KNNBasic(sim_options=sim_options)
    elif model == 'knn_baseline':
        algo = KNNBaseline(sim_options=sim_options)
    elif model == 'knn_with_means':
        algo = KNNWithMeans(sim_options=sim_options)
    elif model == 'knn_with_z_score':
        algo = KNNWithZScore(sim_options=sim_options, k = 20)
    else:
        raise Exception(f"Invalid model passed to function {model}")
    return algo

def cross_validate_model(algo, X, cv=3, measures=['RMSE'], verbose=True):
    X_train = surprise_transform(X)
    print(f"Cross validating algorithm with {cv} folds")
    return cross_validate(algo, X_train, measures=measures, cv=cv, verbose=verbose)

def fit_model(X, model):
    X_train = surprise_transform(X)
    return model.fit(X_train)

def model_based(model='svd'):
    if model == 'svd':
        algo = SVD(n_factors = 200)
    else:
        raise Exception(f"Invalid model passed to function {model}")
    return algo

def random_forest(X_train, X_val, X_test, y_train, y_val, y_test, importances=False, verbose=0):
    if verbose == True: verbose = 3
    rf = RandomForest(X_train, y_train, X_test, y_test, verbose=verbose)
    if importances:
        rf.get_importances()
    plot_true_Vs_predicted(y_val, rf.predict(X_val))
    plot_roc_curve(X_train, y_train, X_test, y_test, RandomForestClassifier())
    plot_confusion_matrix(rf.clf, X_test, y_test, display_labels=get_cuisines()['cuisine_label'], xticks_rotation=65)
    plt.title("Results for Random Forest Classifier")
    plt.tight_layout()
    plt.show()
    X_final = pd.concat([X_test, X_val])
    y_final = pd.concat([y_test, y_val])
    print(f"Accuracy of validation+test sets together: {accuracy_score(y_final.values, rf.predict(X_final))}")
    return rf.clf

def naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test):
    nb = NaiveBayes(X_train, y_train, X_test, y_test)
    plot_true_Vs_predicted(y_val, nb.predict(X_val))
    X_final = pd.concat([X_test, X_val])
    y_final = pd.concat([y_test, y_val])
    print(f"Accuracy of validation+test sets together: {accuracy_score(y_final.values, nb.predict(X_final))}")
    return nb.clf

def svm(X_train, X_val, X_test, y_train, y_val, y_test):
    svm = SVM(X_train, y_train, X_test, y_test, verbose=0)
    plot_true_Vs_predicted(y_val, svm.predict(X_val))
    plot_roc_curve(X_train, y_train, X_test, y_test, SVC())
    X_final = pd.concat([X_test, X_val])
    y_final = pd.concat([y_test, y_val])
    print(f"Accuracy of validation+test sets together: {accuracy_score(y_final.values, svm.predict(X_final))}")
    return svm.clf

def choose_ML_model(option):
    X_train, X_val, X_test, y_train, y_val, y_test = get_data()
    if option == 'random_forest':
        model = random_forest(X_train, X_val, X_test, y_train, y_val, y_test, verbose=opts.verbose)
    elif option == 'naive_bayes':
        model = naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test)
    elif option == 'svm':
        model = svm(X_train, X_val, X_test, y_train, y_val, y_test, verbose=opts.verbose)
    else:
        print("Invalid input, try again.")
        return
    filename = f"models/{opts.cuisine_model}.sav"
    joblib.dump(model, filename)
    print(f"Model {model} saved correctly as {filename}!")
    return model

def load_trained_model(cuisine_model):
    _, X_val, _, _, y_val, _ = get_data()
    model = joblib.load(f'models/{cuisine_model}.sav')
    return model

def compute_confusion_matrix(model, X_test, y_test, labels=[], title=""):
    plot_confusion_matrix(model, X_test, y_test, display_labels=labels, xticks_rotation=65)
    plt.title(title)
    plt.tight_layout()
    plt.show()

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

def evaluate_model(xTrain, yTrain, model, cv=5):
    print(f"Cross validating model {model}")
    scores = cross_val_score(model, xTrain, yTrain, cv=cv)
    print(f'Cross validation score: {sum(scores)/cv}, Scores: {scores}')

def cross_validate_all_models():
    X_train, X_val, X_test, y_train, y_val, y_test = get_data()
    rf = random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(X_train, y_train, rf)
    svm_model = svm(X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(X_train, y_train, svm_model)
    nb = naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test)
    evaluate_model(X_train, y_train, nb)

if __name__ == '__main__':
    opts = get_argparser().parse_args()

    # cross_validate_all_models()

    f = Figlet(font='slant')
    print(f.renderText(opts.render))
    if opts.task == 'cuisine':
        if opts.load_or_train == 'load':
            model = load_trained_model(opts.cuisine_model)
            X_train, X_val, X_test, y_train, y_val, y_test = get_data()
            evaluate_model(X_train, y_train, model)
            compute_confusion_matrix(model, X_test, y_test, get_cuisines()['cuisine_label'], f"Confusion Matrix of {model} for test set")
        else:
            model = choose_ML_model(opts.cuisine_model)
    elif opts.task == 'recommendation':
        print(f"Task: {opts.task}; Model: {opts.recommendation_model}" \
              + f"; Method: {opts.baseline_method}" * (opts.recommendation_model == 'baseline') \
              + f"; Similarity: {opts.similarity}" * (opts.recommendation_model != 'baseline'))
        X_train, X_test = get_data(y_val=False)
        if opts.recommendation_model in ['baseline', 'knn_basic', 'knn_baseline', 'knn_with_means', 'knn_with_z_score']:
            algo = memory_based(model=opts.recommendation_model, similarity=opts.similarity, method=opts.baseline_method)
            print("Cross validating model...")
            results = cross_validate_model(algo, X_train, measures = ['RMSE', 'MSE', 'MAE'])
            fitted_model = fit_model(X_train, algo)
            full_train_data, hidden_rankings, full_test = create_recommendation_set(1)
            algo.fit(full_train_data)
            predictions = []
            print("Running test predictions...")
            for _, row in tqdm(hidden_rankings.iterrows()):
                current_uid = row['recipe_id']
                current_iid = row['ingredient']
                current_r = row['rating']
                current_prediction = algo.predict(current_uid, current_iid, current_r)
                predictions.append(current_prediction)
            with open('rec_predictions_hidden.pkl', 'wb') as output:
                pickle.dump(predictions, output, pickle.HIGHEST_PROTOCOL)
            with open('rec_rankings_hidden.pkl', 'wb') as output:
                pickle.dump(hidden_rankings, output, pickle.HIGHEST_PROTOCOL)
            with open('full_test_hidden.pkl', 'wb') as output:
                pickle.dump(full_test, output, pickle.HIGHEST_PROTOCOL)
            #TODO: print some message saying that predictions are done lol

        if opts.recommendation_model in ['svd', 'svdpp','nmf']:
            algo = model_based(model=opts.recommendation_model)
            results = cross_validate_model(algo, X_train, measures = ['RMSE', 'MSE', 'MAE'])