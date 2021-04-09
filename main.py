from itertools import cycle

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from surprise import KNNWithZScore, KNNWithMeans, KNNBasic, KNNBaseline, BaselineOnly
from surprise.model_selection import cross_validate

from data.split_train_test import *
from data.surprise_preprocess import *
from algorithms.RandomForest import *
from algorithms.naive_bayes import *
from pyfiglet import Figlet
from scipy import interp

def memory_based(X, model='baseline', similarity='cosine', method='als'):
    X_train = surprise_transform(X)
    
    if model == 'baseline':
        if method == 'als':
            # ALS
            bsl_options = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }
            algo = BaselineOnly(bsl_options=bsl_options)
        elif method == 'sgd':
            # SGD
            bsl_options = {'method': 'sgd',
                        'learning_rate': .00005,
                        }
            algo = BaselineOnly(bsl_options=bsl_options)
    elif model == 'knn_basic': 
        sim_options = {'name': similarity, 
                    'user_based': False  
                    }
        algo = KNNBasic(sim_options=sim_options)

    elif model == 'knn_baseline':
        sim_options = {'name': similarity,
                    'user_based': False 
                    }
        algo = KNNBaseline(sim_options=sim_options)
    
    elif model == 'knn_with_means':
        sim_options = {'name': similarity,
               'user_based': False  
               }
        algo = KNNWithMeans(sim_options=sim_options)             
    
    elif model == 'knn_with_z_score':
        sim_options = {'name': similarity,
                    'user_based': False  
                    }
        algo = KNNWithZScore(sim_options=sim_options)
    else:
        print(f"Invalid model: {model}")

    print(f"Running model {model}" + f" using similarity {similarity}" * (model != 'baseline'))

    return cross_validate(algo, X_train, measures=['RMSE', 'MAE', 'MSE'], cv=3, verbose=True)
        

def random_forest(X_train, X_val, X_test, y_train, y_val, y_test, importances=False):
    rf = RandomForest(X_train, y_train, X_test, y_test)
    if importances:
        rf.get_importances()
    plot_true_Vs_predicted(y_val, rf.predict(X_val))
    print(f"Accuracy of predicted outcomes: {rf.accuracy}")
    print("Evaluating Random Forest algorithm")
    evaluate_model(X_train, y_train, rf)
    print("Plotting roc curves for RF model")
    plot_roc_curve(X_train, y_train, X_test, y_test, RandomForestClassifier())
    print(compute_confusion_matrix(y_val, rf.predict(X_val)))

def naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test):
    nb = NaiveBayes(X_train, y_train, X_test, y_test)
    plot_true_Vs_predicted(y_val, nb.predict(X_val))
    print(f"Accuracy of predicted outcomes: {nb.accuracy}")
    print("Evaluating Naive Bayes algorithm")
    evaluate_model(X_train, y_train, nb)
    print(compute_confusion_matrix(y_val, nb.predict(X_val)))

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
    plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
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

if __name__ == '__main__':
    f = Figlet(font='slant')
    print(f.renderText('DME MiniProject'))

    X_train, X_val, X_test, y_train, y_val, y_test = get_data()

    random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
    naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test)

    X_train, X_test = split_data()

    models = ['baseline', 'knn_basic', 'knn_baseline', 'knn_with_means', 'knn_with_z_score']
    baseline_methods = ['als', 'sgd']
    similarities = ['cosine', 'msd', 'pearson']
    
    results = memory_based(X_train, model='knn_with_z_score', similarity='pearson', method='sgd')