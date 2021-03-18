
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from heimat.statistik import stat_op

def check_mlp(x, y):
    global CLF_MLP
    print("+++++++++++++++++++++++++++++++++++++")
    labels_zuordnung_mlp = CLF_MLP.classes_
    beispiel_mlp_x = x
    beispiel_mlp_y = y
    y_true = np.array(beispiel_mlp_y)
    y_pred = np.array([labels_zuordnung_mlp[np.argmax(t)] for t in CLF_MLP.predict_proba(beispiel_mlp_x)])
    accuracy = (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred, labels=labels_zuordnung_mlp)
    if True:
        print("Labels:", labels_zuordnung_mlp)
        print("Confusion Matrix:")
        print(cm)
        for i in range(0, len(cm)):
            precision, recall, f1_score = stat_op.get_confusion_matrix_stats(cm, i)
            print("Label {} - precision {}, recall {}, f1_score {}: ".format(
                i, np.round(precision, 2), np.round(recall, 2), np.round(f1_score, 2)
            ))
        print("accuracy:", accuracy)
    print("+++++++++++++++++++++++++++++++++++++")


def get_clf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    labels_zuordnung = clf.classes_
    print("Beispiel:")
    print(clf.predict_proba(X_test))
    print("Labels:", labels_zuordnung)
    y_true = np.array(y_test)
    y_pred = np.array([labels_zuordnung[np.argmax(t)] for t in clf.predict_proba(X_test)])
    accuracy = (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred, labels=labels_zuordnung)
    print("Confusion Matrix:")
    print(cm)
    print("Präzision:", accuracy)
    return clf, labels_zuordnung
