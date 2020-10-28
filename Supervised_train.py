import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
import pickle
from time import time
from sklearn.metrics import roc_auc_score, roc_curve, auc

t0 = time()
# loading training data: data_X, data_y
# Tree algorithms for outputing the feature important index.
n = 100
feature_select = ExtraTreesClassifier(n_estimators=n, random_state=111)
feature_select.fit(data_X, data_y)

# normalization
data_X = preprocessing.minmax_scale(data_X, feature_range=(-1, 1))
data_X = preprocessing.scale(data_X)

# split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=3)

# =======================
# k-neighbor
# =======================
knn = KNeighborsClassifier(n_neighbors=3, leaf_size=30, algorithm='auto')
knn.fit(X_train, y_train)
print('knn:', knn.score(X_test, y_test))  # default algorithm,
print('knn:', accuracy_score(knn.predict(X_test), y_test))

fpr, tpr, threshold = metrics.roc_curve(y_test, knn.predict(X_test), pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

knn_scores = cross_val_score(knn, X_train, y_train, cv=5)
print('KNN_cross:', knn_scores.mean())

tn, fp, fn, tp = confusion_matrix(y_test, knn.predict(X_test)).ravel()
true_positive = tp
ppv = tp/(tp + fp)
npv = tn/(fn + tn)
accuracy = (tp + tn)/(tp + fp + fn + tn)
gm = np.sqrt(tp*tn)
print("KNN ppv: %f, npv: %f, accuracy: %f, gm: %f" % (ppv, npv, accuracy, gm))

target_names = ['class 0', 'class1']
print(classification_report(y_test, knn.predict(X_test), target_names=target_names))

with open('./knn.pkl', 'wb') as file:
# with open('./knn_wald.pkl', 'wb') as file:
    pickle.dump(knn, file)

# =======================
# MLP: multi-layer perception
# =======================
mlp = MLPClassifier(solver='adam', alpha=1e-5, learning_rate_init=0.1,
                    # activation='logistic',
                    hidden_layer_sizes=(10),
                    max_iter=100, random_state=1)
mlp = mlp.fit(X_train, y_train)
print('mlp', mlp.score(X_test, y_test))  # mlp
# mlp_scores = cross_val_score(mlp, X_train, y_train, cv=5)
# print('MLP_cross:', mlp_scores.mean())

# AUC
fpr, tpr, threshold = metrics.roc_curve(y_test, mlp.predict(X_test), pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print('auc:', metrics.auc(fpr, tpr))

tn, fp, fn, tp = confusion_matrix(y_test, mlp.predict(X_test)).ravel()
ppv = tp/(tp + fp)
npv = tn/(fn + tn)
accuracy = (tp + tn)/(tp + fp + fn + tn)
gm = np.sqrt(tp*tn)
print("MLP ppv: %f, npv: %f, accuracy: %f, gm: %f" % (ppv, npv, accuracy, gm))

target_names = ['class 0', 'class1']
print(classification_report(y_test, mlp.predict(X_test), target_names=target_names))

with open('./mlp.pkl', 'wb') as file:
# with open('./mlp_wald.pkl', 'wb') as file:
    pickle.dump(mlp, file)

duration = time() - t0
print("duration: %f" % duration)