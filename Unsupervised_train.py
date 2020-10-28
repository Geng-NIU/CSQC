import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from time import time
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc

t0 = time()


def transfer_1(a):
# defining transfer function: -1-->1 & 1-->0
    for i in range(len(a)):
        if a[i] == -1:
            a[i] = 1
        elif a[i] == 1:
            a[i] = 0
    return a


# loading training data: data_X
# Tree algorithms for outputing the feature important index.
n = 100
feature_select = ExtraTreesClassifier(n_estimators=n, random_state=111)
feature_select.fit(data_X, data_y)

# normalization
data_X = preprocessing.minmax_scale(data_X, feature_range=(-1, 1))
data_X = preprocessing.scale(data_X)

# split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=3)

# ===================
# Isolation forest
# ===================
IsF = IsolationForest(behaviour='new',
                      max_samples=5000,
                      n_estimators=10,
                      max_features=data_X.shape[1],
                      random_state=222,
                      contamination='auto').fit(data_X)

# y_pred_train = IsF.predict(data_X)
y_pred_train = IsF.fit_predict(data_X)
y_pred_train = transfer_1(y_pred_train)
fpr, tpr, threshold = metrics.roc_curve(data_y, y_pred_train, pos_label=1)
tn, fp, fn, tp = confusion_matrix(data_y, y_pred_train).ravel()
ppv = tp/(tp + fp)
npv = tn/(fn + tn)
accuracy = (tp + tn)/(tp + fp + fn + tn)
gm = np.sqrt(tp*tn)
target_names = ['class0', 'class1']
print(classification_report(data_y, y_pred_train, target_names=target_names))

with open('./IsF.pkl', 'wb') as file:
# with open('./IsF_wald.pkl', 'wb') as file:
    pickle.dump(IsF, file)


# =================================
# K-Means
# =================================
t0 = time()
kmeans = KMeans(
                n_clusters=2,
                init='random',
                precompute_distances='auto',
                max_iter=1000,
                algorithm='auto',
                copy_x=True,
                n_init=200, random_state=0
                ).fit(data_X)

fpr, tpr, threshold = metrics.roc_curve(data_y, kmeans.predict(data_X), pos_label=1)
tn, fp, fn, tp = confusion_matrix(data_y, kmeans.predict(data_X)).ravel()
ppv = tp/(tp + fp)
npv = tn/(fn + tn)
accuracy = (tp + tn)/(tp + fp + fn + tn)
gm = np.sqrt(tp*tn)
target_names = ['class0', 'class1']
print(classification_report(data_y, kmeans.labels_, target_names=target_names))

with open('./K_means.pkl', 'wb') as file:
# with open('./K_means_wald.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

duration = time() - t0
print(duration)