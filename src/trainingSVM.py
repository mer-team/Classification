from sklearn import svm
from sklearn import preprocessing
import csv
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

targets_train = []
features_train = []
targets_test = []
features_test = []
features = []
targets = []
with open('./features.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        targets.append(row[0])
        features.append([row[1][2:],row[2][2:],row[3][2:]])

features_train.extend(x for x in features[:200])
targets_train.extend(targets[:200])
features_train.extend(features[226:425])
targets_train.extend(targets[226:425])
features_train.extend(features[451:650])
targets_train.extend(targets[451:650])
features_train.extend(features[676:875])
targets_train.extend(targets[676:875])

# standardization
#features_scaled = preprocessing.scale(features_train)
scaler = preprocessing.StandardScaler().fit(features_train)
scaled_values = scaler.transform(features_train) 
#min_max_scaler = preprocessing.MinMaxScaler()
#X_train_minmax = min_max_scaler.fit_transform(features_train)
#min_max_scaler = preprocessing.MaxAbsScaler()
#X_train_minmax = min_max_scaler.fit_transform(features_train)
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html - optimization parameters
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']
clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
clf.fit(scaled_values,targets_train)  
#clf.fit(X_train_minmax,targets_train)  
dump(clf,'trainedModel.joblib')
dump(scaler,'scaler.joblib')

#Happy Q1, tense -Q2, sad - q3, calm - Q4