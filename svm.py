import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from imageprocessing import ImageProcess
from sklearn.svm import SVC
from sklearn.externals import joblib


class SVM():

    def __init__(self):
        self.svm = SVC(C=0.001, kernel='linear', verbose=True)

    def train(self, X_train, y_train, X_test, y_test):
        from sklearn.model_selection import cross_val_score
        cv_performance = cross_val_score(self.svm, X_train, y_train, cv=20)
        test_performance = self.svm.fit(X_train, y_train).score(X_test, y_test)       
        self.save(self.svm, True)

    def search_parameter(self, X_train, y_train, X_test, y_test):

        learning_algorithm = SVC(kernel='linear', random_state=101)

        search_space = [{'kernel': ['linear'], 'C': np.logspace(-3, 3, 7)},
                        {'kernel': ['rbf'], 'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 2, 6)}]

        gridsearch = GridSearchCV(learning_algorithm, param_grid=search_space, refit=True, cv=10)
        gridsearch.fit(X_train, y_train)
        print('Best parameter: %s' % str(gridsearch.best_params_))


    def save(self, model, check, filename='svmsaved.sav'):

        if check != True:
            ans = input('Do you want to save the template?')
        else:
            ans = 'y'

        if ans == 'y':
            joblib.dump(model, filename)
            print("Saved")
        else:
            print("Not Saved")

    def test(self, X_test, Y_test, filename='svmsaved.sav'):
        # load the model from disk
        loaded_model = joblib.load(filename)
        result = loaded_model.score(X_test, Y_test)
        print(result)

    def predict(self, test, filename='svmsaved.sav'):
        model = joblib.load(filename)
        predicted_list = []
        for i in test:
            i = i.reshape((1, len(i)))
            pred = model.predict(i)
            print(pred)
            predicted_list.append(pred)

        return predicted_list


class CallSvm():

    def data_creation(self, empty_path='dataset/rawdataset/empty',
                                occupied_path='dataset/rawdataset/occupied'):
        empty = image_process().load_image_from_path(empty_path)
        occup = image_process().load_image_from_path(occupied_path)
        X = np.concatenate([empty, occup])
        y = np.concatenate([np.zeros(len(empty), dtype = np.int), np.ones(len(occup), dtype = np.int)])
        return train_test_split(X, y, test_size = 0.33)


    def training(self):
        X_train, X_test, y_train, y_test = self.data_creation()

        features_train = ImageProcess().extract_features(X_train)
        features_test = ImageProcess().extract_features(X_test)

        SVM().train(features_train, y_train, features_test, y_test)
        features_test = features_test[94].reshape(1, -1)
        SVM().predict(features_test)


    def testing(self):
        X_train, X_test, y_train, y_test = self.data_creation()
        features_test = image_process().extract_features(X_test)
        SVM().test(features_test, y_test)

#CallSvm().training()