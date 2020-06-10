from os.path import join
from time import time

from numpy import mean, std
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, plot_confusion_matrix, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Name: Models
#
# Description: This class represents a model and supports it to be trained for
# determining predictions and calculating its accuracy.
#
from sklearn.neighbors import KNeighborsClassifier

from src.framework.utils.utilities import Utilities as utils


class Models:

    def __init__(self, X, y, test_size=0.3, random_state=0, scaler_type=1):
        self.models_roc_curve_plot_path = 'data/plots/models/roc_curve/'
        self.scaler_suffix = 'standard'
        self.test_size = test_size
        self.random_state = random_state
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        if scaler_type == 1:
            self.scaler_suffix = 'minmax'

    @staticmethod
    def knn(X, y):
        t0 = time()
        print("Begin Execution for KNeighbors Classifier - Training")

        classifier = KNeighborsClassifier(n_jobs=-1)
        distance = 2  # weight points by the inverse of their distance
        k_min = 1
        k_max = 25
        parameter_grid = {
            "n_neighbors": range(k_min, k_max, 2),
            "p": range(1, distance + 1)
        }
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
        grid_searcher = GridSearchCV(estimator=classifier, param_grid=parameter_grid, cv=cv, scoring=(
            "balanced_accuracy", "f1", "precision", "recall", "roc_auc"), refit=False, n_jobs=-1,
                                     return_train_score=True)
        grid_searcher.fit(X, y)

        tt = time() - t0
        print("Execution Time for KNeighbors Classifier - Training: {} seconds".format(utils.convert(round(tt, 3))))
        print(grid_searcher.cv_results_)

    def conf_matrix(self, y_test, y_pred, classifier_name):

        print("Confusion Matrix Results - '{}' Classifier".format(classifier_name))
        matrix = confusion_matrix(y_test, y_pred)
        print(matrix)

    def class_report(self, y_test, y_pred, classifier_name):
        print("Classification Report - '{}' Classifier".format(classifier_name))
        print(classification_report(y_test, y_pred))

    def plot_roc_curve(self, classifier, x_test, y_test, title):
        classifier_roc_auc = roc_auc_score(y_test, classifier.predict(x_test))
        fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(x_test)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label='%s (area = %0.2f' % (title, classifier_roc_auc))
        plt.plot([0, 1], [0, 1], 'b--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(join(self.models_roc_curve_plot_path, '{}_{}.png'
                         .format(title.replace(" ", "_"), self.scaler_suffix)))

    def logistic_regression_model(self):
        # split into train and test sets

        print("\nRunning Logistic Regression Model..")

        t0 = time()
        # fit the modeling
        model = LogisticRegression(solver='liblinear')
        model.fit(self.x_train, self.y_train)

        # evaluate our modeling using 70/30 ratio by default
        lr_score = model.score(self.x_test, self.y_test)
        lr_predict = model.predict(self.x_test)

        # evaluate our predictions
        accuracy = accuracy_score(self.y_test, lr_predict)

        print("--------------------------------------------")
        print("Logistic Regression Prediction Results: ")

        lr_accuracy_score = accuracy_score(lr_predict, self.y_test)
        lr_precision_score = precision_score(lr_predict, self.y_test, average='weighted')
        lr_recall_score = recall_score(lr_predict, self.y_test, average='weighted')
        lr_f1score = f1_score(lr_predict, self.y_test, average='weighted')

        runtime = time() - t0

        print('Accuracy: \t\t\t{:.4f}'.format(lr_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(lr_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(lr_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(lr_f1score))

        print("Logistic Regression Classifier Accuracy: {:.2f}".format(accuracy))
        print('Logistic Regression Classifier: \t\t{:.2f}% in {:.2f} seconds'.format(lr_score * 100, runtime))
        self.conf_matrix(self.y_test, lr_predict, 'Linear Regression')
        self.class_report(self.y_test, lr_predict, 'Linear Regression')

        self.plot_roc_curve(model, self.x_test, self.y_test, 'Linear Regression')

    # @staticmethod
    # def random_forest_model(self:

    #   model = RandomForestClassifier(random_state=101)
    # TODO: complete implementation for RFC


    def gradient_boost_classifier(self):

        t0 = time()
        GB = GradientBoostingClassifier()
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(GB, self.x_train, self.y_train, scoring='accuracy', cv=cv, n_jobs=-1,
                                   error_score='raise')

        print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
        GB.fit(self.x_train, self.y_train)

        # evaluate our modeling using 70/30 ratio by default
        gb_score = GB.score(self.x_test, self.y_test)
        gb_predict = GB.predict(self.x_test)

        # evaluate our predictions
        accuracy = accuracy_score(self.y_test, gb_predict)

        print("--------------------------------------------")
        print("Logistic Regression Prediction Results: ")

        gb_accuracy_score = accuracy_score(gb_predict, self.y_test)
        gb_precision_score = precision_score(gb_predict, self.y_test, average='weighted')
        gb_recall_score = recall_score(gb_predict, self.y_test, average='weighted')
        gb_f1score = f1_score(gb_predict, self.y_test, average='weighted')

        runtime = time() - t0

        print('Accuracy: \t\t\t{:.4f}'.format(gb_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(gb_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(gb_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(gb_f1score))

        print("Gradient Boosting Classifier Accuracy: {:.2f}".format(accuracy))
        print('Gradient Boosting Classifier: \t\t{:.2f}% in {:.2f} seconds'.format(gb_score * 100, runtime))
        self.conf_matrix(self.y_test, gb_predict, 'Linear Regression')
        self.class_report(self.y_test, gb_predict, 'Linear Regression')

        self.plot_roc_curve(GB, self.x_test, self.y_test, 'Linear Regression')


    @staticmethod
    def knn_classifier(self):
        t0 = time()
        for i in range(1, len(self.xtrain) + 1):
            knn = KNeighborsClassifier(n_neighbors=i)
            model = knn.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            print("For iteration '{}', accuracy is : ".format(accuracy_score(self.y_test, pred)))

        tt = time() - t0
        print("read_data() took {} seconds".format(self.utils.convert(round(tt, 3))))

    def calc_confusion_matrix(self, title, show_plot=False):
        display = plot_confusion_matrix(self.classifier, self.x_test, self.y_test,
                                        display_labels=self.class_names, cmap=plt.cm.Blues,
                                        normalize=True)
        display.ax_.set_title(title)

        if show_plot:
            plt.show()
