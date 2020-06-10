from os.path import join
from time import time

# import for plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
# Model Fitting
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from src.framework.utils.utilities import Utilities


class FeatureImportance:

    def __init__(self):
        self.feature_importance_plot_path = 'data/plots/features/feature_importance/'
        self.utils = Utilities

    # Name: extra_trees_classifier
    #
    # Description: Discovers the important features using the ExtraTreesClassifier
    #
    def extra_trees_classifier(self, x_train, y_train, k=10, show_plot=False):
        t0 = time()
        print("\nRunning Feature Importance - Extra Trees Classifier...")

        # Feature Importance
        model = ExtraTreesClassifier()
        model.fit(x_train, y_train)

        plt.figure(figsize=(10, 10))

        # plot feature importance
        important_features = pd.Series(model.feature_importances_, index=x_train.columns)

        # plot the series in a bar graph
        important_features.nlargest(k).plot(kind='barh')

        # display the bar graph
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(join(self.feature_importance_plot_path, 'Extra_Trees_Classifier.png'))
        print("Created a Feature Importance Plot - Extra Trees Classifier")
        tt = time() - t0

        if show_plot:
            plt.show()
        print("Finished Running Feature Importance - Extra Trees Classifier...")
        print("Execution Time for extra_trees_classifier(): {} seconds".format(self.utils.convert(round(tt, 3))))

    #
    # Name: random_forest_classifier
    #
    # Description: Discovers the important features using the Random Forest Classifier
    #
    def random_forest_classifier(self, x_train, y_train, num_estimators=100, show_plot=False):
        # TODO: This is completed
        t0 = time()
        print("\nRunning Feature Importance - Random Forest Classifier...")
        # Feature Selection
        rfc = RandomForestClassifier(random_state=0, n_jobs=-1)

        # fit random forest classifier on the training set
        rfc.fit(x_train, y_train)

        # extract important features
        score = np.round(rfc.feature_importances_, 3)
        importances = pd.DataFrame({'feature': x_train.columns, 'importance': score})
        importances = importances.sort_values('importance', ascending=False).set_index('feature')

        # select the features which importance is greater than the mean importance of
        # all the features by default
        #
        sel = SelectFromModel(RandomForestClassifier(n_estimators=num_estimators))
        sel.fit(x_train, y_train)
        selected_features = x_train.columns[(sel.get_support())]
        print("Random Forest Selected Features: {}".format(selected_features))

        # plot importances
        plt.rcParams['figure.figsize'] = (18, 11)
        plt.title('Feature Importance - Random Forest Classifier')
        importances.plot.bar()
        plt.savefig(join(self.feature_importance_plot_path, 'RandomForestClassifier.png'))
        plt.tight_layout()

        if show_plot:
            plt.show()

        tt = time() - t0
        print("Finished Running Feature Importance - Random Forest Classifier...")
        print("Execution Time for random_forest_classifier(): {} seconds".format(self.utils.convert(round(tt, 3))))

        return selected_features

    def linear_regression_classifier(self, data, y_train, show_plot=False):
        t0 = time()
        print("\nRunning Feature Importance - Linear Regression...")
        lr = LinearRegression()
        lr.fit(data, y_train)

        # retrieve the importance
        importances = lr.coef_

        for i, v in enumerate(importances):
            print("Feature: %0d, Score: %.5f" % (i, v))

        plt.rcParams['figure.figsize'] = (18, 11)
        plt.title('Feature Importance - Linear Regression')
        plt.bar([x for x in range(len(importances))], importances)
        plt.savefig(join(self.feature_importance_plot_path, 'LinearRegression.png'))
        plt.tight_layout()

        if show_plot:
            plt.show()

        tt = time() - t0
        print("Running Feature Importance - Linear Regression...")
        print("Execution Time for linear_regression_classifier(): {} seconds"
              .format(self.utils.convert(round(tt, 3))))

    def logistic_regression_classifier(self, x_train, y_train, show_plot=False):
        t0 = time()
        print("\nRunning Feature Importance - Logistic Regression...")
        lr = LogisticRegression()
        lr.fit(x_train, y_train)

        importances = lr.coef_[0]

        for i, v in enumerate(importances):
            print("Feature: %0d, Score: %.5f" % (i, v))

        plt.rcParams['figure.figsize'] = (18, 11)
        plt.title('Feature Importance - Logistic Regression')
        plt.bar([x for x in range(len(importances))], importances)
        plt.savefig(join(self.feature_importance_plot_path, 'LogisticRegression.png'))
        plt.tight_layout()

        if show_plot:
            plt.show()
        tt = time() - t0
        print("Finished Running Feature Importance - Logistic Regression...")
        print("Execution Time for logistic_regression_classifier(): {} seconds"
              .format(self.utils.convert(round(tt, 3))))

    def decision_tree_regression_classifier(self, x_train, y_train, show_plot=False):
        t0 = time()
        print("\nRunning Feature Importance - Decision Tree Regression...")
        model = DecisionTreeRegressor()

        # fit the modeling
        model.fit(x_train, y_train)

        # retrieve the importances
        importances = model.feature_importances_

        for i, v in enumerate(importances):
            print("Feature: %0d, Score: %.5f" % (i, v))

        plt.rcParams['figure.figsize'] = (18, 11)
        plt.title('Feature Importance - Decision Tree Regression')
        plt.bar([x for x in range(len(importances))], importances)
        plt.savefig(join(self.feature_importance_plot_path, 'DecisionTreeRegression.png'))
        plt.tight_layout()

        if show_plot:
            plt.show()

        tt = time() - t0
        print("Finished Running Feature Importance - Decision Tree Regression...")
        print("Execution Time for decision_tree_regression_classifier(): {} seconds"
              .format(self.utils.convert(round(tt, 3))))

    def decision_tree_classifier(self, x_train, y_train, show_plot=False):
        t0 = time()
        print("\nRunning Feature Importance - Decision Tree Classifier...")
        model = DecisionTreeClassifier()

        # fit the modeling
        model.fit(x_train, y_train)

        # retrieve the importances
        importances = model.feature_importances_

        for i, v in enumerate(importances):
            print("Feature: %0d, Score: %.5f" % (i, v))

        plt.rcParams['figure.figsize'] = (18, 11)
        plt.title('Feature Importance - Decision Tree Classifier')
        plt.bar([x for x in range(len(importances))], importances)
        plt.savefig(join(self.feature_importance_plot_path, 'DecisionTreeClassifier.png'))
        plt.tight_layout()

        if show_plot:
            plt.show()

        tt = time() - t0
        print("Finished Running Feature Importance - Decision Tree Classifier...")
        print("Execution Time for decision_tree_classifier(): {} seconds".format(self.utils.convert(round(tt, 3))))

    def knn_classifier(self, x_train, y_train, k=3, display=False):
        t0 = time()
        print("\nRunning Feature Importance - KNN Classifier...")
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

        model.fit(x_train, y_train)

        # perform permutation importance
        results = permutation_importance(model, x_train, y_train, scoring='accuracy')

        # retrieve the importances
        importances = results.importances_mean

        for i, v in enumerate(importances):
            print("Feature: %0d, Score: %.5f" % (i, v))

        plt.rcParams['figure.figsize'] = (18, 11)
        plt.title('Feature Importance - KNeighbors Classifier')
        plt.bar([x for x in range(len(importances))], importances)
        plt.savefig(join(self.feature_importance_plot_path, 'KNClassifier.png'))
        plt.tight_layout()

        if display:
            plt.show()

        tt = time() - t0
        print("Finished Running Feature Importance - KNN Classifier...")
        print("Execution Time for knn_classifier(): {} seconds".format(self.utils.convert(round(tt, 3))))

    def select_top_k_best(self, x_train, y_train, k_best=20):
        t0 = time()
        print("\nRunning SelectKBest...")
        # use SelectKBest class to extract the top K best features
        best_features = SelectKBest(score_func=chi2, k=k_best)
        f: object = best_features.fit(x_train, y_train)

        # calculate score for each feature in regards to the target
        df_scores = pd.DataFrame(f.scores_)

        # capture the columns in our feature set
        df_columns = pd.DataFrame(x_train.columns)

        # concat two dataframes for better visualization
        feature_scores = pd.concat([df_columns, df_scores], axis=1)

        print("SelectKBest Features (Top {}):".format(k_best))

        # name the dataframe columns
        feature_scores.columns = ['Specs', 'Score']

        k_largest = feature_scores.nlargest(k_best, 'Score')
        # display the Top K best features
        print("SelectKBest Results: {}".format(k_largest))
        print("Finished Running SelectKBest...")
        tt = time() - t0
        print("Execution Time for select_top_k_best(): {} seconds".format(self.utils.convert(round(tt, 3))))
        return k_largest['Specs'].values
