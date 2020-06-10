import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from os.path import join

# import for plotting
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder


class FeatureSelection:

    def __init__(self):
        self.feature_selection_plot_path = 'data/plots/features/feature_selection/'

    # Name: feature_selection_pca
    #
    # Description: Performs a PCA transform

    def feature_selection_pca(self, data, num_components=2, show_plot=False):
        print("Running PCA - Feature Selection")

        num_elements = data.values
        pca = PCA(n_components=num_components)
        x = pca.fit_transform(num_elements)
        plt.figure(figsize=(5, 5))
        plt.scatter(x[:, 0], x[:, 1])
        plt.savefig(join(self.feature_importance_plot_path, 'roc_curve.png'))

        if show_plot:
            plt.show()

        print("Finished Running PCA...")

    @staticmethod
    def random_forest_regressor(self, data, target_name='Label'):
        print("\nRunning Random Forest Regressor - Feature Selection")
        features = (data.drop([target_name], axis=1)).columns.values
        label_encoder = LabelEncoder()

        # encode the target column (i.e. Label)
        data.iloc[:, -1] = label_encoder.fit_transform(data.iloc[:, -1])

        # capture the training set and drop the target from original dataset
        x_train = data.drop([target_name], axis=1)
        y_train = data.iloc[:, -1].values.reshape(-1, 1)

        # train our model
        random_forest = RandomForestRegressor()
        random_forest.fit(x_train, y_train)

        # display the results
        results = map(lambda item: round(item, 3), random_forest.feature_importances_)
        print("\nFeatures By Score: {}".format(results))

    def highest_correlated_features(self, x_train, y_train, top_n=10):
        print("\nRunning Highest Correlated Features - Feature Selection")

        corr_matrix = x_train.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=top_n).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

        # print("\nFeature Correlation scores: \n" + str(corr_matrix.to_string()) + "\n")
        print("Highest {} Correlated Features To Drop: {}".format(len(to_drop), to_drop))
        print("Finished Running Highest Correlated Features - Feature Selection")

        return to_drop

    def random_forest_classifier(self, x_train, y_train, columns, random_state=0, show_plot=False):
        print("\nRunning Random Forest Classifier - Feature Selection")
        # create classifier instance
        rf = RandomForestClassifier(random_state=101, n_jobs=-1)
        rf.fit(x_train, y_train)

        # extract important features
        score = np.round(rf.feature_importances_, 3)
        importances = pd.DataFrame({'feature': x_train.columns, 'importance': score})
        to_drop = importances[importances['importance'] < 0.001]
        importances = importances.sort_values('importance', ascending=False).set_index('feature')

        # plot importances
        plt.rcParams['figure.figsize'] = (11, 4)
        plt.tight_layout()
        importances.plot.bar()

        if show_plot:
            plt.show()

        print("# Items Dropped: {}".format(len(to_drop)))
        print("Finished Running Random Forest Classifier - Feature Selection")

        return to_drop['feature'].values.tolist()
