from os.path import join

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class Modeling:

    def __int__(self):
        pass

    @staticmethod
    def visualize_pca(x_train, y_train):
        models_plot_path = 'data/plots/models/roc_curve/'

        # create a roc_curve object
        pca = PCA(n_components=2)

        # fit and transform
        result = pca.fit_transform(x_train)

        # plot the data
        plt.figure(figsize=(10, 10))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], label="Class #0",
                   alpha=0.5, c='r', s=70)
        ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], label="Class #1",
                   alpha=0.5, s=70)
        #ax.legend()
        #plt.savefig(join(models_plot_path, 'pca_model.png'))
        #plt.show()
