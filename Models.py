from abc import ABC
import numpy as np
from sklearn.utils import shuffle


####################
# Helpers
####################
def batch_generator(X, y, batch_size=None):
    data_size = len(X)
    X, y = shuffle(X, y)  # Should happen before every epoch.. Good idea
    num_batches = int(np.ceil(data_size / batch_size))
    for i in range(num_batches):
        start_ind = i * batch_size
        end_ind = min((i + 1) * batch_size, data_size)
        yield X[start_ind:end_ind], y[start_ind:end_ind]


def dummy_data_gen(betas, data_size=10000, noise_var=0.1, std_range=None):
    feature_size = len(betas)
    betas = np.array(betas).reshape((feature_size, 1))
    X = np.random.normal(0, 1, (data_size, feature_size))

    # Change the Standard Deviation of Each column
    if std_range is not None:
        feature_sds = np.random.random(feature_size).reshape(feature_size, 1) * std_range
        transformer = np.zeros((feature_size, feature_size))
        np.fill_diagonal(transformer, feature_sds)
        X = np.dot(X, transformer)

    noise = np.random.normal(0, noise_var, (data_size, 1))
    y = np.dot(X, betas) + noise

    return X, y


####################
# Model Classes
####################
# Define an interface for Classifiers
class AbstractClassifier(ABC):
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class LinearRegression(AbstractClassifier):
    def __init__(self, batch_size=None, epochs=500, opt=None):  # None means full Batch
        self.betas = None  # (d+1) x 1 vector where X in R^d
        self.error = None  # n x 1 vector where n is batch size
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_change_thresh = 1e-6  # Some arbitrarily small value
        self.opt = opt

        self.history = {'loss': []}  # Dont know if we want to track other stuff. We'll change this later

    def loss(self, batch_x, batch_y):
        """
        Mean Squared Error Loss. 1/(2m) || X * Beta - y ||^2
        :param batch_x:
        :param batch_y:
        :return:
        """
        m = len(batch_y)  # TODO: We dont need this if we didnt have that last annoying batch
        self.error = np.dot(batch_x, self.betas) - batch_y  # (X * Beta - y)
        loss = (1 / (2 * m)) * np.dot(self.error.T, self.error)[0][0]

        return loss

    def grad(self, batch_x, batch_y):
        """
        Gradient of the MSE Loss - 1/m * X^T * ( X * Beta - y )
        :return:
        """
        m = len(batch_y)
        grads = (1 / m) * np.dot(batch_x.T, self.error)  # (d x n) . (n x 1) = (d x 1)
        self.opt.state['grads'] = grads  # Grad should never be modified by opt

    def fit(self, X, y, epoch_loss=True):
        # Attach a constant value to X for B_naught
        constant = np.ones((X.shape[0], 1))
        X = np.hstack((X, constant))
        self.batch_size = len(X) if self.batch_size is None else self.batch_size

        # Initialize Weights to random Normal (Zero not good for Gradient Descent)
        dims = X[0].shape[0]
        np.random.seed(42)  # So we can make sure every optimizer starts at the same point
        self.betas = np.random.normal(0, 0.01, (dims, 1))

        # This is mutable. When opt.step modifies its weight state, the betas get updated here as well.
        self.opt.state['weights'] = self.betas
        global_step = 0
        for epoch in range(1, self.epochs + 1):
            # Shuffle dataset and return a batch generator
            batch_gen = batch_generator(X, y, self.batch_size)
            for batch_x, batch_y in batch_gen:

                loss = self.loss(batch_x, batch_y)  # ForwardProp

                if global_step == 0:
                    self.history['loss'].append(loss)
                self.grad(batch_x, batch_y)  # BackProp
                self.opt.step()  # Pytorch like magic.
                global_step += 1

                if not epoch_loss:
                    self.history['loss'].append(loss)

            if epoch_loss:
                self.history['loss'].append(loss)

    def predict(self, X):
        constant = np.ones((X.shape[0], 1))
        X = np.hstack((X, constant))
        return np.dot(X, self.betas)
