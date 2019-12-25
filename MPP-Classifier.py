import numpy as np
import os

# Load in data based on filepath
# get file path
test = os.path.abspath("synth.te.txt")
train = os.path.abspath("synth.tr.txt")

# Print file path to show where data files are
print(test)
print(train)

"""
    Create class for "Maximum Posterior Probability" - Classifier
    MPP uses three different discriminant functions
    1) linear machine --> Euclidean Distance --> independent features and equal variance
    2) linear machine --> Mahalanobis Distance --> equal covariance matrix
    3) Quadratic classifier --> arbitrary covariance matrices for each class
"""


def get_accuracy(y, y_model):
    """ return accuracy score """
    assert len(y) == len(y_model)
    return np.count_nonzero(y == y_model) / len(y)


def get_accuracy_dif_prior(model, X_training, y_training, X_testing, y_testing):
    prior_prob = np.linspace(0, 1, 100)
    list_accuracy = []
    for prior in prior_prob:
        model.fit(X_training, y_training, prior)
        y_model = model.predict(X_testing)
        accuracy = get_accuracy(y_testing, y_model)
        list_accuracy.append((accuracy, prior))
    return list_accuracy

    # get data from txt.file with certain format
    # return X (input features) and y (output labels) for each sample


def load_data(file):
    """
    Assume data format:
    feature1 (space) feature2 (space) ... label
    """
    # process data from file
    data = np.genfromtxt(file)
    # split features from output into X, y
    # change labels into integer values
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    # print(X)
    # print(y)
    return X, y


class MPP:
    def __init__(self, case):
        self.covavg_ = None
        self.varavg = None
        self.prior = None
        self.case = case

    # fit the model
    def fit(self, Train_data, y, prior=None):
        # empty dictionaries to store covariance and mean values
        self.covs_ = {}
        self.means_ = {}
        self.prior_prob_ = {}

        # get unqiue classes --> each unique item in label-array y represents one class
        self.classes_ = np.unique(y)
        # get number of unqiue classes
        self.classn_ = len(self.classes_)

        for c in self.classes_:
            arr = Train_data[y == c]
            # arbitrary covariance matrices used for MPP case 3
            self.covs_[c] = np.cov(np.transpose(arr))
            self.means_[c] = np.mean(arr, axis=0)
            if prior is None:
                self.prior_prob_[c] = len(arr) / Train_data.shape[0]
            else:
                if c == 0:
                    self.prior_prob_[c] = prior
                elif c == 1:
                    self.prior_prob_[c] = 1 - self.prior_prob_[c - 1]

            # compute average covariance matrix: covavg_ = sum of all covariance matrices divided by number of classes
        # Average covariance matrix used for MPP case 2
        self.covavg_ = sum(self.covs_.values()) / self.classn_

        ## Average variance (scalar) used for MPP case 1
        self.varavg_ = np.sum(np.diagonal(self.covavg_)) / Train_data.shape[1]
        # print(self.prior_)

    def predict(self, Test_data):
        y_pred = []
        gx_ = np.zeros(self.classn_)

        # for each sample (values for feature space) in test data compute g(x) for class 0 [c == 0] and for class 1
        # [c == 1] compare both results --> maximum value represents class
        for sample in Test_data:
            for c in self.classes_:
                if self.case == 1:
                    gx_[c] = -np.dot((sample - self.means_[c]).T, (sample - self.means_[c])) / \
                             (2 * self.varavg_ ** 2) + np.log(self.prior_prob_[c])
                elif self.case == 2:
                    gx_[c] = -0.5 * np.dot(np.dot((sample - self.means_[c]).T, np.linalg.inv(self.covavg_)),
                                           (sample - self.means_[c])) + np.log(self.prior_prob_[c])
                elif self.case == 3:
                    gx_[c] = -0.5 * np.dot(np.dot((sample - self.means_[c]).T, np.linalg.inv(self.covs_[c])),
                                           (sample - self.means_[c])) \
                             - 0.5 * np.log(np.linalg.det(self.covs_[c])) \
                             + np.log(self.prior_prob_[c])
            y_pred.append(gx_.argmax())

        return y_pred


# driver function
def main():
    Xtrain, ytrain = load_data(train)
    Xtest, ytest = load_data(test)
    model = MPP(1)
    # add prior probability for class 0 in method "fit" as third argument

    # get accuracy for different prior probabilities
    # accuracy = get_accuracy_dif_prior(model, Xtrain, ytrain, Xtest, ytest)

    # get accuracy for one prior probability (0.5/0.5)
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)
    accuracy = get_accuracy(ytest, y_model)
    print('Accuracy: {}! '.format(accuracy))


if __name__ == '__main__':
    main()
