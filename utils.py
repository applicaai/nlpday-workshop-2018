"""
Utilities for PolEval 2018 notebook
"""

# lots of import for checking the environment
import numpy as np
import scipy
import sklearn
import modAL
import tqdm
import lime

# actual imports
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import fetch_20newsgroups
from typing import Tuple, List, Any, Union


def load_news20(real: bool = False) -> Tuple[Tuple[list, list], Tuple[list, list], List[str]]:
    """
    Loads the 20 News Group dataset split by train and test as a raw text with class names,
    see: http://qwone.com/~jason/20Newsgroups/ for details.
    :param real: bool, default False
        Whether to use the `real` dataset, with headers, footers and quotes stripped

    :return: tuple (train set, test set, class names)
        Train and test set as tuples (data, target) and the class names is a list
    """

    if real:
        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    else:
        train_data = fetch_20newsgroups(subset='train')
        test_data = fetch_20newsgroups(subset='test')

    # extract class names
    class_names = train_data['target_names']

    # extract raw data and labels
    X_raw_train, y_train_full = train_data['data'], train_data['target']
    X_raw_test, y_test_full = test_data['data'], test_data['target']

    # reformat class names to more readable
    class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in class_names]
    class_names[3] = 'pc.hardware'
    class_names[4] = 'mac.hardware'
    class_names[5] = 'ms-windows.x'

    return (X_raw_train, y_train_full), (X_raw_test, y_test_full), class_names


def passive_sampling(classifier: Any,
                     X: Union[np.ndarray, list],
                     n_instances: int = 1,
                     rng: np.random.RandomState = None) -> Tuple[list, np.ndarray]:
    """
    Passive sampling strategy for AL that picks samples uniformly at random.
    :param classifier: not used
        Messy way of keeping with the query function abstraction
    :param X: np.ndarray, list
        Data pool to pick queries from
    :param n_instances: int, default 1
        Number of instances to pick
    :param rng: np.random.RandomState, optional
        Random number generator to use for seeding

    :return: tuple (list, np.ndarray)
        Tuple with indices of queried samples and the samples themselves
    """

    if rng is not None:
        query_ids = rng.choice(X.shape[0], size=n_instances, replace=False)
    else:
        query_ids = np.random.choice(X.shape[0], size=n_instances, replace=False)

    return query_ids, X[query_ids]


def delete_rows_csr(arr: scipy.sparse.csr_matrix,
                    obj: Union[np.ndarray, list]):
    """
    Deletes rows of a sparse matrix returning the new matrix.
    :param arr: scipy.sparse.csr_matrix
        Matrix from which to delete the rows
    :param obj: 1D np.ndarray, list
        Indices of rows to delete

    :return: scipy.sparse.csr_matrix
        Sparse matrix with rows deleted
    """

    if not isinstance(arr, scipy.sparse.csr_matrix):
        raise ValueError("This function works only for CSR format, try using X.tocsr()")
    obj = list(obj)
    mask = np.ones(arr.shape[0], dtype=bool)
    mask[obj] = False
    return arr[mask]


def extract_subset(X: np.ndarray,
                   y: np.ndarray,
                   classes: List[Union[str, int]],
                   names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a sub-dataset with the chosen classes, which can be passed by name or index.
    :param X: np.ndarray
        Data to extract from
    :param y: np.ndarray
        Target labels used for the data
    :param classes: list of either strings or ints
        Which classes to extract, if passed by name, the `names` parameter is required
    :param names: list of strings, optional
        List with class names, order of this list determines class labels instances,
        required when passing classes by name

    :return: tuple with (data, labels)
        Extracted sub-dataset
    """

    assert len(classes) >= 2, "Please pass at least 2 classes"

    # global mask for extracting the subset with desired classes
    mask = np.zeros(shape=y.shape, dtype=bool)
    # iterate over the classes to extract
    for class_name in classes:
        if names is None:  # using class index
            assert isinstance(class_name, int), "Please pass classes as ints if you do not pass `names` parameter"
            assert class_name in np.unique(y)
            class_label = class_name
        else:  # using class names
            assert isinstance(class_name, str), \
                "Unsupported class type: {}. Supported are: str, int".format(type(class_name))
            assert class_name in names, "Unrecognized class name: {}".format(class_name)
            class_label = names.index(class_name)

        # filter the samples of current class
        class_mask = (y == class_label)
        # add class mask to the global mask
        mask = np.logical_or(mask, class_mask)

    return X[mask], y[mask]


def plot(scores: List[Union[np.ndarray, list]],
         names: List[str]) -> plt.axes:
    """
    Plotting for AL experiments, supports both single and multiple experiments for each query strategy.
    With multiple experiments it will plot mean and std score at every iteration.
    :param scores: np.ndarray, list
        List of scores for each of the strategies, for a single strategy if the score is 1D it will just plot it,
        if it's 2D it will average over multiple runs nad plot mean and std
    :param names: list
        Names from each of the strategies to use as the legend

    :return: matplotlib.pyplot.axes
        Ax with the plot
    """

    # prepare the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # X axis for the plot
    budget = len(scores[0])
    t = np.arange(budget)

    # enumerate the query strategies
    for i, values in enumerate(scores):
        assert len(values) == budget

        values = np.array(values)
        if len(values.shape) == 1:  # single experiment
            ax.plot(t, values, label=names[i])
        elif len(values.shape) == 2:  # multiple experiments
            t = np.arange(values.shape[1])
            # calculate mean and std
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            # plot the moments
            ax.plot(t, mean, label=names[i])
            ax.fill_between(t, mean - std, mean + std, alpha=0.1)

    # set appropriate title
    if len(np.array(scores[0]).shape) == 1:
        title = 'Single run comparision between query strategy'
    elif len(np.array(scores[0]).shape) == 2:
        title = 'Comparision between query strategies averaged over {} runs with std'.format(len(scores[0]))
    else:
        raise ValueError("Unsupported scores shape: {}".format(np.array(scores[0]).shape))

    # add cosmetic stuff
    ax.set_xlabel('Number of Labelled Samples')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(title)
    ax.legend()

    return ax


def committee_decision(votes: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Committee voting prediction for binary classification, this function is not recommended for multiclass tasks
    as well as committees with even number of models due to lack of obvious consensus.
    :param votes: np.ndarray, list
        Committee votes for each sample
    :return: np.ndarray
        Prediction on the committee for each sample
    """

    if np.array(votes).shape[1] % 2 == 0:
        warnings.warn("The decision for a committee with even number of models may be not deterministic")

    def f(vote):
        labels, counts = np.unique(vote, return_counts=True)
        return labels[np.argmax(counts)]

    return np.array(list(map(f, votes)), dtype=int)
