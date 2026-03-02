import numpy as np
import operator as op
import matplotlib.pyplot as plt
import  numpy as np
import datetime
from collections import Counter
import copy
import matplotlib

"""
Sliding Window k-NN Implementation
-----------------------------------

This file implements a k-Nearest Neighbor classifier using a
modified sliding-window L2 distance metric to mitigate small
spatial misalignments in image data.

The method is based on:

Grover, D., & Toghi, B. (2020).
"MNIST Dataset Classification Utilizing k-NN Classifier with Modified Sliding-Window Metric"
In: Arai, K., Kapoor, S. (eds) Advances in Computer Vision.
Springer, Cham. pp. 583–591.
DOI: 10.1007/978-3-030-17798-0_47
"""


## Functions


class SlidingWindowKNN:
    """k-NN classifier using sliding-window distance metric.

    Only the distance is altered compared with vanilla k-NN.  Images are not
    projected or weighted; votes are uniform.  During fitting we precompute
    nine shifted copies of the training set so that distances can be
    evaluated in O(9nd) time instead of recomputing shifts repeatedly.
    """

    def __init__(self):
        self._trained = False

    @staticmethod
    def shift_image(img_2d, dx, dy):
        padded = np.pad(img_2d, 1, mode="constant", constant_values=0)
        return padded[1+dx:1+dx+28, 1+dy:1+dy+28]

    def fit(self, X_train_img, y_train):
        self.X_train_img = X_train_img
        self.y_train = y_train
        self.n_train = X_train_img.shape[0]
        self.X_train_flat = X_train_img.reshape(self.n_train, -1)
        shifts = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 0), (0, 1),
                  (1, -1), (1, 0), (1, 1)]
        self.shifted_sets = []
        for dx, dy in shifts:
            shifted = np.array([
                self.shift_image(img, dx, dy) for img in X_train_img
            ])
            self.shifted_sets.append(shifted.reshape(self.n_train, -1))
        self._trained = True

    def _min_distances(self, x_flat):
        min_dists = np.full(self.n_train, np.inf)
        for shifted in self.shifted_sets:
            dists = np.linalg.norm(shifted - x_flat, axis=1)
            min_dists = np.minimum(min_dists, dists)
        return min_dists

    def predict(self, X_query_img, k):
        assert self._trained, "Call fit() before predict()"
        preds = []
        for x in X_query_img.reshape(-1, 784):
            dists = self._min_distances(x)
            idx = np.argpartition(dists, k)[:k]
            preds.append(Counter(self.y_train[idx]).most_common(1)[0][0])
        return np.array(preds)

    def _train_distance_matrix(self):
        n = self.n_train
        D = np.full((n, n), np.inf)
        sq1 = np.sum(self.X_train_flat**2, axis=1)
        for shifted in self.shifted_sets:
            sq2 = np.sum(shifted**2, axis=1)
            mat = np.sqrt(sq1[:, None] + sq2[None, :] - 2.0 *
                          (self.X_train_flat @ shifted.T))
            D = np.minimum(D, mat)
        np.fill_diagonal(D, np.inf)
        return D

    def kfold_cv(self, ks, n_folds=10, random_state=None):
        """Perform k-fold cross validation on the training set.

        Arguments:
            ks: iterable of k values to evaluate
            n_folds: number of folds (default 10)
            random_state: seed for shuffling
        Returns:
            dict mapping k -> cv error
        """
        assert self._trained, "Call fit() before cross-validation"
        n = self.n_train
        D = self._train_distance_matrix()
        indices = np.arange(n)
        if random_state is not None:
            rng = np.random.default_rng(random_state)
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)
        fold_sizes = [n // n_folds] * n_folds
        for i in range(n % n_folds):
            fold_sizes[i] += 1
        results = {k: 0.0 for k in ks}
        start = 0
        classes = np.unique(self.y_train)
        for size in fold_sizes:
            test_idx = indices[start:start+size]
            train_idx = np.setdiff1d(indices, test_idx)
            D_fold = D[test_idx][:, train_idx]
            labels_train = self.y_train[train_idx]
            labels_test = self.y_train[test_idx]
            for k in ks:
                errors = 0
                for i_row, drow in enumerate(D_fold):
                    idx_k = np.argpartition(drow, k)[:k]
                    pred = Counter(labels_train[idx_k]).most_common(1)[0][0]
                    if pred != labels_test[i_row]:
                        errors += 1
                results[k] += errors
            start += size
        for k in ks:
            results[k] /= n
        return results


## Functions
def dataread():
    # Read the data
    convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
            "mnist_train.csv", 60000)
    convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
            "mnist_test.csv", 10000)
    print('Reading Data ', str(datetime.datetime.now()))
    with open("mnist_test.csv") as fl:
        reader = csv.reader(fl)
        testset = []
        for row in reader:
            testset.append(list(map(int, row)))
    with open("mnist_train.csv") as fl:
        reader = csv.reader(fl)
        trainset = []
        for row in reader:
            trainset.append(list(map(int, row)))
    # Divide the data into labels and pixels
    train_pixels = []
    train_labels = []
    for row in trainset:
        train_pixels.append(row[1:len(row)])
        train_labels.append(row[0:1])
    test_pixels = []
    test_labels = []
    for row in testset:
        test_pixels.append(row[1:len(row)])
        test_labels.append(row[0:1])
    print('Reading Data Finish ', str(datetime.datetime.now()))
    return test_pixels, test_labels, train_pixels, train_labels

def visualize(pixels, labels, title):
    # Visualize sample training data
    image = []
    fig, axs = plt.subplots(5,10)
    fig.subplots_adjust(hspace = 0.5, wspace=0.5)
    fig.suptitle(title)
    axs = axs.ravel()
    for k in range(50):
        image.append ([pixels[k][i:i+28] for i in range(0, len(pixels[k]), 28)])
    image = np.array(image)
    for l in range(50):
        axs[l].imshow(image[l], cmap='gray')
        axs[l].set_title('Label{}={}'.format(l,labels[l]))
    plt.show()

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")
    f.read(16)
    l.read(8)
    images = []
    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

def eucd(train, test):
    # Matrix L2 distance
    dist = (np.sqrt(-2*(np.dot(test, train.T)) + (np.square(train).sum(axis = 1)) + np.matrix((np.square(test).sum(axis = 1))).T))
    return np.array(dist)

def matdistwrite (matdist, csvfile):
    savedist = []
    for row in matdist:
        savedist.append(list(row))
    with open(csvfile, "w") as output:
        writer = csv.writer(output)
        writer.writerows(savedist)

def distread (filename):
    print('Reading dist matrix ', str(datetime.datetime.now()))
    with open(filename) as fl:
        reader = csv.reader(fl)
        dist = []
        for row in reader:
            dist.append(row)
    print('    Training(H) size is: ', len(dist[0]))
    print('    Test(V) size is: ', len(dist))
    print('Reading dist matrix finish ', str(datetime.datetime.now()))
    return dist

def getKNN(dist, k):
    distsort = sorted(enumerate(dist), key=lambda i: i[1])
    index = [j[0] for j in distsort][0:k]
    neighdist = [j[1] for j in distsort][0:k]
    return index, neighdist

def getKNNlabels(index, labels):
    knnlabels = []
    for i in index:
        knnlabels.append(int(labels[i][0]))
    return knnlabels

def getknnresults(labels):
    counter = Counter(labels)
    knnout = counter.most_common(1)[0][0]
    return knnout

def getaccuracy(k, matdist, train_labels, test_labels):
    print('Run for k =', k, str(datetime.datetime.now()))
    results = []
    confmat = np.zeros((10, 10), dtype=int)
    for jj in range (len(test_labels)):
        index, neighdist = getKNN(matdist[jj], k)
        knnlabels = getKNNlabels(index, train_labels)
        # print('number is: ', test_labels[jj][0])
        # print('neighbors are: ', knnlabels)
        knnout = getknnresults(knnlabels)
        if test_labels[jj][0] == knnout:
            flag = 1
            confmat [test_labels[jj][0]][knnout] += 1
        else:
            flag = 0
            confmat[test_labels[jj][0]][knnout] += 1
        # print('knn results is: ', knnout ,'flag is: ', flag)
        results.append(flag)
    # print(results)
    accuracy = 100*sum(results)/len(results)
    print('For k =', k,', Accuracy is:', accuracy, '%', str(datetime.datetime.now()))
    return accuracy, confmat

def getfold(k, train_labels, train_pixels):
    train_labels = np.array(train_labels)
    train_pixels = np.array(train_pixels)
    slice_start = int((k-1)*len(train_labels)/10)
    slice_end = int(k*len(train_labels)/10-1)
    # print('for k = ', k,', slice is:', slice_start, 'to', slice_end)
    test_labels_fold = train_labels [slice_start:slice_end]
    train_labels_fold = np.delete(train_labels, slice(slice_start,slice_end), 0)
    test_pixels_fold =  train_pixels [slice_start:slice_end]
    train_pixels_fold = np.delete(train_pixels, slice(slice_start,slice_end), 0 )
    matdist_fold = eucd(np.array(train_pixels_fold), np.array(test_pixels_fold))
    return matdist_fold, train_labels_fold, test_labels_fold

def extendpic (pixels):
    size = int(math.sqrt(len(pixels)))
    for k in range(size):
        pixels.insert(k*(size+2), 0)
        pixels.insert((k+1)*(size+2)-1, 0)
    ext_pixels = list(np.zeros((size+2), dtype=int))
    ext_pixels.extend(pixels)
    ext_pixels.extend(list(np.zeros((size + 2), dtype=int)))
    return ext_pixels

def extendset(pixels):
    extend_set = []
    for i in range(len(pixels)):
        extend_set.append(extendpic(pixels[i]))
    return extend_set

def croppic (n , pixels):
    size = int(math.sqrt(len(pixels)))
    croppedpic = []
    for i in range(size-2):
        croppedpic = croppedpic + pixels[((n//3)*size)+i*size+(n%3):((n//3)*size)+i*size+(n%3)+(size-2)]
    return croppedpic

def cropset (n, pixels):
    crop_set = []
    for i in range(len(pixels)):
        crop_set.append(croppic(n, pixels[i]))
    return crop_set

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel('Actual Number')
    ax.set_ylabel('Predicted Number')
    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.0f}", textcolors=["black", "white"], threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts

def plotconfmat(confmat):
    Prediction = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    Actual = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    fig, ax = plt.subplots()
    im, cbar = heatmap(confmat, Prediction, Actual, ax=ax, cmap="YlGn", cbarlabel="Occurrence Rate")
    texts = annotate_heatmap(im, valfmt="{x:.0f}")
    fig.tight_layout()
    plt.show()
    return

def slidewin (n, extended_test_pixels, train_pixels):
    print("n =", n)
    crop_set = cropset(n, extended_test_pixels)
    matdist = eucd(np.array(train_pixels), np.array(crop_set))
    return matdist

def getmin(matdist1, matdist2):
    fin_matdist = np.zeros((len(matdist1), len(matdist1[1])), dtype=int)
    for i in range(len(matdist1)):
        for j in range(len(matdist1[0])):
            fin_matdist[i][j] += min(matdist1[i][j], matdist2[i][j] )
    return fin_matdist

def main():
    print('Running... ', str(datetime.datetime.now()))
    # Read the test and training sets and split each into two "pixels" and "labels" sebsets.
    test_pixels, test_labels, train_pixels, train_labels = dataread()
    visualize(train_pixels, train_labels, "Training Set")
    visualize(test_pixels, test_labels, "Test set")

    #######################################################
    ####COMMENT FOLLOWING LINES TO RUN ON FULL DATA SET####
    #######################################################
    ## Get a subset of the data
    train_pixels = train_pixels[0:600]
    train_labels = train_labels[0:600]
    test_pixels = test_pixels[0:100]
    test_labels = test_labels[0:100]
    #######################################################


    ### Calculate and save the L2 distance matrix
    print('Calculating dist matrix ', str(datetime.datetime.now()))
    matdist = eucd(np.array(train_pixels), np.array(test_pixels))
    # matdistwrite(matdist, "matdist600100.csv")
    print('Calculating dist matrix finish ', str(datetime.datetime.now()))
    # Read the L2 distance matrix from .csv file
    # matdist = distread("matdist600100.csv")
    

    ### Run for k = 1 on full data set
    print('Run for k = 1 on full data set')
    getaccuracy(1, matdist, train_labels, test_labels)
    
    
    #### Run for k = 1-10 on kFolds
    accu = []
    print('Running kFolds cross validation', str(datetime.datetime.now()))
    for k in range (1, 11):
        matdist_fold, train_labels_fold, test_labels_fold = getfold (k, train_labels, train_pixels)
        accu.append(getaccuracy(k, matdist_fold, train_labels_fold, test_labels_fold)[0])
    print('Accuracy results are:', accu)
    optk = [j[0] for j in sorted(enumerate(accu), key=lambda i: i[1], reverse=True)][0]+1
    print('Optimum k value is k =', optk, ', with accuracy =', accu[optk-1], '%')


    ### Run for optimum k
    print('Running baseline kNN for k = 3', str(datetime.datetime.now()))
    k = 3
    matdist = eucd(np.array(train_pixels), np.array(test_pixels))
    accuracy, confmat = getaccuracy(k, matdist, train_labels, test_labels)
    print(confmat)
    plotconfmat(confmat)


    ### Run with sliding window
    print('Running sliding window kNN for k = 3', str(datetime.datetime.now()))
    k = 3
    # Extend images to 30*30px
    extended_test_pixels = copy.deepcopy(test_pixels)
    extended_test_pixels = extendset(extended_test_pixels)
    # Get the minimum distance for 9 possible sliding window positions
    fin_matdist = slidewin(0, extended_test_pixels, train_pixels)
    for n in range (1, 9):
        matdist = slidewin(n, extended_test_pixels, train_pixels)
        fin_matdist = getmin(matdist, fin_matdist)
    accuracy, confmat = getaccuracy(k, fin_matdist, train_labels, test_labels)
    print(confmat)
    plotconfmat(confmat)

if __name__ == '__main__':
    main()