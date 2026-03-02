"""
Compare raw Euclidean k-NN against sliding-window k-NN for MNIST.

Model A (baseline):
    - Raw pixels, Euclidean distance
    - k tuned by 10-fold cross-validation on training data

Model B (paper):
    - Raw pixels, sliding-window L2 distance (±1 pixel shifts)
    - k tuned with 10-fold CV

"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# improved k-NN helper class
from question_d.sliding_window import SlidingWindowKNN

TRAIN_CSV = r"C:\Users\thoma\SLT_Part1_ComputationAssignment\Data\MNIST_train_small.csv"
TEST_CSV  = r"C:\Users\thoma\SLT_Part1_ComputationAssignment\Data\MNIST_test_small.csv"

def standardize(X, mean=None, std=None, eps=1e-8):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + eps
    return (X - mean) / std, mean, std


def fit_lda(X, y, reg=1e-6):
    """Compute LDA projection matrix (d x r) with r = min(C-1, d).
    Returns projection matrix W (d x r) where columns are discriminant directions.
    """
    classes = np.unique(y)
    C = len(classes)
    n, d = X.shape
    mu = X.mean(axis=0)

    Sw = np.zeros((d, d), dtype=float)
    Sb = np.zeros((d, d), dtype=float)
    overall_mean = mu
    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        if nc == 0:
            continue
        mean_c = Xc.mean(axis=0)
        Xc_centered = Xc - mean_c
        Sw += Xc_centered.T @ Xc_centered
        mean_diff = (mean_c - overall_mean)[:, None]
        Sb += nc * (mean_diff @ mean_diff.T)

    # regularize Sw
    Sw += reg * np.eye(d)

    # solve generalized eigenvalue problem Sw^{-1} Sb v = lambda v
    # compute matrix A = pinv(Sw) @ Sb
    try:
        Sw_inv = np.linalg.inv(Sw)
    except np.linalg.LinAlgError:
        Sw_inv = np.linalg.pinv(Sw)
    A = Sw_inv @ Sb
    eigvals, eigvecs = np.linalg.eigh(A)
    # sort eigenvectors by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    r = min(C - 1, d)
    W = eigvecs[:, :r]
    return W, eigvals[:r]


def pca_fit(X, n_components):
    """Compute PCA projection matrix (d x n_components)."""
    mean = X.mean(axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return Vt[:n_components].T


def project(X, P):
    return X @ P


def shift_image(img_2d, dx, dy):
    """Shift a 28x28 image by (dx, dy) pixels using padding and cropping.
    img_2d: shape (28, 28)
    returns: shifted 28x28 image
    """
    # Pad with zeros to 30x30
    padded = np.pad(img_2d, 1, mode='constant', constant_values=0)
    # Extract 28x28 crop from shifted position
    # shift by (dx, dy): pad[1+dx:1+dx+28, 1+dy:1+dy+28]
    return padded[1+dx:1+dx+28, 1+dy:1+dy+28]

if __name__ == '__main__':
    # load data
    train = pd.read_csv(TRAIN_CSV, header=None)
    test = pd.read_csv(TEST_CSV, header=None)
    X_train = train.drop(columns=[0]).values
    y_train = train[0].values
    X_test = test.drop(columns=[0]).values
    y_test = test[0].values
    
    # Reshape images to 28x28 for sliding window (advanced model)
    X_train_img = X_train.reshape(-1, 28, 28)
    X_test_img = X_test.reshape(-1, 28, 28)

    # Baseline: raw Euclidean k-NN with 10-fold CV for k
    print('Baseline: raw Euclidean k-NN, 10-fold CV')
    ks = list(range(1, 15))
    # precompute standard Euclidean distance matrix once
    sq = np.sum(X_train.astype(float)**2, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (X_train.astype(float) @ X_train.astype(float).T)
    dist_sq = np.maximum(dist_sq, 0.0)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    n = X_train.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_sizes = [n // 10] * 10
    for i in range(n % 10):
        fold_sizes[i] += 1
    baseline_cv = {k: 0.0 for k in ks}
    start = 0
    for size in fold_sizes:
        test_idx = indices[start:start+size]
        train_idx = np.setdiff1d(indices, test_idx)
        D_fold = dist[test_idx][:, train_idx]
        labels_train = y_train[train_idx]
        labels_test = y_train[test_idx]
        for k in ks:
            errors = 0
            for i_row, drow in enumerate(D_fold):
                idx_k = np.argpartition(drow, k)[:k]
                pred = Counter(labels_train[idx_k]).most_common(1)[0][0]
                if pred != labels_test[i_row]:
                    errors += 1
            baseline_cv[k] += errors
        start += size
    for k in ks:
        baseline_cv[k] /= n
    baseline_k = min(baseline_cv, key=baseline_cv.get)
    print(f'  selected baseline k = {baseline_k}')
    # baseline test error
    baseline_test_pred = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        idx = np.argpartition(dists, baseline_k)[:baseline_k]
        lab = np.bincount(y_train[idx]).argmax()
        baseline_test_pred.append(lab)
    baseline_test_error = np.mean(np.array(baseline_test_pred) != y_test)
    print(f'Baseline test error = {baseline_test_error:.4f}\n')

    # Sliding window model: only change distance metric from raw pixels
    print('Step 1: Prepare sliding-window k-NN model')
    ks_tune = list(range(1, 15))  # paper uses k up to 10
    sw_model = SlidingWindowKNN()
    sw_model.fit(X_train_img, y_train)
    print('  Performing 10-fold cross-validation (may take a while)...')
    cv_results = sw_model.kfold_cv(ks_tune, n_folds=10, random_state=42)
    best_k = min(cv_results, key=cv_results.get)
    print(f'  selected k = {best_k} with cv error {cv_results[best_k]:.4f}')

    # evaluate on test set
    print('Step 2: Test evaluation using sliding-window k-NN')
    y_pred = sw_model.predict(X_test_img, best_k)
    test_error = np.mean(y_pred != y_test)
    print(f'  Test error = {test_error:.4f}\n')

    # Summary
    print('='*70)
    print('SUMMARY')
    print('='*70)
    print(f'Baseline (raw Euclidean k-NN):')
    print(f'  k={baseline_k}, test_error={baseline_test_error:.4f}')
    print(f'\nAdvanced Model (Sliding Window Distance k-NN):')
    print(f'  k={best_k}, test_error={test_error:.4f}')
    improvement = baseline_test_error - test_error
    print(f'Absolute improvement: {improvement:.4f}')
    if improvement > 0:
        rel_imp = 100 * improvement / baseline_test_error
        print(f'Relative improvement: {rel_imp:.2f}%')
    print('='*70 + '\n')

    baseline_curve = [baseline_cv[k] for k in ks]
    lda_curve = [cv_results[k] for k in ks]

    plt.figure(figsize=(8,5))
    plt.plot(ks, baseline_curve, marker='o', label='Baseline (raw Euclidean k-NN)')
    plt.plot(ks, lda_curve, marker='s', label='Sliding Window k-NN')
    plt.axvline(baseline_k, color='gray', linestyle='--', alpha=0.6)
    plt.axvline(best_k, color='green', linestyle='--', alpha=0.6)
    plt.xlabel('k (neighbors)')
    plt.ylabel('LOOCV error')
    plt.title('Baseline vs Sliding Window k-NN LOOCV curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r"c:\Users\thoma\SLT_Part1_ComputationAssignment\question_d\lda_comparison.png", dpi=150)
    print('Saved plot: question_d/lda_comparison.png')
    plt.show()
