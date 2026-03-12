from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

def compute_distance_matrix(X, p=8):
    D = pairwise_distances(X, metric="minkowski", p=p)
    np.fill_diagonal(D, np.inf)
    return D

def loss_function(y_pred, y_true):
    return y_pred != y_true

def knn_predict(X_train, y_train, X_query, k, p, distance_measure, leave_self_out=False):
    predictions = []
    
    for i, x in enumerate(tqdm(X_query, desc=f"k={k}", leave=False)):
        if distance_measure == 1: # Euclidean
            distances = np.linalg.norm(X_train - x, axis=1)
        elif distance_measure == 2: # Minkowski
            distances = np.sum(np.abs(X_train - x) ** p, axis=1) ** (1/p)

        if leave_self_out:
            distances[i] = np.inf  
        
        k_idx = np.argpartition(distances, k - 1)[:k]
        k_neighbors = y_train[k_idx]
        pred = Counter(k_neighbors).most_common(1)[0][0]
        predictions.append(pred)
    
    return np.array(predictions)

def train_k_b(X_train_np, y_train_np, X_query):
    train_results = []

    for k in tqdm(range(1, 20 + 1), desc="LOOCV loop over k"):
        y_train_pred = knn_predict(X_train_np, y_train_np, X_query, k, 1, 1, leave_self_out=True)
        
        train_error = np.mean(y_train_pred != y_train_np)
        
        train_results.append({
            "k": k,
            "train_error": train_error
        })
        
        #print(f"k={k} | train error = {train_error:.4f}")

    train_df = pd.DataFrame(train_results)

    return train_df

def train_k_c(X_train_np, y_train_np, X_query):
    train_results = []

    for p in tqdm(range(1, 15 + 1), desc="LOOCV loop over p"):
        for k in range(1, 20 + 1):
            y_train_pred = knn_predict(X_train_np, y_train_np, X_query, k, p, 2, leave_self_out=True)
            
            train_error = np.mean(y_train_pred != y_train_np)
            
            train_results.append({
                "p": p,
                "k": k,
                "train_error": train_error
            })
            
            #print(f"p={p}, k={k} | train error = {train_error:.4f}")

    train_df = pd.DataFrame(train_results)

    return train_df

def train_k_e(X_train, y_train):
    results = []

    # These were the best k's and p we found
    min_k = 3
    max_k = 6
    p = 8

    # Using a scipy KNN model since it is considerably faster, we want the neirest max k + 1 neighbours
    print("Fitting model")
    nn = NearestNeighbors(n_neighbors= max_k + 1, p = p)
    nn.fit(X_train)

    print("Computing indices")
    distances, indices = nn.kneighbors(X_train)

    # Remove self neighbour
    indices = indices[:,1:]

    for k in tqdm(range(min_k, max_k + 1), desc="Evaluating k"):
        neigh_labels = y_train[indices[:,:k]]

        # Vectorized voting to speed up the progress
        votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=10), 1, neigh_labels)
        preds = votes.argmax(axis=1)

        error = np.mean(preds != y_train)

        results.append({
            "k": k,
            "train_error": error
        })

    return pd.DataFrame(results)

def task_B():
    train_df = train_k_b(X_train_np, y_train_np, X_train_np)

    # Getting the best k
    best_k = train_df.loc[train_df["train_error"].idxmin(), "k"]

    # Apply on test set
    y_test_pred = knn_predict(X_test_np, y_test_np, X_test_np, best_k, 1, 1)
    best_k_error = np.mean(y_test_pred != y_test_np)

    print(f"Test error using best k={best_k}: {best_k_error:.4f}")

    # Plotting the k's
    plt.figure(figsize=(8,5))
    plt.plot(train_df['k'], train_df["train_error"], marker='o', label='Euclidean distance')
    plt.axvline(best_k, color='green', linestyle='--', alpha=0.6)
    plt.xlabel('k')
    plt.ylabel('LOOCV error')
    plt.ylim((0,1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(train_df.to_string())

def task_C():
    train_df = train_k_c(X_train_np, y_train_np, X_train_np)

    # Getting the best p and k
    best_row = train_df.loc[train_df["train_error"].idxmin()]
    best_p = int(best_row["p"])
    best_k = int(best_row["k"])

    # Apply on test set
    y_test_pred = knn_predict(X_test_np, y_test_np, X_test_np, best_k, best_p, 2)
    best_k_p_error = np.mean(y_test_pred != y_test_np)

    print(f"Test error with best ({best_p}, {best_k}): {best_k_p_error:.4f}")

    # Plotting the k's and p's in a heatmap
    pivot = train_df.pivot(index="p", columns="k", values="train_error")
    sns.heatmap(pivot, cmap="viridis")
    plt.xlabel("k")
    plt.ylabel("p")
    plt.show()

    print(train_df.to_string())

def task_E():
    train_df = train_k_e(X_train_np, y_train_np)

    # Getting the best k
    best_k = train_df.loc[train_df["train_error"].idxmin(), "k"]

    # Plotting the k's
    plt.figure(figsize=(8,5))
    plt.plot(train_df['k'], train_df["train_error"], marker='o', label='Minkowski distance with p = 8')
    plt.axvline(best_k, color='green', linestyle='--', alpha=0.6)
    plt.xlabel('k')
    plt.ylabel('LOOCV error')
    plt.ylim((0,1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(train_df.to_string())

def task_F():
    # Best values found
    k = 3
    p = 8

    y_test_pred = knn_predict(X_test_np, y_test_np, X_test_np, k, p, 2, leave_self_out=False)
    test_error = np.mean(y_test_pred != y_test_np)

    print("Test error: ", test_error)


if __name__ == "__main__":
    # test_small = pd.read_csv("Data\MNIST_test_small.csv", header = None)
    # train_small = pd.read_csv("Data\MNIST_train_small.csv", header = None)

    test = pd.read_csv("Data\MNIST_test.csv", header = None)
    train = pd.read_csv("Data\MNIST_train.csv", header = None)

    X_train = train.drop(columns=[0])
    y_train = train[0]

    X_test = test.drop(columns=[0])
    y_test = test[0]

    # Normalizing and vectorizing the data
    X_train_np = X_train.values / 255.0
    y_train_np = y_train.values
    X_test_np = X_test.values / 255.0
    y_test_np = y_test.values


    #print("task B: ")
    #task_B()

    #print("\ntask C: ")
    #task_C()

    # print("\ntask E: ")
    # task_E()

    print("\ntask F: ")
    task_F()
