from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

def loss_function(y_pred, y_true):
    return y_pred != y_true

def knn_predict(X_train, y_train, X_query, k, p, distance_measure, leave_self_out=False):
    predictions = []
    
    for i, x in enumerate(tqdm(X_query, desc=f"k={k}", leave=False)):
        if distance_measure == 1:
            distances = np.linalg.norm(X_train - x, axis=1)
        elif distance_measure == 2:
            distances = np.array([minkowski_distance(x, x_train, p)
                                  
    for x_train in X_train
])

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

def task_B():
    train_df = train_k_b(X_train_np, y_train_np, X_train_np)

    # Getting the best k
    best_k = train_df.loc[train_df["train_error"].idxmin(), "k"]

    # Apply on test set
    y_test_pred = knn_predict(X_test_np, y_test_np, X_test_np, best_k, 1, 1)
    best_k_error = np.mean(y_test_pred != y_test_np)

    print(f"Test error using best k={best_k}: {best_k_error:.4f}")

    # Plotting the k's
    plt.figure()
    plt.plot(train_df['k'], train_df['train_error'])
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()

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


if __name__ == "__main__":
    test_small = pd.read_csv("Data\MNIST_test_small.csv", header = None)
    train_small = pd.read_csv("Data\MNIST_train_small.csv", header = None)

    X_train = train_small.drop(columns=[0])
    y_train = train_small[0]

    X_test = test_small.drop(columns=[0])
    y_test = test_small[0]
    
    # X_train = X_train.loc[:25]
    # y_train = y_train.loc[:25]
    # X_test = X_test.loc[:25]
    # y_test = y_test.loc[:25]

    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values
    y_test_np = y_test.values


    print("task B: ")
    #task_B()

    print("\ntask C: ")
    task_C()