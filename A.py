from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter

test_small = pd.read_csv("SLT_Part1_ComputationAssignment\Data\MNIST_test_small.csv", header = None)
train_small = pd.read_csv("SLT_Part1_ComputationAssignment\Data\MNIST_train_small.csv", header = None)

X_train = train_small.drop(columns=[0])
y_train = train_small[0]

X_test = test_small.drop(columns=[0])
y_test = test_small[0]

X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def loss_function(y_pred, y_true):
    return y_pred != y_true

def knn_predict(X_train, y_train, X_query, k, leave_self_out=False):
    predictions = []
    
    for i, x in enumerate(tqdm(X_query, desc=f"k={k}", leave=False)):
        distances = np.linalg.norm(X_train - x, axis=1)
        
        if leave_self_out:
            distances[i] = np.inf  
        
        k_idx = np.argpartition(distances, k)[:k]
        k_neighbors = y_train[k_idx]
        pred = Counter(k_neighbors).most_common(1)[0][0]
        predictions.append(pred)
    
    return np.array(predictions)

results = []

for k in tqdm(range(1, 20+1), desc="Loop over k"):
    y_test_pred = knn_predict(X_train_np, y_train_np, X_test_np, k)
    test_error = np.mean(y_test_pred != y_test_np)
    
    y_train_pred = knn_predict(X_train_np, y_train_np, X_train_np, k, leave_self_out=True)
    train_error = np.mean(y_train_pred != y_train_np)

    results.append({
        "k": k,
        "train_error": train_error,
        "test_error": test_error
    })
    print(f"k={k} | Train error = {train_error:.4f} | Test error = {test_error:.4f}")
    
results_df = pd.DataFrame(results)

print(results_df)
    


    