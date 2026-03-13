from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

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

    for k in tqdm(range(1, 10 + 1), desc="LOOCV loop over k"):
        y_train_pred = knn_predict(X_train_np, y_train_np, X_query, k, 1, 1, leave_self_out=True)
        
        train_error = np.mean(y_train_pred != y_train_np)
        
        train_results.append({
            "k": k,
            "train_error": train_error
        })
        
        #print(f"k={k} | train error = {train_error:.4f}")

    train_df = pd.DataFrame(train_results)

    return train_df



# %%
def plot_first_pcs(pca_model, n_pcs_available, title_prefix=""):
    n_to_plot = min(16, n_pcs_available)
    pcs_images = pca_model.components_[:n_to_plot].reshape((n_to_plot, 28, 28))
    
    grid_size = 4  # fixed 4x4 grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8,8))
    axes = axes.flatten()
    
    for i in range(16):
        axes[i].axis('off')  # turn off all axes by default
    
    for i in range(n_to_plot):
        axes[i].imshow(pcs_images[i], cmap='bwr')
        axes[i].axis('on')
        axes[i].set_title(f"PC {i+1}", fontsize=8)
    
    plt.suptitle(f"{title_prefix} First {n_to_plot} PCs")
    plt.tight_layout()
    plt.show()
    
def pca_knn_optimal_components(X_train_df, y_train_np, X_test_df, y_test_np, candidate_pcs=[5,10,20,40,75,100], visualize_pcs=True):
    """
    Find optimal number of PCA components based on LOOCV on training set.
    Returns PCA model with selected components, best k, and test error.
    """
    X_train_orig = X_train_df.values / 255.0
    X_test_orig = X_test_df.values / 255.0
    
    best_train_error = np.inf
    best_pca = None
    best_k = None
    best_X_train_pca = None
    best_X_test_pca = None
    
    for n_pc in candidate_pcs:
        print(f"\nEvaluating {n_pc} principal components...")
        pca = PCA(n_components=n_pc, random_state=42)
        X_train_pca = pca.fit_transform(X_train_orig)
        X_test_pca = pca.transform(X_test_orig)
        
        
        plot_first_pcs(pca, n_pc, title_prefix=f"{n_pc} PCs")

        # Override global arrays for your existing train_k_b
        global X_train_np, X_test_np
        X_train_np = X_train_pca
        X_test_np = X_test_pca
        
        # LOOCV to pick best k
        train_df = train_k_b(X_train_np, y_train_np, X_train_np)
        k_candidate = train_df.loc[train_df["train_error"].idxmin(), "k"]
        train_error = train_df["train_error"].min()
        
        print(f" -> Best k={k_candidate}, LOOCV error={train_error:.4f}")
        
        if train_error < best_train_error:
            best_train_error = train_error
            best_k = k_candidate
            best_pca = pca
            best_X_train_pca = X_train_pca
            best_X_test_pca = X_test_pca
    
    print(f"\nSelected {best_X_train_pca.shape[1]} PCs with best LOOCV error {best_train_error:.4f} (k={best_k})")
    
    # Optional: visualize first 16 PCs
    if visualize_pcs:
        n_pcs = best_X_train_pca.shape[1]  # total PCs we actually have
        n_to_plot = min(16, n_pcs)         # plot up to 16 or the number of PCs we have
        pcs_images = best_pca.components_[:n_to_plot].reshape((n_to_plot, 28, 28))
        
        # Determine grid size: ceil(sqrt(n_to_plot)) for rows/cols
        import math
        grid_size = math.ceil(math.sqrt(n_to_plot))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
        axes = axes.flatten()
        
        for i in range(n_to_plot):
            axes[i].imshow(pcs_images[i], cmap='bwr')
            axes[i].axis('off')
            axes[i].set_title(f"PC {i+1}")
        
        # Turn off any extra axes if grid is bigger than n_to_plot
        for i in range(n_to_plot, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"First {n_to_plot} PCs (selected {n_pcs} PCs)")
        plt.tight_layout()
        plt.show()
    
    # Evaluate test error with best k
    y_test_pred = knn_predict(best_X_train_pca, y_train_np, best_X_test_pca, best_k, p=1, distance_measure=1)
    test_error = np.mean(y_test_pred != y_test_np)
    print(f"Test error on PCA-reduced data: {test_error:.4f}")
    
    return best_pca, best_k, test_error, best_train_error


# %%

test = pd.read_csv("SLT_Part1_ComputationAssignment\Data\MNIST_test.csv", header = None)
train = pd.read_csv("SLT_Part1_ComputationAssignment\Data\MNIST_train.csv", header = None)

test = test.iloc[:4000,:]
train = train.iloc[:10000,:]
X_train = train.drop(columns=[0])
y_train = train[0]

X_test = test.drop(columns=[0])
y_test = test[0]


X_train_np = X_train.values / 255.0
X_test_np = X_test.values / 255.0

# Step 1: Compute test error WITHOUT PCA for k = 1..10
test_errors_full = []
for k in tqdm(range(1, 11) , desc="without pca"):
    y_test_pred = knn_predict(X_train_np, y_train, X_test_np, k, p=1, distance_measure=1)
    error = np.mean(y_test_pred != y_test)
    test_errors_full.append(error)
    print(f"Full data: k={k}, test error={error:.4f}")

# Step 2: Apply PCA with 40 components

best_pca, _, _, _ = pca_knn_optimal_components(
    X_train_df=X_train,
    y_train_np=y_train,   # corrected variable name
    X_test_df=X_test,
    y_test_np=y_test,     # corrected variable name
    candidate_pcs=[10,20,40,60,75,100],
    visualize_pcs=True    # optional: visualize first 16 PCs
)

n_pcs = best_pca.n_components_
pca = PCA(n_components=n_pcs, random_state=42)
X_train_pca = pca.fit_transform(X_train_np)
X_test_pca = pca.transform(X_test_np)

# Step 3: Compute test error WITH PCA for k = 1..10
test_errors_pca = []
for k in tqdm(range(1, 11) , desc="with pca"):
    y_test_pred = knn_predict(X_train_pca, y_train, X_test_pca, k, p=1, distance_measure=1)
    error = np.mean(y_test_pred != y_test)
    test_errors_pca.append(error)
    print(f"PCA ({n_pcs} PCs): k={k}, test error={error:.4f}")

# Step 4: Plot comparison
plt.figure(figsize=(8,5))
plt.plot(range(1,11), test_errors_full, 'o-', label='Full 784-dim data')
plt.plot(range(1,11), test_errors_pca, 's--', label=f'PCA {n_pcs} PCs')
plt.xlabel('k')
plt.ylabel('Test error')
plt.title(f'k-NN test error: PCA vs no PCA (fixed {n_pcs} PCs)')
plt.xticks(range(1,11))
plt.grid(True)
plt.legend()
plt.show()


plot_first_pcs(best_pca, 16)
