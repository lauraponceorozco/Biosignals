import matplotlib.pyplot as plt
import numpy as np

def fisher_score(X, y):
    """
    Computes Fisher score for each feature.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Class labels (binary: 0 or 1).

    Returns
    -------
    scores : np.ndarray, shape (n_features,)
        Fisher scores for each feature.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Fisher score is defined for binary classification only.")

    X0 = X[y == classes[0]]
    X1 = X[y == classes[1]]
    
    mean0 = np.mean(X0, axis=0)
    mean1 = np.mean(X1, axis=0)
    
    var0 = np.var(X0, axis=0, ddof=1)
    var1 = np.var(X1, axis=0, ddof=1)
    
    numerator = (mean0 - mean1)**2
    denominator = var0 + var1 + 1e-10  # avoid division by zero
    
    return numerator / denominator


def select_top_k_fisher(X, y, k):
    """
    Selects top k features based on Fisher score.

    Returns
    -------
    X_selected : np.ndarray
        X with only top-k features.
    selected_indices : list of int
        Indices of the selected features.
    """
    scores = fisher_score(X, y)
    top_k_idx = np.argsort(scores)[::-1][:k]
    return X[:, top_k_idx], top_k_idx


def plot_fisher_scores(fisher_scores, channel_names, title="Fisher Scores (Target vs Non-target)"):
    """
    Plots Fisher scores for 300 ms and 200 ms features side-by-side for each channel.

    Parameters
    ----------
    fisher_scores : np.ndarray, shape (28,)
        Fisher scores in the order: [AF3_300, F7_300, ..., AF4_300, AF3_200, ..., AF4_200]
    channel_names : list of str, shape (14,)
        Names of EEG channels.
    title : str
        Plot title.
    """
    assert len(fisher_scores) == 2 * len(channel_names), "Expected 28 features (14 channels × 2 timepoints)"

    # Split the scores
    fisher_300 = fisher_scores[:14]
    fisher_200 = fisher_scores[14:]

    x = np.arange(len(channel_names))

    plt.figure(figsize=(12, 5))
    plt.plot(x, fisher_300, marker='o', label="300 ms Features")
    plt.plot(x, fisher_200, marker='s', label="200 ms Features")

    plt.xticks(x, channel_names, rotation=45)
    plt.ylabel("Fisher Score")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("FisherScores_28Features.png", dpi=300)
    plt.show()

def select_top_k_fisher_multiclass(X, y, k):
    """
    Compute Fisher scores for each feature across multiple classes.
    Returns:
        X_selected: (n_samples, k) matrix with selected features
        selected_indices: list of selected feature indices
    """
    n_classes = len(np.unique(y))
    n_samples, n_features = X.shape

    overall_mean = np.mean(X, axis=0)

    # Between-class variance
    SB = np.zeros(n_features)
    SW = np.zeros(n_features)

    for cls in np.unique(y):
        X_c = X[y == cls]
        n_c = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0)
        var_c = np.var(X_c, axis=0) + 1e-6  # Avoid division by zero

        SB += n_c * (mean_c - overall_mean) ** 2
        SW += n_c * var_c

    fisher_score = SB / SW
    selected_indices = np.argsort(fisher_score)[-k:][::-1]  # Top-k features

    X_selected = X[:, selected_indices]
    return X_selected, selected_indices