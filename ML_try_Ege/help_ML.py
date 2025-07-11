import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from help_fischer import *

def run_lda_svm_fisher_cv_bestC(X, y, feature_ks=[3, 5, 7, 9], C_values=[0.1, 1, 10], n_splits=5, random_state=42):
    results = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for k in feature_ks:
        X_k, top_k_idx = select_top_k_fisher(X, y, k)

        lda_scores = []
        svm_scores_dict = {C: [] for C in C_values}

        for train_idx, test_idx in skf.split(X_k, y):
            X_train, X_test = X_k[train_idx], X_k[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # LDA
            lda = LDA(priors=[np.mean(y == 0), np.mean(y == 1)])
            lda.fit(X_train, y_train)
            lda_pred = lda.predict(X_test)
            lda_scores.append(accuracy_score(y_test, lda_pred))

            # SVM for all C
            for C in C_values:
                svm = SVC(C=C, kernel='linear', class_weight='balanced', random_state=random_state)
                svm.fit(X_train, y_train)
                svm_pred = svm.predict(X_test)
                svm_scores_dict[C].append(accuracy_score(y_test, svm_pred))

        # Determine best C based on average accuracy
        best_C = max(C_values, key=lambda c: np.mean(svm_scores_dict[c]))

        results.append({
            'Top_K': k,
            'LDA_Accuracy_Mean': np.mean(lda_scores),
            'LDA_Accuracy_Std': np.std(lda_scores),
            'SVM_Best_C': best_C,
            'SVM_Accuracy_Mean': np.mean(svm_scores_dict[best_C]),
            'SVM_Accuracy_Std': np.std(svm_scores_dict[best_C]),
        })

    df_results = pd.DataFrame(results)
    return df_results
