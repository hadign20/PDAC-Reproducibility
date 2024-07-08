import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_curve)
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_classifiers():
    """
    Returns a dictionary of classifiers to be evaluated.
    """
    return {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'LogisticRegression': LogisticRegression(),
        'NaiveBayes': GaussianNB(),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    }


def compute_metrics(y_true, y_pred, y_pred_prob):
    """
    Compute evaluation metrics and their confidence intervals.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    y_pred_prob (array-like): Predicted probabilities.

    Returns:
    dict: Evaluation metrics and their confidence intervals.
    """

    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_prob) if y_pred_prob is not None else None
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    sensitivity = recall_score(y_true, y_pred)
    ppv = precision_score(y_true, y_pred)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1
    }

    ci = {}
    for metric, value in metrics.items():
        if value is not None:
            ci[metric] = compute_confidence_interval(value, y_true.size)

    return metrics, ci


def compute_confidence_interval(metric, n, alpha=0.95):
    """
    Compute confidence interval for a metric.

    Parameters:
    metric (float): Metric value.
    n (int): Sample size.
    alpha (float): Confidence level.

    Returns:
    tuple: Lower and upper bounds of the confidence interval.
    """
    se = np.sqrt((metric * (1 - metric)) / n)
    h = se * stats.norm.ppf((1 + alpha) / 2)
    return metric - h, metric + h


def train_test_split_evaluation(X, y, test_size=0.2, random_state=42):
    """
    Perform train/test split and evaluate models.

    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target vector.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    dict: Results for each classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    classifiers = get_classifiers()
    results = {}
    train_results = {}
    test_results = {}

    for classifier_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        y_pred_prob = clf.predict_proba(X_train)[:, 1] if hasattr(clf, "predict_proba") else None
        metrics, ci = compute_metrics(y_train, y_pred, y_pred_prob)
        train_results[classifier_name] = {
            'metrics': metrics,
            'confidence_intervals': ci
        }

    for classifier_name, clf in classifiers.items():
        clf.fit(X_train, y_train)

        # train set
        y_pred_train = clf.predict(X_train)
        y_pred_prob_train = clf.predict_proba(X_train)[:, 1] if hasattr(clf, "predict_proba") else None
        metrics_train, ci_train = compute_metrics(y_train, y_pred_train, y_pred_prob_train)

        # test set
        y_pred_test = clf.predict(X_test)
        y_pred_prob_test = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        metrics_test, ci_test = compute_metrics(y_test, y_pred_test, y_pred_prob_test)

        results[classifier_name] = {
            'train': {'metrics': metrics_train, 'confidence_intervals': ci_train},
            'test': {'metrics': metrics_test, 'confidence_intervals': ci_test}
        }

    return results


def cross_validation_evaluation(X, y, cv=10):
    """
    Perform cross-validation and evaluate models.

    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target vector.
    cv (int): Number of cross-validation folds.

    Returns:
    dict: Results for each classifier.
    """
    classifiers = get_classifiers()
    results = {}

    for name, clf in classifiers.items():
        skf = StratifiedKFold(n_splits=cv)
        metrics_list = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_prob = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else None
            metrics, _ = compute_metrics(y_test, y_pred, y_pred_prob)
            metrics_list.append(metrics)

        averaged_metrics = {metric: np.mean([m[metric] for m in metrics_list if m[metric] is not None]) for metric in metrics_list[0]}
        ci = {metric: compute_confidence_interval(averaged_metrics[metric], y.size) for metric in averaged_metrics}

        results[name] = {
            'metrics': averaged_metrics,
            'confidence_intervals': ci
        }
    return results



def hyperparameter_tuning(X, y, model, param_grid, cv=5):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='roc_auc')
    grid_search.fit(X, y)
    return grid_search.best_estimator_



def evaluate_models(X, y, method='train_test_split', hyperparam_tune=False, **kwargs):
    """
    Evaluate models using the specified method.

    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target vector.
    method (str): Evaluation method ('train_test_split' or 'cross_validation').
    kwargs: Additional arguments for the evaluation methods.

    Returns:
    dict: Evaluation results for each classifier.
    """
    classifiers = get_classifiers()
    results = {}

    if method == 'train_test_split':
        X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)
    else:
        skf = StratifiedKFold(n_splits=kwargs.get('cv', 5))

    for name, clf in classifiers.items():
        if hyperparam_tune:
            param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}  # Example grid
            clf = hyperparameter_tuning(X_train, y_train, clf, param_grid) if method == 'train_test_split' else clf

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        metrics, ci = compute_metrics(y_test, y_pred, y_pred_prob)
        results[name] = {
            'metrics': metrics,
            'confidence_intervals': ci
        }
    return results


    # if method == 'train_test_split':
    #     return train_test_split_evaluation(X, y, **kwargs)
    # elif method == 'cross_validation':
    #     return cross_validation_evaluation(X, y, **kwargs)
    # else:
    #     raise ValueError("Invalid method. Choose 'train_test_split' or 'cross_validation'.")


def save_classification_results(results, output_file, method='train_test_split'):
    """
    Save evaluation results to an Excel file.

    Parameters:
    results (dict): Evaluation results for each classifier.
    output_file (str): Path to save the Excel file.
    """
    if method == 'train_test_split':
        rows = []
        for dataset, classification_results in results.items():
            for classifier, data in classification_results.items():
                metrics = data['metrics']
                ci = data['confidence_intervals']
                row = [
                    dataset.capitalize(),
                    classifier,
                    f"{metrics['roc_auc']:.2f} ({ci['roc_auc'][0]:.2f}, {ci['roc_auc'][1]:.2f})" if 'roc_auc' in metrics else 'N/A',
                    f"{metrics['sensitivity']:.2f} ({ci['sensitivity'][0]:.2f}, {ci['sensitivity'][1]:.2f})",
                    f"{metrics['specificity']:.2f} ({ci['specificity'][0]:.2f}, {ci['specificity'][1]:.2f})",
                    f"{metrics['ppv']:.2f} ({ci['ppv'][0]:.2f}, {ci['ppv'][1]:.2f})",
                    f"{metrics['npv']:.2f} ({ci['npv'][0]:.2f}, {ci['npv'][1]:.2f})",
                ]
                rows.append(row)

        df = pd.DataFrame(rows, columns=['Dataset', 'Classifier', 'AUC (95% CI)', 'Sensitivity (95% CI)', 'Specificity (95% CI)',
                                         'PPV (95% CI)', 'NPV (95% CI)'])
        df.to_excel(output_file, index=False)

    elif method == 'cross_validation':
        rows = []
        for classifier, data in results.items():
            metrics = data['metrics']
            ci = data['confidence_intervals']
            row = [
                classifier,
                f"{metrics['roc_auc']:.2f} ({ci['roc_auc'][0]:.2f}, {ci['roc_auc'][1]:.2f})" if 'roc_auc' in metrics else 'N/A',
                f"{metrics['sensitivity']:.2f} ({ci['sensitivity'][0]:.2f}, {ci['sensitivity'][1]:.2f})",
                f"{metrics['specificity']:.2f} ({ci['specificity'][0]:.2f}, {ci['specificity'][1]:.2f})",
                f"{metrics['ppv']:.2f} ({ci['ppv'][0]:.2f}, {ci['ppv'][1]:.2f})",
                f"{metrics['npv']:.2f} ({ci['npv'][0]:.2f}, {ci['npv'][1]:.2f})",
            ]
            rows.append(row)

        df = pd.DataFrame(rows, columns=['Classifier', 'AUC (95% CI)', 'Sensitivity (95% CI)', 'Specificity (95% CI)',
                                         'PPV (95% CI)', 'NPV (95% CI)'])
        df.to_excel(output_file, index=False)

    else:
        raise ValueError("Invalid method. Choose 'train_test_split' or 'cross_validation'.")