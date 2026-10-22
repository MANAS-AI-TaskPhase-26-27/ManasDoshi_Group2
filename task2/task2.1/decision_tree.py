from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, matthews_corrcoef, roc_auc_score


def multiclass_auc(y_true, probabilities, labels):
    present = np.unique(y_true)
    if len(present) != len(labels):
        return np.nan
    return roc_auc_score(y_true, probabilities, labels=labels, multi_class='ovr', average='macro')


def main():
    file_path = Path(__file__).resolve().parent / 'yeast.csv'
    data = pd.read_csv(file_path)
    target = 'name'
    X = data.drop(columns=[target]).apply(pd.to_numeric, errors='coerce')
    y = data[target].astype(str)
    if X.isna().any().any() or y.isna().any():
        raise ValueError('Missing or invalid data detected; inspect the dataset before training.')
    if y.value_counts().min() < 5:
        raise ValueError('Every class must have at least five samples for five-fold CV.')

    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
        'recall_macro': 'recall_macro',
        'mcc': make_scorer(matthews_corrcoef),
        'auc_ovr': 'roc_auc_ovr'
    }
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score='raise')
    print('Repeated Stratified 5-Fold CV (3 repeats, 15 evaluations)')
    for metric in scoring:
        values = scores['test_' + metric]
        print(f'{metric:13s}: {values.mean():.4f} +/- {values.std():.4f}')

    one_pass_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, X, y, cv=one_pass_cv)
    print('\nOut-of-fold classification report (one stratified 5-fold pass):')
    print(classification_report(y, predictions, zero_division=0))
    labels = sorted(y.unique())
    print('Confusion matrix (rows=actual, columns=predicted):')
    print(pd.DataFrame(confusion_matrix(y, predictions, labels=labels), index=labels, columns=labels))

    model.fit(X, y)
    print('\nDecision tree (trained on all rows for future predictions):')
    print(export_text(model, feature_names=list(X.columns), max_depth=3))
    pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values(
        'importance', ascending=False
    ).to_csv(file_path.parent / 'yeast_feature_importance.csv', index=False)
    pd.DataFrame({'actual': y, 'out_of_fold_prediction': predictions}).to_csv(
        file_path.parent / 'yeast_cv_predictions.csv', index=False
    )


if __name__ == '__main__':
    main()
