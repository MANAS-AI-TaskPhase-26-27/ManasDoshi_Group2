
from pathlib import Path
import pandas as pd
import numpy as np


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def cost_function(y, logits):
    return np.mean(np.logaddexp(0, logits) - y * logits)


def train_and_predict_cases(train_path, test_path):
    base_path = Path(__file__).resolve().parent
    train_df = pd.read_csv(base_path / train_path)
    test_df = pd.read_csv(base_path / test_path)

    target = 'closed'
    y_train = train_df[target].map({'Yes': 1, 'No': 0}).to_numpy(dtype=float)
    y_test = test_df[target].map({'Yes': 1, 'No': 0}).to_numpy(dtype=float) if target in test_df else None

    if np.isnan(y_train).any():
        raise ValueError('Training labels must be Yes or No')

    
    for df in (train_df, test_df):
        df.drop(columns=['Unnamed: 0', 'Num', target], errors='ignore', inplace=True)
        if 'case_filed' in df.columns:
            dates = pd.to_datetime(df.pop('case_filed'), format='%m-%d-%Y %H:%M', errors='coerce')
            df['year'] = dates.dt.year
            df['month'] = dates.dt.month
            df['day'] = dates.dt.day
            df['hour'] = dates.dt.hour

    categorical = train_df.select_dtypes(include=['object', 'str', 'category']).columns.tolist()
    X_train_df = pd.get_dummies(train_df, columns=categorical, dtype=float)
    X_test_df = pd.get_dummies(test_df, columns=categorical, dtype=float)
    X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)


    X_train = X_train_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)
    X_train[~np.isfinite(X_train)] = np.nan
    X_test[~np.isfinite(X_test)] = np.nan

    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1
    X_train = (X_train - means) / stds
    X_test = (X_test - means) / stds

    X_train = np.column_stack((np.ones(len(X_train)), X_train))
    X_test = np.column_stack((np.ones(len(X_test)), X_test))
    theta = np.zeros(X_train.shape[1])
    learning_rate = 0.1
    iterations = 1500

    for i in range(iterations):
        logits = X_train @ theta
        probabilities = sigmoid(logits)
        gradient = X_train.T @ (probabilities - y_train) / len(y_train)
        theta -= learning_rate * gradient
        if (i + 1) % 300 == 0:
            print(f'Iteration {i + 1}, Training Cost: {cost_function(y_train, X_train @ theta):.6f}')

    test_logits = X_test @ theta
    test_predictions = (sigmoid(test_logits) >= 0.5).astype(int)
    if y_test is not None and np.isfinite(y_test).all():
        print(f'Test Accuracy: {np.mean(test_predictions == y_test) * 100:.2f}%')
        print(f'Test Cost: {cost_function(y_test, test_logits):.6f}')
    print('First 5 Predictions:', test_predictions[:5])
    return test_predictions, theta


if __name__ == '__main__':
    predictions, model_weights = train_and_predict_cases('crime_train.csv', 'crime_test.csv')
