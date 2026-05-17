import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from tqdm import tqdm

from joblib import Parallel, delayed
import copy


def get_train_val_test_split(df_ml, target_city, target_col='temperature_2m', val_days=30, test_days=14):
    print(f"\n--- {target_city} ---")
    df_city = df_ml[df_ml['city'] == target_city].copy()

    # Количество часов для отрезания с конца
    test_size = test_days * 24
    val_size = val_days * 24

    train_size = len(df_city) - test_size - val_size

    if train_size <= 0:
        print(f"ОШИБКА: На Train не остается данных после выделения Val и Test: {train_size}")
        return [pd.DataFrame()] * 6

    train_df = df_city.iloc[:train_size]
    val_df = df_city.iloc[train_size:train_size + val_size]
    test_df = df_city.iloc[train_size + val_size:]

    drop_cols = [target_col, 'city']

    X_train, y_train = train_df.drop(columns=drop_cols), train_df[target_col]
    X_val, y_val = val_df.drop(columns=drop_cols), val_df[target_col]
    X_test, y_test = test_df.drop(columns=drop_cols), test_df[target_col]

    print(f"Train: {X_train.shape[0]} часов, Val: {X_val.shape[0]} часов, Test: {X_test.shape[0]} часов")

    return X_train, y_train, X_val, y_val, X_test, y_test


def select_best_base_model_recursive(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
            'loss': 'squared_error',
            'random_state': 42
        }

        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    print("Optuna", end="")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    best_hgb_params = study.best_params
    best_hgb_params['random_state'] = 42

    print(f": MAE на Val: {study.best_value:.3f} -> Сравнение моделей:")

    boosting_name = 'Boosting'

    models = {
        'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        boosting_name: HistGradientBoostingRegressor(**best_hgb_params)
    }

    best_model = None
    best_mae = float('inf')
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        print(f"{name:<25} MAE = {mae:.3f} °C")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

    print(f"–––– Выбран:  {best_name}!\n")

    return best_model


def forecast_direct(X_train, y_train, X_test, base_model, horizon=168):
    print(f"\n[Direct] Обучение {horizon} моделей")

    ar_cols = [c for c in X_train.columns if 'temp_lag' in c]
    time_cols = ['hour', 'day_of_year', 'month', ]
    exo_cols = [c for c in X_train.columns if c not in ar_cols and c not in time_cols]

    def train_single_step(step):
        y_train_step = y_train.shift(-step + 1)
        X_train_step = X_train.copy()
        X_train_step[exo_cols + time_cols] = X_train_step[exo_cols + time_cols].shift(-step + 1)

        valid_idx = y_train_step.dropna().index.intersection(X_train_step.dropna().index)
        y_step = y_train_step.loc[valid_idx]
        X_step = X_train_step.loc[valid_idx]

        model_step = copy.deepcopy(base_model)

        original_depth = getattr(model_step, 'max_depth', 10)
        if original_depth is None: original_depth = 10

        # Динамическая смена глубины дерева – чем больше шагов, тем меньше
        if step <= 12:
            pass
        elif step <= 48:
            model_step.max_depth = max(4, original_depth - 3)
            model_step.min_samples_leaf = max(30, getattr(model_step, 'min_samples_leaf', 20))
        else:
            model_step.max_depth = 3
            model_step.min_samples_leaf = 50

        model_step.fit(X_step, y_step)
        return step, model_step

    trained_models_list = Parallel(n_jobs=-1)(
        delayed(train_single_step)(step) for step in tqdm(range(1, horizon + 1))
    )

    direct_models = dict(trained_models_list)

    direct_preds = []
    start_point_ar = X_test.iloc[[0]][ar_cols].copy().reset_index(drop=True)

    for step in range(1, horizon + 1):
        future_step_exo_time = X_test.iloc[[step - 1]][exo_cols + time_cols].copy().reset_index(drop=True)
        current_features = pd.concat([start_point_ar, future_step_exo_time], axis=1)
        current_features = current_features[X_train.columns]

        pred = direct_models[step].predict(current_features)[0]
        direct_preds.append(pred)

    # Сглаживание
    preds_series = pd.Series(direct_preds)
    smoothed_preds = preds_series.ewm(alpha=0.5, adjust=False).mean().tolist()

    return smoothed_preds


def extract_features_from_history(history_series, date_index):
    """Аналог create_features, но для рекурсии"""
    features = {'hour': date_index.hour, 'day_of_year': date_index.dayofyear, 'month': date_index.month}

    h_list = [1, 2, 3, 24, 48, 168]
    for h in h_list:
        features[f'temp_lag_{h}h'] = history_series[-h]

    return pd.DataFrame([features])


def forecast_recursive(X_train, y_train, X_val, y_val, X_test, base_model, horizon=168):
    print(f"\n[Recursive] Обучение 1 модели")

    model = base_model
    model.fit(X_train, y_train)

    history = list(y_val.iloc[-168:].values)

    recursive_preds = []

    X_test_horizon = X_test.iloc[:horizon]

    ar_cols = [c for c in X_train.columns if 'temp_lag' in c]

    for i in tqdm(range(len(X_test_horizon))):
        current_features_row = X_test_horizon.iloc[[i]].copy()

        # Сначала всё NaN
        current_features_row[ar_cols] = np.nan

        temp_history_series = pd.Series(history)

        for h in [1, 2, 3, 24, 48, 168]:
            if f'temp_lag_{h}h' in current_features_row.columns:
                current_features_row.loc[current_features_row.index, f'temp_lag_{h}h'] = temp_history_series.iloc[-h]

        current_features_row = current_features_row[X_train.columns]
        pred = model.predict(current_features_row)[0]
        recursive_preds.append(pred)
        history.append(pred)
        history.pop(0)

    return recursive_preds


from sklearn.metrics import mean_absolute_error, r2_score


def calculate_advanced_metrics(y_true, y_pred, y_history_last_step):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)


    mae = mean_absolute_error(y_true, y_pred)

    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6) * 100

    true_diff = np.diff(np.insert(y_true, 0, y_history_last_step))
    pred_diff = np.diff(np.insert(y_pred, 0, y_history_last_step))
    da = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100

    dir_r2 = r2_score(true_diff, pred_diff)

    return {"MAE": mae, "MAPE (%)": mape, "WAPE (%)": wape, "DA (%)": da, "Dir_R2": dir_r2}

def select_best_base_model_for_direct(X_train, y_train, X_val, y_val):
    REPRESENTATIVE_HORIZON = 24


    y_train_step = y_train.shift(-REPRESENTATIVE_HORIZON + 1)
    X_train_step = X_train.shift(-REPRESENTATIVE_HORIZON + 1)
    y_val_step = y_val.shift(-REPRESENTATIVE_HORIZON + 1)
    X_val_step = X_val.shift(-REPRESENTATIVE_HORIZON + 1)

    train_idx = y_train_step.dropna().index
    X_train_step, y_train_step = X_train_step.loc[train_idx], y_train_step.loc[train_idx]
    val_idx = y_val_step.dropna().index
    X_val_step, y_val_step = X_val_step.loc[val_idx], y_val_step.loc[val_idx]

    def objective_direct(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 45),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 15, 50),
            'loss': 'squared_error',
            'random_state': 42
        }
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train_step, y_train_step)
        preds = model.predict(X_val_step)
        return mean_absolute_error(y_val_step, preds)

    print("Optuna (Direct, 24h)", end="")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_direct, n_trials=25)

    print(f": MAE на Val: {study.best_value:.3f} -> Сравнений не будет, берём бустинг")

    best_hgb_params = study.best_params
    best_hgb_params['random_state'] = 42

    return HistGradientBoostingRegressor(**best_hgb_params)
