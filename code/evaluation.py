import json
import pathlib
import pickle
import tarfile

import joblib
import numpay as np
import pandas as pd
import xgboost

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


if __name__ == '__main__':
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    model = pickle.load(open("xgboost-model", "rb"))

    train_path = "/opt/ml/processing/train/train.csv"
    test_path = "/opt/ml/processing/test/test.csv"

    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)

    X_train = df_train.drop(['ALLSKY_SFC_SW_DWN_shifted'], axis=1)
    y_train = df_train['ALLSKY_SFC_SW_DWN_shifted']
    X_test = df_test.drop(['ALLSKY_SFC_SW_DWN_shifted'], axis=1)
    y_test = df_test['ALLSKY_SFC_SW_DWN_shifted']

    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, predictions_train)
    mse_train = mean_squared_error(y_train, predictions_train)
    rmse_train = np.sqrt(mse_train)
    msle_train = mean_squared_log_error(y_train, predictions_train)
    rmsle_train = np.sqrt(msle_train)
    r2_score_train = r2_score(y_train, predictions_train)

    mae_test = mean_absolute_error(y_test, predictions_test)
    mse_test = mean_squared_error(y_test, predictions_test)
    rmse_test = np.sqrt(mse_test)
    msle_test = mean_squared_log_error(y_test, predictions_test)
    rmsle_test = np.sqrt(msle_test)
    r2_score_test = r2_score(y_test, predictions_test)

    mae_diff = mae_train - mae_test
    mse_diff = mse_train - mse_test
    rmse_diff = rmse_train - rmse_test
    msle_diff = msle_train - msle_test
    rmsle_diff = rmsle_train - rmsle_test
    r2_diff = r2_score_train - r2_score_test

    report_dict = {
        "regression_metrics":{
            "mae_train": mae_train,
            "mse_train": mse_train,
            "rmse_train": rmse_train,
            "msle_train": msle_train,
            "rmsle_train": rmsle_train,
            "r2_score_train": r2_score_train,
            "mae_test": mae_test,
            "mse_test": mse_test,
            "rmse_test": rmse_test,
            "msle_test": msle_test,
            "rmsle_test": rmsle_test,
            "r2_score_test": r2_score_test,
            "mae_diff": mae_diff,
            "mse_diff": mse_diff,
            "rmse_diff": rmse_diff,
            "msle_diff": msle_diff,
            "rmsle_diff": rmsle_diff,
            "r2_diff": r2_diff
        }
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))