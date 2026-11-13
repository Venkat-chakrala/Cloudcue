import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from xgboost import XGBRegressor, XGBClassifier

from utils.features import FEATURE_COLUMNS


DATA_PATH = Path("datasets/processed_weather_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run preprocessing.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    return df


def get_feature_importances(model, feature_names):
    """Safely extracts feature importances from tree-based models or linear coefficients."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "named_steps"):
        last_step = list(model.named_steps.values())[-1]
        if hasattr(last_step, "coef_"):
            coef = last_step.coef_
            return np.abs(coef.flatten())
    return None


def train_regression_models(X_train, X_test, y_train, y_test, target_name):
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.06,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    print(f"\n========== {target_name.upper()} PREDICTION ==========")

    for name, model in models.items():
        print(f"Training {name} for {target_name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = r2_score(y_test, predictions)

        results.append({
            "Target": target_name,
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
            "F1": None
        })

        trained_models[name] = (model, rmse)
        print(f"  {name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    best_result = min(results, key=lambda x: x["RMSE"])
    best_name = best_result["Model"]
    best_model, best_rmse = trained_models[best_name]

    joblib.dump(best_model, MODEL_DIR / f"best_{target_name}_model.pkl")
    print(f"Saved best model: {best_name} (RMSE: {best_rmse:.4f})")

    # Extract feature importance for the best model (or XGBoost if pipeline)
    importance_model = best_model
    if not hasattr(importance_model, "feature_importances_") and "XGBoost" in trained_models:
        importance_model = trained_models["XGBoost"][0]

    fi_scores = get_feature_importances(importance_model, list(X_train.columns))

    return results, best_name, best_rmse, fi_scores


def train_rainfall_models(X_train, X_test, y_train, y_test, target_suffix=""):
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain both rain and no-rain examples.")

    scale_pos_weight = negative_count / max(positive_count, 1)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.06,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    target_label = f"Rainfall{target_suffix}"
    print(f"\n========== {target_label.upper()} PREDICTION ==========")

    for name, model in models.items():
        print(f"Training {name} for {target_label}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, zero_division=0)
        rec = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        results.append({
            "Target": target_label,
            "Model": name,
            "MAE": None,
            "RMSE": None,
            "R2": None,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

        trained_models[name] = model
        print(f"  {name} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    best_result = max(results, key=lambda x: x["F1"])
    best_name = best_result["Model"]
    best_model = trained_models[best_name]

    filename = f"best_rainfall{target_suffix}_model.pkl"
    joblib.dump(best_model, MODEL_DIR / filename)
    print(f"Saved best model: {best_name} -> {filename}")

    # Feature importance
    importance_model = best_model
    if not hasattr(importance_model, "feature_importances_") and "Random Forest" in trained_models:
        importance_model = trained_models["Random Forest"]

    fi_scores = get_feature_importances(importance_model, list(X_train.columns))

    return results, best_name, best_result["F1"], fi_scores


def main():
    df = load_data()

    X = df[FEATURE_COLUMNS]

    # Chronological split: 80% train, 20% test
    split_index = int(len(df) * 0.80)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Number of features: {len(FEATURE_COLUMNS)}")

    all_results = []
    metadata = {
        "features": FEATURE_COLUMNS,
        "models": {},
        "confidence_intervals": {}
    }

    feature_importances_dict = {}

    # 1. Temperature (1h)
    temp_results, temp_best, temp_rmse, temp_fi = train_regression_models(
        X_train, X_test,
        df["target_temperature"].iloc[:split_index],
        df["target_temperature"].iloc[split_index:],
        "temperature"
    )
    all_results.extend(temp_results)
    metadata["models"]["temperature"] = temp_best
    metadata["confidence_intervals"]["temperature_rmse"] = temp_rmse
    if temp_fi is not None:
        feature_importances_dict["temperature"] = temp_fi

    # 2. Humidity (1h)
    hum_results, hum_best, hum_rmse, hum_fi = train_regression_models(
        X_train, X_test,
        df["target_humidity"].iloc[:split_index],
        df["target_humidity"].iloc[split_index:],
        "humidity"
    )
    all_results.extend(hum_results)
    metadata["models"]["humidity"] = hum_best
    metadata["confidence_intervals"]["humidity_rmse"] = hum_rmse
    if hum_fi is not None:
        feature_importances_dict["humidity"] = hum_fi

    # 3. Rainfall (1h)
    rain_results, rain_best, rain_f1, rain_fi = train_rainfall_models(
        X_train, X_test,
        df["target_rainfall"].iloc[:split_index],
        df["target_rainfall"].iloc[split_index:],
        ""
    )
    all_results.extend(rain_results)
    metadata["models"]["rainfall"] = rain_best
    if rain_fi is not None:
        feature_importances_dict["rainfall"] = rain_fi

    # 4. Multi-Horizon: 3-Hour & 6-Hour Predictions
    print("\n========== MULTI-HORIZON EXPANSION (3H & 6H) ==========")
    t3_results, t3_best, t3_rmse, _ = train_regression_models(
        X_train, X_test,
        df["target_temperature_3h"].iloc[:split_index],
        df["target_temperature_3h"].iloc[split_index:],
        "temperature_3h"
    )
    all_results.extend(t3_results)

    r3_results, r3_best, r3_f1, _ = train_rainfall_models(
        X_train, X_test,
        df["target_rainfall_3h"].iloc[:split_index],
        df["target_rainfall_3h"].iloc[split_index:],
        "_3h"
    )
    all_results.extend(r3_results)

    t6_results, t6_best, t6_rmse, _ = train_regression_models(
        X_train, X_test,
        df["target_temperature_6h"].iloc[:split_index],
        df["target_temperature_6h"].iloc[split_index:],
        "temperature_6h"
    )
    all_results.extend(t6_results)

    r6_results, r6_best, r6_f1, _ = train_rainfall_models(
        X_train, X_test,
        df["target_rainfall_6h"].iloc[:split_index],
        df["target_rainfall_6h"].iloc[split_index:],
        "_6h"
    )
    all_results.extend(r6_results)

    # 5. Export Feature Importance CSV
    if feature_importances_dict:
        fi_df = pd.DataFrame({"Feature": FEATURE_COLUMNS})
        for target, scores in feature_importances_dict.items():
            # Normalize to percentage
            norm_scores = (scores / np.sum(scores)) * 100 if np.sum(scores) > 0 else scores
            fi_df[f"{target.capitalize()}_Importance_%"] = norm_scores.round(2)

        fi_df = fi_df.sort_values(by=f"{list(feature_importances_dict.keys())[0].capitalize()}_Importance_%", ascending=False)
        fi_df.to_csv("datasets/feature_importance.csv", index=False)
        print("\nSaved feature importances to datasets/feature_importance.csv")

    # 6. Save Model Evaluation & Metadata
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("datasets/model_evaluation.csv", index=False)

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n==============================================")
    print("ALL MODELS AND HORIZONS TRAINED SUCCESSFULLY!")
    print("==============================================")
    print("Evaluation summary saved to datasets/model_evaluation.csv")
    print("Metadata and confidence intervals saved to models/model_metadata.json")


if __name__ == "__main__":
    main()
