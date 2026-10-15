"""Module 3: ML prediction engine.
Linear Regression learns the NORMAL water use PER PERSON in each area of the house
(bathing, washroom, kitchen, laundry, garden, other) for every weekday and month.
Expected usage for a household = predicted per-person usage x number of people."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from generate_dataset import AREA_COLUMNS

INDOOR = ["bathing", "toilet", "kitchen", "laundry", "other"]


class ModelError(Exception):
    """An error with a message that a student can understand."""


def build_features(dates):
    ts = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    table = pd.DataFrame({
        "day_of_week": pd.Categorical(ts.dt.dayofweek, categories=range(7)),
        "month": pd.Categorical(ts.dt.month, categories=range(1, 13)),
    })
    return pd.get_dummies(table, columns=["day_of_week", "month"], dtype=float)


def _robust_sigma(residuals):
    """Typical size of the normal variation (not distorted by a few wastage days)."""
    residuals = np.asarray(residuals)
    return float(max(1.4826 * np.median(np.abs(residuals - np.median(residuals))), 0.3))


def _fit(df):
    """Fits one model for the indoor areas and one for the garden (garden households only)."""
    garden_rows = df["has_garden"] == 1
    main = LinearRegression().fit(build_features(df["date"]), df[[a + "_pp" for a in INDOOR]])
    garden = LinearRegression().fit(build_features(df.loc[garden_rows, "date"]),
                                    df.loc[garden_rows, "garden_pp"])
    sigma = {}
    resid = df[[a + "_pp" for a in INDOOR]].to_numpy() - main.predict(build_features(df["date"]))
    for i, area in enumerate(INDOOR):
        sigma[area] = _robust_sigma(resid[:, i])
    sigma["garden"] = _robust_sigma(df.loc[garden_rows, "garden_pp"].to_numpy()
                                    - garden.predict(build_features(df.loc[garden_rows, "date"])))
    return {"main": main, "garden": garden, "sigma_pp": sigma}


def train_model(df):
    """Trains the final model on all the training data (fast)."""
    if len(df) < 500 or (df["has_garden"] == 1).sum() < 100:
        raise ModelError("Not enough training data to train the model.")
    try:
        return _fit(df)
    except Exception as error:
        raise ModelError(f"The model could not be trained: {error}")


def evaluate_model(df):
    """Honest test: train on the oldest 80% of the days, test on the newest 20%.
    Run it with `python model.py`. Returns the average mistake in litres per day."""
    cutoff = sorted(df["date"].unique())[int(df["date"].nunique() * 0.8)]
    train, test = df[df["date"] < cutoff], df[df["date"] >= cutoff]
    if len(train) < 500 or len(test) < 100 or (train["has_garden"] == 1).sum() < 100:
        raise ModelError("Not enough training data to test the model.")
    tested = _fit(train)
    X = build_features(test["date"])
    people = test["household_size"].to_numpy()
    indoor = tested["main"].predict(X).clip(min=0).sum(axis=1)
    garden = tested["garden"].predict(X).clip(min=0) * test["has_garden"].to_numpy()
    predicted = (indoor + garden) * people
    actual = test[list(AREA_COLUMNS.values())].sum(axis=1).to_numpy()
    baseline = train[[a + "_pp" for a in AREA_COLUMNS.values()]].sum(axis=1).mean() * people
    return {"model_MAE_litres_per_day": mean_absolute_error(actual, predicted),
            "baseline_MAE_litres_per_day": mean_absolute_error(actual, baseline),
            "train_rows": len(train), "test_rows": len(test)}


def predict_expected(model, date, people, has_garden):
    """Expected litres for each area on one day. Returns {display name: litres}."""
    X = build_features([date])
    indoor = model["main"].predict(X).clip(min=0)[0]
    pp = dict(zip(INDOOR, indoor))
    pp["garden"] = float(model["garden"].predict(X).clip(min=0)[0]) if has_garden else 0.0
    return {name: float(pp[col] * people) for name, col in AREA_COLUMNS.items()}


if __name__ == "__main__":
    from preprocessing import load_training_data
    data, _ = load_training_data("data/training_data.csv")
    report = evaluate_model(data)
    for key, value in report.items():
        print(f"{key}: {value:,.2f}" if isinstance(value, float) else f"{key}: {value:,}")