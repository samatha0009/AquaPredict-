"""Module 2: Data preprocessing. Cleans the TRAINING data the model learns from.
(Values typed into the form are checked by the form itself, so they need no cleaning.)"""
import pandas as pd

AREAS = ["bathing", "toilet", "kitchen", "laundry", "garden", "other"]
REQUIRED = ["date", "household_size", "has_garden"] + AREAS


class DataError(Exception):
    """An error with a message that a student can understand."""


def load_training_data(source):
    """Reads and cleans the training data. Returns (table, notes)."""
    try:
        df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    except Exception as error:
        raise DataError(f"The training data could not be read: {error}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"The training data is missing these columns: {', '.join(missing)}")

    notes = []
    start = len(df)
    df = df.drop_duplicates()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df.isna().any(axis=1) | (df["household_size"] <= 0) | (df[AREAS] < 0).any(axis=1)
    if bad.sum():
        notes.append(f"Removed {int(bad.sum())} rows with missing, unreadable or invalid values.")
    df = df[~bad]
    if len(df) < 1000:
        raise DataError("The training data is too small. At least 1000 rows are needed.")
    if start - len(df) > 0.2 * start:
        notes.insert(0, "Warning: more than 20% of the training rows were removed.")

    df = df.sort_values("date").reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    for area in AREAS:
        df[area + "_pp"] = df[area] / df["household_size"]      # litres per person
    return df, notes