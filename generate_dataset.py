"""Creates SYNTHETIC (fake) training data: daily water use of many households, split by area.
The app learns "normal water use per person" for each area from this file.
Replace it with real data later (same columns)."""
from pathlib import Path

import numpy as np
import pandas as pd

# Display name -> column name in the data file (the washroom's column is still called "toilet")
AREA_COLUMNS = {"Bathing": "bathing", "Washroom": "toilet", "Kitchen": "kitchen",
                "Laundry": "laundry", "Garden and outdoor": "garden", "Other": "other"}

# Typical litres used per person per day in each area
PER_PERSON = {"bathing": 45, "toilet": 30, "kitchen": 25, "laundry": 20, "garden": 15, "other": 10}


def make_training_data(households=24, days=730, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    weekend = np.asarray(dates.dayofweek >= 5)
    summer = np.asarray(dates.month.isin([3, 4, 5, 6]))
    monsoon = np.asarray(dates.month.isin([7, 8, 9]))
    winter = np.asarray(dates.month.isin([11, 12, 1, 2]))

    factor = {
        "bathing": np.where(weekend, 1.05, 1.0) * np.where(summer, 1.25, np.where(winter, 0.9, 1.0)),
        "toilet": np.ones(days),
        "kitchen": np.where(weekend, 1.1, 1.0) * np.where(summer, 1.05, 1.0),
        "laundry": np.where(weekend, 1.3, 1.0) * np.where(monsoon, 1.1, 1.0),
        "garden": np.where(weekend, 1.3, 1.0) * np.where(summer, 1.4, np.where(monsoon, 0.4, 0.8)),
        "other": np.ones(days),
    }
    frames = []
    for _ in range(households):
        size = int(rng.integers(2, 7))
        garden = bool(rng.random() < 0.5)
        style = rng.normal(1, 0.08)              # some households simply use a bit more or less
        table = {"date": dates, "household_size": size, "has_garden": int(garden)}
        for area, per_person in PER_PERSON.items():
            use = per_person * size * style * factor[area] * rng.normal(1, 0.08, days)
            table[area] = np.zeros(days) if (area == "garden" and not garden) else use
        data = pd.DataFrame(table)
        # A few days with wastage: a leaking washroom tank, a tap left running, overwatering...
        for day in rng.choice(days, size=12, replace=False):
            area = rng.choice(["toilet", "kitchen", "bathing", "other"] + (["garden"] if garden else []))
            data.loc[day, area] += rng.uniform(30, 90)
        frames.append(data)
    out = pd.concat(frames, ignore_index=True)
    for area in PER_PERSON:
        out[area] = out[area].clip(lower=0).round(1)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


if __name__ == "__main__":
    path = Path(__file__).parent / "data" / "training_data.csv"
    path.parent.mkdir(exist_ok=True)
    make_training_data().to_csv(path, index=False)
    print(f"Synthetic training data saved to {path}")