"""Module 4 (part 1): finds WHERE water is used more than normal and where wastage may be happening.

Method (a documented threshold): for every area, extra = actual - expected litres.
  - Possible wastage : extra is more than 3 typical variations (per person) AND more than 10 litres.
  - Higher than usual: extra is more than 1.5 typical variations (per person) AND more than 5 litres.
The 'typical variation' (sigma) is measured from the training data."""
import pandas as pd

from generate_dataset import AREA_COLUMNS

ADVICE = {
    "Bathing": "Keep showers to about 5 minutes, close the tap while soaping, and fix any leaking shower head.",
    "Washroom": "Check for a running or leaking flush tank and dripping washroom taps (put food colour in the tank and see if it reaches the bowl). Fix them quickly.",
    "Kitchen": "Wash vegetables and dishes in a basin, not under a running tap, and fix any dripping tap.",
    "Laundry": "Run the washing machine only with full loads and use the water-saving setting.",
    "Garden and outdoor": "Water early morning or evening, use a watering can or drip pipe, and do not leave the hose running.",
    "Other": "Use a bucket instead of a hose to wash floors and vehicles, and check outdoor taps and pipes.",
}


def analyse_day(expected, actual, people, sigma_pp):
    """expected and actual are {area name: litres}. Returns one row per area."""
    rows = []
    for area, column in AREA_COLUMNS.items():
        exp, act = expected[area], actual[area]
        extra = act - exp
        extra_pp = extra / people
        sigma = sigma_pp[column]
        if extra_pp > 3 * sigma and extra > 10:
            status = "Possible wastage"
        elif extra_pp > 1.5 * sigma and extra > 5:
            status = "Higher than usual"
        else:
            status = "Normal"
        severity = ("High" if act >= 2 * exp else "Medium") if status == "Possible wastage" else ""
        rows.append({"Area": area, "Actual (L)": act, "Expected (L)": exp, "Extra (L)": extra,
                     "Compared with expected": (extra / exp * 100) if exp > 0 else 0.0,
                     "Status": status, "Severity": severity})
    return pd.DataFrame(rows)