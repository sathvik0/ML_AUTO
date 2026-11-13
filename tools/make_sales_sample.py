"""Generate sample_data/store_sales.csv: two years of daily sales for eight
stores, for forecasting.

Each store's sales follow a trend, a weekly pattern (busy weekends), a
yearly one (a December peak), promotions that lift a day's sales, public
holidays that close some stores, and noise that carries over from one day
to the next. Tomorrow's sales are best predicted from recent sales, the
day of the week and whether a promotion runs: what lag features and a
time split are for.

Every value is made up; the generator is seeded:

    python tools/make_sales_sample.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parent.parent / "sample_data"
STORES = {"S01": ("North", 1.00), "S02": ("North", 0.80),
          "S03": ("South", 1.35), "S04": ("South", 0.65),
          "S05": ("East", 1.10), "S06": ("East", 0.90),
          "S07": ("West", 1.50), "S08": ("West", 0.75)}
WEEKDAY = np.array([0.85, 0.80, 0.85, 0.95, 1.10, 1.35, 1.25])  # Mon..Sun


def generate(seed=21, start="2024-01-01", days=730):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    holidays = {d for d in dates if (d.month, d.day) in (
        (1, 1), (5, 1), (8, 15), (12, 25), (12, 26))}
    rows = []
    for store, (region, size) in STORES.items():
        level = 1_000 * size
        noise = 0.0
        promo_days = set(rng.choice(days, size=days // 9, replace=False))
        for i, d in enumerate(dates):
            trend = 1 + 0.18 * i / days
            yearly = 1 + 0.12 * np.sin(2 * np.pi * (d.day_of_year - 80) / 365) \
                + (0.35 if d.month == 12 and d.day >= 10 else 0)
            promo = i in promo_days
            noise = 0.6 * noise + rng.normal(0, 0.06)
            sales = level * trend * yearly * WEEKDAY[d.day_of_week] \
                * (1.30 if promo else 1.0) * (1 + noise)
            closed = d in holidays and rng.random() < 0.6
            rows.append({"Date": d.strftime("%Y-%m-%d"), "Store": store,
                         "Region": region, "Promotion": "Yes" if promo
                         else "No", "Holiday": "Yes" if d in holidays
                         else "No",
                         "Sales": 0.0 if closed else round(max(sales, 0), 2)})
    return pd.DataFrame(rows)


def main():
    DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(DIR / "store_sales.csv", index=False)
    print(f"{DIR / 'store_sales.csv'}: {len(df):,} rows, "
          f"{df['Store'].nunique()} stores, {df['Date'].min()} to "
          f"{df['Date'].max()}")


if __name__ == "__main__":
    main()
