"""Generate sample_data/insurance_churn.csv: 10,000 synthetic insurance
customers, and sample_data/insurance_churn_recent.csv: 5,000 newer ones whose
data has drifted (for Drift analysis).

One file, three tasks:
- Classification: `Churned` (Yes/No), whether the customer left at renewal.
- Regression: `Annual_Premium`, or `Customer_Lifetime_Value`.
- Clustering: the behaviour columns (tenure, premium, claims, engagement,
  service calls) fall into four customer segments that are not in the file.

The recent file drifts on purpose: more young digital customers, larger
price rises, more of the book in the West, a shift from auto-debit to card,
higher engagement, more missing credit scores — and one new relationship:
late payments now rise with the price rise (customers under strain).

Every value is made up; the generator is seeded, so the file is the same on
every run:

    python tools/make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parent.parent / "sample_data"

# Hidden segments: share, tenure (years), engagement, claims rate, calls rate
SEGMENTS = {
    "loyal_family":    (0.30, 9.0, 55, 0.35, 1.0),
    "young_digital":   (0.28, 2.0, 82, 0.20, 0.8),
    "price_sensitive": (0.25, 3.0, 35, 0.30, 2.8),
    "high_claims":     (0.17, 5.5, 45, 1.60, 3.5),
}


RECENT_SHARES = {"loyal_family": 0.22, "young_digital": 0.40,
                 "price_sensitive": 0.26, "high_claims": 0.12}


def generate(N, seed, drift=False):
    """The customers as a frame. With `drift`, the shifts described above;
    without it, the draws are exactly those of the original file."""
    rng = np.random.default_rng(seed)
    names = list(SEGMENTS)
    shares = [RECENT_SHARES[s] if drift else SEGMENTS[s][0] for s in names]
    seg = rng.choice(names, N, p=shares)
    p = pd.DataFrame([SEGMENTS[s][1:] for s in seg],
                     columns=["tenure", "engage", "claims", "calls"])

    age = np.where(seg == "young_digital", rng.normal(29, 5, N),
                   np.where(seg == "loyal_family", rng.normal(46, 9, N),
                            rng.normal(41, 12, N)))
    age = np.clip(age, 18, 85).round().astype(int)
    tenure = np.clip(rng.gamma(2.0, p["tenure"] / 2.0), 0, 40).round(1)
    tenure = np.minimum(tenure, age - 17)

    region = rng.choice(["North", "South", "East", "West", "Central"], N,
                        p=[0.18, 0.20, 0.17, 0.33, 0.12] if drift else
                        [0.22, 0.24, 0.18, 0.21, 0.15])
    gender = rng.choice(["Female", "Male"], N)
    married = np.where(seg == "loyal_family", rng.random(N) < 0.85,
                       rng.random(N) < 0.40)
    marital = np.where(married, "Married",
                       rng.choice(["Single", "Divorced", "Widowed"], N,
                                  p=[0.75, 0.18, 0.07]))
    dependents = np.where(seg == "loyal_family", rng.poisson(1.8, N),
                          rng.poisson(0.5, N))

    policy = rng.choice(["Auto", "Home", "Health", "Life"], N,
                        p=[0.38, 0.27, 0.22, 0.13])
    coverage = rng.choice(["Basic", "Standard", "Premium"], N,
                          p=[0.35, 0.45, 0.20])
    policies_held = 1 + rng.poisson(np.where(seg == "loyal_family", 1.4, 0.4))
    channel = np.where(seg == "young_digital",
                       rng.choice(["Online", "Agent", "Broker"], N,
                                  p=[0.75, 0.15, 0.10]),
                       rng.choice(["Online", "Agent", "Broker"], N,
                                  p=[0.30, 0.50, 0.20]))
    payment = np.where(seg == "price_sensitive",
                       rng.choice(["Auto-debit", "Card", "Bank transfer",
                                   "Cheque"], N, p=[0.25, 0.35, 0.25, 0.15]),
                       rng.choice(["Auto-debit", "Card", "Bank transfer",
                                   "Cheque"], N,
                                  p=[0.38, 0.42, 0.15, 0.05] if drift else
                                  [0.55, 0.25, 0.15, 0.05]))
    frequency = rng.choice(["Monthly", "Quarterly", "Annual"], N,
                           p=[0.55, 0.20, 0.25])

    claims = rng.poisson(p["claims"] * 1.5)
    claim_amount = np.where(
        claims > 0,
        claims * rng.lognormal(np.log(np.select(
            [policy == "Auto", policy == "Home", policy == "Health"],
            [2_800, 6_500, 3_900], 9_000)), 0.6), 0.0).round(2)
    satisfaction = np.where(claims > 0,
                            np.clip(np.round(rng.normal(
                                3.6 - 0.25 * (claims - 1), 1.0)), 1, 5),
                            np.nan)
    calls = rng.poisson(p["calls"])
    late = rng.poisson(np.where(seg == "price_sensitive", 1.6, 0.4))
    engagement = np.clip(rng.normal(p["engage"] + (8 if drift else 0), 12),
                         0, 100).round()
    credit = np.clip(rng.normal(np.where(seg == "price_sensitive", 610, 690),
                                70), 300, 850).round()

    base = np.select([policy == "Auto", policy == "Home", policy == "Health"],
                     [1_100, 950, 2_400], 700)
    mult = pd.Series(coverage).map({"Basic": 0.8, "Standard": 1.0,
                                    "Premium": 1.45}).to_numpy()
    age_load = 1 + 0.012 * np.maximum(age - 30, 0) \
        + np.where(policy == "Auto", 0.35 * (age < 25), 0)
    region_load = pd.Series(region).map({"North": 1.00, "South": 1.08,
                                         "East": 1.12, "West": 1.15,
                                         "Central": 0.95}).to_numpy()
    premium = (base * mult * age_load * region_load
               * (1 + 0.09 * claims) * (1 - 0.004 * np.minimum(tenure, 20))
               * rng.lognormal(0, 0.12, N)).round(2)
    discount = np.where((policies_held > 1) | (tenure > 5),
                        rng.random(N) < 0.70, rng.random(N) < 0.15)
    premium = np.where(discount, premium * 0.92, premium).round(2)
    increase = np.clip(rng.normal(np.where(claims > 1, 9, 4)
                                  + (5 if drift else 0), 4), -5, 30
                       ).round(1)
    if drift:
        # The new relationship: the steeper the rise, the more late payments.
        late = late + rng.poisson(np.clip(increase - 5, 0, None) * 0.15)

    # Churn at renewal: price rises, friction and few ties push customers out.
    logit = (-1.8
             + 0.11 * increase
             + 0.30 * calls
             + 0.35 * late
             - 0.13 * np.minimum(tenure, 15)
             - 0.45 * (policies_held - 1)
             - 0.60 * (payment == "Auto-debit")
             + 0.40 * (frequency == "Monthly")
             - 0.55 * discount
             - 0.012 * (engagement - 50)
             + np.where(np.isnan(satisfaction), 0,
                        0.45 * (3 - np.nan_to_num(satisfaction)))
             + 0.25 * (seg == "price_sensitive"))
    churned = rng.random(N) < 1 / (1 + np.exp(-logit))

    # Estimated at the start of the year, before anyone knows who will
    # leave: built from churn, it would hand a churn model the answer.
    clv = (premium * (2.5 + 0.35 * policies_held + 0.12 * np.minimum(tenure, 20)
                      + 0.02 * (engagement - 50).clip(-30, 30))
           * rng.lognormal(0, 0.10, N) - 0.35 * claim_amount).round(2)

    df = pd.DataFrame({
        "Customer_ID": [f"C{100000 + i}" for i in range(N)],
        "Age": age,
        "Gender": gender,
        "Marital_Status": marital,
        "Dependents": dependents,
        "Region": region,
        "Policy_Type": policy,
        "Coverage_Level": coverage,
        "Policies_Held": policies_held,
        "Sales_Channel": channel,
        "Payment_Method": payment,
        "Payment_Frequency": frequency,
        "Tenure_Years": tenure,
        "Annual_Premium": premium,
        "Premium_Change_Pct": increase,
        "Discount_Applied": np.where(discount, "Yes", "No"),
        "Claims_Last_3Y": claims,
        "Total_Claim_Amount": claim_amount,
        "Claim_Satisfaction": satisfaction,
        "Late_Payments_12M": late,
        "Service_Calls_12M": calls,
        "Digital_Engagement": engagement,
        "Credit_Score": credit,
        "Customer_Lifetime_Value": clv,
        "Churned": np.where(churned, "Yes", "No"),
    })
    # Realistic gaps: credit checks not on file, engagement not tracked.
    df.loc[rng.random(N) < (0.12 if drift else 0.04), "Credit_Score"] = np.nan
    df.loc[rng.random(N) < 0.02, "Digital_Engagement"] = np.nan
    return df


def main():
    DIR.mkdir(exist_ok=True)
    for name, df in (("insurance_churn.csv", generate(10_000, 7)),
                     ("insurance_churn_recent.csv",
                      generate(5_000, 11, drift=True))):
        if name != "insurance_churn.csv":
            # Newer customers: their IDs follow on from the first file's.
            df["Customer_ID"] = [f"C{200000 + i}" for i in range(len(df))]
        df.to_csv(DIR / name, index=False)
        print(f"{DIR / name}: {len(df):,} rows, churn rate "
              f"{(df['Churned'] == 'Yes').mean():.1%}")


if __name__ == "__main__":
    main()
