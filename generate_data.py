"""
data/generate_data.py
─────────────────────
Synthetic ride-hailing dataset for Careem UAE.
Generates 3,000 records with realistic Dubai demand distributions
and intentional bias patterns for fairness analysis.
"""

import numpy as np
import pandas as pd


def generate_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic ride-hailing records.

    Bias patterns embedded:
      • Higher surge in Tourist-Heavy zones
      • Longer wait times in Residential zones + Low-income riders
      • Higher cancellation when surge > 2.5×
      • Income-linked cancellation sensitivity
    """
    rng = np.random.default_rng(seed)

    # ── Lookup tables ──────────────────────────────────────────────────────
    areas = {
        "Tourist-Heavy": [
            "Dubai Marina", "Downtown Dubai", "Palm Jumeirah",
            "Jumeirah Beach Residence", "Dubai Mall Area",
        ],
        "Business": [
            "DIFC", "Business Bay", "Sheikh Zayed Road",
            "Deira", "Bur Dubai",
        ],
        "Residential": [
            "Mirdif", "Al Quoz", "Jumeirah Village Circle",
            "International City", "Silicon Oasis",
        ],
        "Airport/Transport": [
            "Dubai International Airport", "Al Maktoum Airport",
            "Union Metro Station", "Ibn Battuta Mall",
        ],
    }
    flat_areas = [a for lst in areas.values() for a in lst]
    area_zone  = {a: z for z, lst in areas.items() for a in lst}

    nationalities = [
        "Emirati", "Indian", "Pakistani", "Filipino", "Egyptian",
        "British", "American", "Saudi", "Bangladeshi", "Sri Lankan",
        "Lebanese", "Jordanian", "Nepalese", "Other Arab", "Western European",
    ]
    nat_weights = [
        0.12, 0.22, 0.12, 0.08, 0.07,
        0.05, 0.04, 0.05, 0.06, 0.04,
        0.03, 0.03, 0.03, 0.03, 0.03,
    ]

    vehicle_types  = ["Economy", "Business", "SUV", "Bike", "Luxury"]
    loyalty_levels = ["Bronze", "Silver", "Gold", "Platinum"]
    income_levels  = ["Low", "Middle", "Upper-Middle", "High"]
    weather_list   = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Sandstorm", "Fog"]
    day_list       = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    time_slots     = ["Morning Peak", "Midday", "Afternoon", "Evening Peak", "Late Night"]

    # ── Core demographics ──────────────────────────────────────────────────
    ride_ids     = [f"RD{100000 + i}" for i in range(n)]
    customer_ids = [f"CU{int(rng.integers(10000, 99999))}" for _ in range(n)]

    ages    = rng.integers(18, 67, n).astype(int)
    genders = rng.choice(["Male", "Female", "Prefer Not to Say"],
                          p=[0.55, 0.42, 0.03], size=n)
    nats    = rng.choice(nationalities, p=nat_weights, size=n)
    income  = rng.choice(income_levels,  p=[0.25, 0.35, 0.25, 0.15], size=n)
    loyalty = rng.choice(loyalty_levels, p=[0.40, 0.30, 0.20, 0.10], size=n)

    # ── Locations ──────────────────────────────────────────────────────────
    pickups  = rng.choice(flat_areas, size=n)
    dropoffs = rng.choice(flat_areas, size=n)
    zones    = np.array([area_zone[p] for p in pickups])

    # ── Temporal ───────────────────────────────────────────────────────────
    time_of_day = rng.choice(time_slots, p=[0.25, 0.15, 0.18, 0.28, 0.14], size=n)
    # FIX: use rng.choice with size=n, NOT the plain 7-item day_list
    day_of_week = rng.choice(day_list, size=n)

    # ── Contextual ─────────────────────────────────────────────────────────
    # Events more common in Tourist/Business zones (BIAS seed)
    event_probs = np.where(np.isin(zones, ["Tourist-Heavy", "Business"]), 0.35, 0.12)
    nearby_event = np.array(["Yes" if rng.random() < p else "No" for p in event_probs])

    weather  = rng.choice(weather_list, p=[0.45, 0.20, 0.15, 0.08, 0.07, 0.05], size=n)
    vehicles = rng.choice(vehicle_types, p=[0.45, 0.25, 0.15, 0.08, 0.07], size=n)

    # ── Distance & ride time ───────────────────────────────────────────────
    dist_params = {
        "Morning Peak": (5, 8), "Evening Peak": (6, 9),
        "Midday": (4, 6),       "Afternoon": (5, 7), "Late Night": (7, 12),
    }
    distances = np.array([
        max(1.0, rng.normal(*dist_params[t]))
        for t in time_of_day
    ], dtype=float)
    distances = np.where(vehicles == "Luxury",  distances * 1.3, distances)
    distances = np.where(vehicles == "Business", distances * 1.1, distances)
    distances = np.round(distances, 2)

    speed_factor = rng.uniform(0.8, 1.2, n)
    ride_time    = np.round(distances / speed_factor * 5 + rng.normal(3, 1, n), 0).astype(int)
    ride_time    = np.clip(ride_time, 3, 90)

    # ── Driver metrics ─────────────────────────────────────────────────────
    driver_accept = np.round(rng.uniform(0.55, 0.99, n), 2)
    # Residential areas → lower driver acceptance (BIAS)
    driver_accept = np.where(
        zones == "Residential",
        np.clip(driver_accept - 0.12, 0.40, 0.99),
        driver_accept,
    )
    driver_dist_pickup = np.round(rng.exponential(1.8, n), 2)

    # Wait time: residential + low-income riders wait longer (BIAS)
    wait_base  = rng.normal(6, 3, n)
    wait_base  = np.where(zones == "Residential", wait_base + 3.5, wait_base)
    wait_base  = np.where(income == "Low",         wait_base + 2.0, wait_base)
    wait_times = np.round(np.clip(wait_base, 1, 30), 0).astype(int)

    # ── Fares ──────────────────────────────────────────────────────────────
    base_rate = {"Economy": 1.8, "Business": 2.8, "SUV": 3.2, "Bike": 1.1, "Luxury": 5.5}
    base_fares = np.array([
        round(3.0 + base_rate[v] * d + rng.normal(0, 1.5), 2)
        for v, d in zip(vehicles, distances)
    ])
    base_fares = np.clip(base_fares, 5.0, 250.0)

    # ── Surge multiplier (key bias source) ────────────────────────────────
    surge = np.ones(n)
    surge = np.where(
        np.isin(time_of_day, ["Morning Peak", "Evening Peak"]),
        surge + rng.uniform(0.2, 0.8, n), surge,
    )
    # Tourist zones charged premium surge (BIAS)
    surge = np.where(
        zones == "Tourist-Heavy",
        surge + rng.uniform(0.3, 1.0, n), surge,
    )
    surge = np.where(nearby_event == "Yes",              surge + rng.uniform(0.2, 0.7, n), surge)
    surge = np.where(np.isin(weather, ["Heavy Rain", "Fog"]), surge + rng.uniform(0.3, 0.8, n), surge)
    surge = np.where(time_of_day == "Late Night",        surge + rng.uniform(0.1, 0.5, n), surge)
    surge = np.round(np.clip(surge, 1.0, 4.0), 2)

    final_fares = np.round(base_fares * surge, 2)

    # ── Discounts ──────────────────────────────────────────────────────────
    disc_prob = np.where(
        np.isin(loyalty, ["Gold", "Platinum"]), 0.50,
        np.where(loyalty == "Silver", 0.25, 0.08),
    )
    discount_applied = np.array(["Yes" if rng.random() < p else "No" for p in disc_prob])

    # ── Cancellation probability (BIAS: high surge + low income → cancel) ──
    cancel_prob = np.full(n, 0.12)
    cancel_prob += (surge - 1.0) * 0.12
    cancel_prob += np.where(surge > 2.5, 0.20, 0)          # extreme surge
    cancel_prob += np.where(wait_times > 12, 0.10, 0)
    cancel_prob += np.where(income == "Low",    0.06, 0)
    cancel_prob += np.where(income == "Middle", 0.03, 0)
    cancel_prob += np.where(zones == "Residential", 0.04, 0)
    cancel_prob -= np.where(np.isin(loyalty, ["Gold", "Platinum"]), 0.08, 0)
    cancel_prob  = np.clip(cancel_prob, 0.03, 0.75)
    cancelled    = np.array(["Yes" if rng.random() < p else "No" for p in cancel_prob])

    # ── Customer fairness rating ────────────────────────────────────────────
    fair = 4.5 - (surge - 1.0) * 0.6 - (wait_times / 30) * 0.8
    fair += rng.normal(0, 0.4, n)
    fair  = np.where(discount_applied == "Yes", fair + 0.2, fair)
    fair  = np.round(np.clip(fair, 1.0, 5.0), 1)

    # ── Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        "Ride_ID":                     ride_ids,
        "Customer_ID":                 customer_ids,
        "Customer_Age":                ages,
        "Customer_Gender":             genders,
        "Customer_Nationality":        nats,
        "Customer_Income_Bracket":     income,
        "Customer_Loyalty_Status":     loyalty,
        "Pickup_Location":             pickups,
        "Dropoff_Location":            dropoffs,
        "Pickup_Zone":                 zones,
        "Ride_Time_of_Day":            time_of_day,
        "Ride_Day_of_Week":            day_of_week,        # ← now length-n array
        "Nearby_Event":                nearby_event,
        "Weather_Condition":           weather,
        "Ride_Distance_KM":            distances,
        "Estimated_Ride_Time_Minutes": ride_time,
        "Vehicle_Type_Requested":      vehicles,
        "Driver_Acceptance_Rate":      driver_accept,
        "Driver_Distance_to_Pickup":   driver_dist_pickup,
        "Estimated_Wait_Time":         wait_times,
        "Base_Fare":                   base_fares,
        "Surge_Multiplier":            surge,
        "Final_Fare":                  final_fares,
        "Discount_Applied":            discount_applied,
        "Customer_Fairness_Rating":    fair,
        "Ride_Cancelled":              cancelled,
    })

    # Ordered categoricals
    df["Customer_Income_Bracket"] = pd.Categorical(
        df["Customer_Income_Bracket"],
        categories=["Low", "Middle", "Upper-Middle", "High"],
        ordered=True,
    )
    df["Customer_Loyalty_Status"] = pd.Categorical(
        df["Customer_Loyalty_Status"],
        categories=["Bronze", "Silver", "Gold", "Platinum"],
        ordered=True,
    )

    return df
