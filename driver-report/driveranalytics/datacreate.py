import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
DRIVER_SAMPLES = 1200
FEATURE_COLUMNS = [
    "speed_kmh",
    "accel_intensity",
    "brake_intensity",
    "throttle_position",
    "steering_variation",
    "jerk_score",
]


def generate_driver_telematics(sample_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    driving_context = rng.choice(
        ["urban_commute", "highway_cruise", "stop_and_go", "late_braking"],
        size=sample_count,
        p=[0.35, 0.30, 0.20, 0.15],
    )

    context_profiles = {
        "urban_commute": {
            "speed_kmh": (42, 10),
            "accel_intensity": (0.45, 0.15),
            "brake_intensity": (0.40, 0.15),
            "throttle_position": (0.48, 0.12),
            "steering_variation": (12, 4),
        },
        "highway_cruise": {
            "speed_kmh": (105, 12),
            "accel_intensity": (0.28, 0.10),
            "brake_intensity": (0.18, 0.08),
            "throttle_position": (0.55, 0.10),
            "steering_variation": (5, 2),
        },
        "stop_and_go": {
            "speed_kmh": (28, 8),
            "accel_intensity": (0.62, 0.18),
            "brake_intensity": (0.58, 0.18),
            "throttle_position": (0.42, 0.14),
            "steering_variation": (15, 5),
        },
        "late_braking": {
            "speed_kmh": (88, 14),
            "accel_intensity": (0.72, 0.16),
            "brake_intensity": (0.74, 0.14),
            "throttle_position": (0.70, 0.12),
            "steering_variation": (10, 3),
        },
    }

    rows = []
    for context in driving_context:
        profile = context_profiles[context]
        speed = np.clip(rng.normal(*profile["speed_kmh"]), 0, 180)
        acceleration = np.clip(rng.normal(*profile["accel_intensity"]), 0, 1.4)
        braking = np.clip(rng.normal(*profile["brake_intensity"]), 0, 1.4)
        throttle = np.clip(rng.normal(*profile["throttle_position"]), 0, 1.0)
        steering = np.clip(rng.normal(*profile["steering_variation"]), 0, 30)
        jerk = np.clip(
            0.45 * acceleration + 0.45 * braking + 0.10 * (steering / 30),
            0,
            1.5,
        )

        rows.append(
            {
                "driving_context": context,
                "speed_kmh": speed,
                "accel_intensity": acceleration,
                "brake_intensity": braking,
                "throttle_position": throttle,
                "steering_variation": steering,
                "jerk_score": jerk,
            }
        )

    return pd.DataFrame(rows)


def classify_driving_style(row: pd.Series) -> str:
    aggressive_signal = (
        row["speed_kmh"] > 95
        and row["brake_intensity"] > 0.60
        and row["jerk_score"] > 0.65
    )
    eco_signal = (
        row["accel_intensity"] < 0.35
        and row["brake_intensity"] < 0.28
        and row["jerk_score"] < 0.30
    )

    if aggressive_signal or row["driving_context"] == "late_braking":
        return "Aggressive"
    if eco_signal and row["driving_context"] == "highway_cruise":
        return "Eco"
    return "Normal"


def calculate_efficiency_score(row: pd.Series) -> float:
    score = 100.0
    score -= row["accel_intensity"] * 18
    score -= row["brake_intensity"] * 25
    score -= row["jerk_score"] * 20
    score -= max(row["speed_kmh"] - 120, 0) * 0.15
    return round(max(score, 0), 1)


def build_training_dataset(sample_count: int, seed: int) -> pd.DataFrame:
    dataset = generate_driver_telematics(sample_count, seed)
    dataset["driving_style"] = dataset.apply(classify_driving_style, axis=1)
    dataset["efficiency_score"] = dataset.apply(calculate_efficiency_score, axis=1)
    return dataset


def train_style_classifier(dataset: pd.DataFrame):
    features = dataset[FEATURE_COLUMNS]
    target = dataset["driving_style"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=4,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "classification_report": classification_report(
            y_test, test_predictions, output_dict=True
        ),
        "feature_importance": pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False),
    }

    return model, metrics


def build_single_driver_record(
    speed_kmh: int,
    accel_intensity: float,
    brake_intensity: float,
    throttle_position: float,
    steering_variation: int,
) -> pd.DataFrame:
    jerk_score = np.clip(
        0.45 * accel_intensity + 0.45 * brake_intensity + 0.10 * (steering_variation / 30),
        0,
        1.5,
    )
    return pd.DataFrame(
        [
            {
                "speed_kmh": speed_kmh,
                "accel_intensity": accel_intensity,
                "brake_intensity": brake_intensity,
                "throttle_position": throttle_position,
                "steering_variation": steering_variation,
                "jerk_score": jerk_score,
            }
        ]
    )


dataset = build_training_dataset(DRIVER_SAMPLES, RANDOM_SEED)
classifier, model_metrics = train_style_classifier(dataset)

st.set_page_config(page_title="Driver Report", layout="wide")
st.title("Driver Report: Synthetic Telematics Style Analysis")
st.write(
    "This demo simulates vehicle telemetry, labels synthetic driving behaviour with explicit "
    "rules, and trains a classifier to estimate whether a trip looks Eco, Normal, or Aggressive."
)

st.subheader("Why this project is structured this way")
st.write(
    "The dataset is synthetic on purpose. The goal is to show a complete ML workflow with "
    "feature engineering, rule-based labeling, model training, and transparent limitations."
)

left_col, right_col = st.columns(2)
left_col.subheader("Sample telemetry rows")
left_col.dataframe(
    dataset[
        [
            "driving_context",
            "speed_kmh",
            "accel_intensity",
            "brake_intensity",
            "jerk_score",
            "driving_style",
            "efficiency_score",
        ]
    ].head(12)
)

right_col.subheader("Driving style distribution")
right_col.dataframe(
    dataset["driving_style"].value_counts().rename_axis("driving_style").reset_index(name="count")
)

metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
metric_col_1.metric("Train Accuracy", f"{model_metrics['train_accuracy']:.2%}")
metric_col_2.metric("Test Accuracy", f"{model_metrics['test_accuracy']:.2%}")
metric_col_3.metric(
    "Avg Efficiency Score", f"{dataset['efficiency_score'].mean():.1f}"
)

st.subheader("Feature importance")
st.dataframe(model_metrics["feature_importance"], use_container_width=True)

st.subheader("Try a new telemetry sample")
speed_kmh = st.slider("Speed (km/h)", 0, 180, 65)
accel_intensity = st.slider("Acceleration intensity", 0.0, 1.4, 0.45, 0.05)
brake_intensity = st.slider("Brake intensity", 0.0, 1.4, 0.30, 0.05)
throttle_position = st.slider("Throttle position", 0.0, 1.0, 0.50, 0.05)
steering_variation = st.slider("Steering variation", 0, 30, 8)

if st.button("Analyze driving sample"):
    input_df = build_single_driver_record(
        speed_kmh=speed_kmh,
        accel_intensity=accel_intensity,
        brake_intensity=brake_intensity,
        throttle_position=throttle_position,
        steering_variation=steering_variation,
    )
    predicted_style = classifier.predict(input_df[FEATURE_COLUMNS])[0]
    efficiency_score = calculate_efficiency_score(input_df.iloc[0])

    result_col_1, result_col_2 = st.columns(2)
    result_col_1.subheader("Input features")
    result_col_1.dataframe(input_df, use_container_width=True)
    result_col_2.subheader("Model output")
    result_col_2.write(f"Predicted driving style: **{predicted_style}**")
    result_col_2.write(f"Estimated efficiency score: **{efficiency_score} / 100**")

st.subheader("Known limitations")
st.write(
    "- Labels come from handcrafted rules, so the model learns a simplified view of driving behaviour.\n"
    "- Real telematics systems would use time-series trip data, not one-row summary samples.\n"
    "- The current version focuses on explainability over realism."
)
