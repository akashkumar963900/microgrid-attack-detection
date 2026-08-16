"""
Microgrid Cyberattack Detection & Mitigation System
====================================================

A machine learning pipeline that detects cyberattacks on smart microgrid
sensor data (voltage, current, frequency, power) and triggers an automated
mitigation response.

Attack types simulated:
    0 - Normal operation
    1 - False Data Injection (FDI) attack
    2 - Denial of Service (DoS) attack
    3 - Replay attack

Pipeline:
    1. Synthetic sensor data generation (Normal + 3 attack types)
    2. Statistical feature extraction
    3. Train / test split
    4. Model training & comparison (Random Forest, SVM, ANN)
    5. Evaluation (accuracy, confusion matrix, classification report, ROC-AUC)
    6. Frequency-domain (FFT) signal analysis
    7. Statistical (3-sigma) anomaly detection
    8. Automated mitigation logic for each detected attack type

Outputs (in ./outputs/):
    voltage_comparison.png, confusion_matrix.png, model_comparison.png,
    feature_importance.png, roc_curve.png, fft_analysis.png,
    anomaly_detection.png

Author: <your name>
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SAMPLES_PER_CLASS = 1000

ATTACK_NAMES = {0: "Normal", 1: "FDI Attack", 2: "DoS Attack", 3: "Replay Attack"}
ATTACK_COLORS = {0: "green", 1: "red", 2: "orange", 3: "purple"}

MITIGATION_ACTIONS = {
    0: "System normal - no action needed",
    1: "FDI detected - isolating affected sensor, switching to backup data source",
    2: "DoS detected - activating islanded mode, redistributing loads",
    3: "Replay detected - resetting communication channel, activating backup controller",
}


# --------------------------------------------------------------------------- #
# 1. Synthetic sensor data generation
# --------------------------------------------------------------------------- #
def generate_sensor_data(n_samples: int = N_SAMPLES_PER_CLASS, seed: int = RANDOM_SEED):
    """Simulate microgrid sensor readings under normal operation and three
    attack scenarios (FDI, DoS, Replay)."""
    rng = np.random.default_rng(seed)

    profiles = {
        # label: (voltage mean/std, current mean/std, frequency mean/std)
        0: ((230, 2), (10, 0.5), (50, 0.1)),      # Normal
        1: ((245, 5), (13, 1), (50, 0.1)),        # FDI - spoofed high voltage/current
        2: ((180, 15), (5, 2), (48, 1)),          # DoS - degraded, noisy readings
        3: ((231, 2), (10.2, 0.5), (50.05, 0.1)), # Replay - near-normal, stale data
    }

    data = {"voltage": [], "current": [], "frequency": [], "power": [], "label": []}
    for label, (v, i, f) in profiles.items():
        voltage = rng.normal(v[0], v[1], n_samples)
        current = rng.normal(i[0], i[1], n_samples)
        frequency = rng.normal(f[0], f[1], n_samples)
        power = voltage * current

        data["voltage"].append(voltage)
        data["current"].append(current)
        data["frequency"].append(frequency)
        data["power"].append(power)
        data["label"].append(np.full(n_samples, label))

    return {k: np.concatenate(v) for k, v in data.items()}, profiles


# --------------------------------------------------------------------------- #
# 2. Feature extraction
# --------------------------------------------------------------------------- #
def extract_features(voltage, current, frequency, power):
    """Build a statistical feature matrix from raw sensor readings."""
    stacked = np.vstack([voltage, current, frequency]).T
    mean = stacked.mean(axis=1)
    std = stacked.std(axis=1)
    var = stacked.var(axis=1)
    v_i_diff = voltage - current
    power_ratio = power / (voltage + 1e-3)

    return np.column_stack(
        [voltage, current, frequency, power, mean, std, var, v_i_diff, power_ratio]
    )


FEATURE_NAMES = [
    "Voltage", "Current", "Frequency", "Power",
    "Mean", "Std Dev", "Variance", "V-I Diff", "Power Ratio",
]


# --------------------------------------------------------------------------- #
# 3. Visualization helpers
# --------------------------------------------------------------------------- #
def plot_voltage_patterns(sensor_data, profiles, n_points=100):
    plt.figure(figsize=(10, 5))
    offset = 0
    for label in profiles:
        seg = sensor_data["voltage"][offset: offset + N_SAMPLES_PER_CLASS]
        plt.plot(seg[:n_points], label=ATTACK_NAMES[label], color=ATTACK_COLORS[label])
        offset += N_SAMPLES_PER_CLASS
    plt.title("Voltage Pattern - Normal vs Attack Conditions", fontsize=14, fontweight="bold")
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/voltage_comparison.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(ATTACK_NAMES.values()),
        yticklabels=list(ATTACK_NAMES.values()),
    )
    plt.title("Confusion Matrix - Attack Detection (Random Forest)", fontsize=14, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()


def plot_model_comparison(accuracies: dict):
    models = list(accuracies.keys())
    values = [accuracies[m] * 100 for m in models]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, values, color=colors, width=0.5)
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)
    for bar, acc in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                  f"{acc:.2f}%", ha="center", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
    plt.close()


def plot_feature_importance(model):
    plt.figure(figsize=(10, 5))
    plt.bar(FEATURE_NAMES, model.feature_importances_, color="steelblue")
    plt.title("Feature Importance - Random Forest", fontsize=14, fontweight="bold")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.close()


def plot_roc_curves(y_test, y_score):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    for label in range(4):
        fpr, tpr, _ = roc_curve(y_test_bin[:, label], y_score[:, label])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=ATTACK_COLORS[label], lw=2,
                 label=f"{ATTACK_NAMES[label]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.title("ROC Curve - Multi-Class Attack Detection", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
    plt.close()


def plot_fft_analysis(sensor_data, profiles, n_points=100):
    def apply_fft(signal):
        fft_vals = np.abs(fft(signal))
        fft_freq = np.fft.fftfreq(len(signal))
        half = len(fft_vals) // 2
        return fft_freq[:half], fft_vals[:half]

    plt.figure(figsize=(12, 5))
    offset = 0
    for label in profiles:
        seg = sensor_data["voltage"][offset: offset + N_SAMPLES_PER_CLASS]
        freq, mag = apply_fft(seg)
        plt.plot(freq[:n_points], mag[:n_points], label=ATTACK_NAMES[label], color=ATTACK_COLORS[label])
        offset += N_SAMPLES_PER_CLASS
    plt.title("FFT Analysis - Frequency Domain of Voltage Signals", fontsize=14, fontweight="bold")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fft_analysis.png", dpi=150)
    plt.close()


def plot_anomaly_detection(sensor_data, threshold=3.0):
    normal_voltage = sensor_data["voltage"][:N_SAMPLES_PER_CLASS]
    mean_v, std_v = normal_voltage.mean(), normal_voltage.std()

    test_voltages = {
        "Normal Sample": 231.5,
        "FDI Sample": 248.0,
        "DoS Sample": 165.0,
        "Replay Sample": 231.8,
    }
    deviations = [abs(v - mean_v) / std_v for v in test_voltages.values()]
    colors = ["green" if d <= threshold else "red" for d in deviations]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(list(test_voltages.keys()), deviations, color=colors)
    plt.axhline(y=threshold, color="black", linestyle="--", linewidth=2,
                label=f"Threshold (3σ = {threshold})")
    plt.title("Anomaly Detection - Sigma Deviation per Sample", fontsize=14, fontweight="bold")
    plt.ylabel("Deviation (σ)")
    plt.legend()
    for bar, dev in zip(bars, deviations):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                  f"{dev:.2f}σ", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/anomaly_detection.png", dpi=150)
    plt.close()
    return mean_v, std_v, test_voltages, deviations


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main():
    print("=" * 60)
    print("MICROGRID CYBERATTACK DETECTION & MITIGATION SYSTEM")
    print("=" * 60)

    # 1. Data generation
    sensor_data, profiles = generate_sensor_data()
    print(f"\nGenerated {len(sensor_data['label'])} samples "
          f"({N_SAMPLES_PER_CLASS} per class x 4 classes)")

    # 2. Feature extraction
    X = extract_features(sensor_data["voltage"], sensor_data["current"],
                          sensor_data["frequency"], sensor_data["power"])
    y = sensor_data["label"]
    print(f"Feature matrix shape: {X.shape}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 4. Random Forest (primary model)
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest accuracy: {rf_accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred, target_names=list(ATTACK_NAMES.values())))

    # 5. SVM + ANN for comparison (need scaled features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training SVM...")
    svm_model = SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)
    svm_model.fit(X_train_scaled, y_train)
    svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test_scaled))
    print(f"SVM accuracy: {svm_accuracy * 100:.2f}%")

    print("Training ANN (MLP)...")
    ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_SEED)
    ann_model.fit(X_train_scaled, y_train)
    ann_accuracy = accuracy_score(y_test, ann_model.predict(X_test_scaled))
    print(f"ANN accuracy: {ann_accuracy * 100:.2f}%")

    accuracies = {"Random Forest": rf_accuracy, "SVM": svm_accuracy, "ANN": ann_accuracy}

    # 6. Visualizations
    print("\nGenerating visualizations...")
    plot_voltage_patterns(sensor_data, profiles)
    plot_confusion_matrix(y_test, rf_pred)
    plot_model_comparison(accuracies)
    plot_feature_importance(rf_model)
    plot_roc_curves(y_test, rf_model.predict_proba(X_test))
    plot_fft_analysis(sensor_data, profiles)
    mean_v, std_v, test_voltages, deviations = plot_anomaly_detection(sensor_data)
    print(f"All plots saved to ./{OUTPUT_DIR}/")

    # 7. Real-time detection + mitigation demo
    print("\n" + "=" * 60)
    print("REAL-TIME DETECTION & MITIGATION DEMO (first 8 test samples)")
    print("=" * 60)
    for i in range(8):
        prediction = int(rf_model.predict([X_test[i]])[0])
        actual = int(y_test[i])
        print(f"\nSample {i + 1}:")
        print(f"  Actual   : {ATTACK_NAMES[actual]}")
        print(f"  Predicted: {ATTACK_NAMES[prediction]}")
        print(f"  Action   : {MITIGATION_ACTIONS[prediction]}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Random Forest : {rf_accuracy * 100:.2f}%")
    print(f"  SVM           : {svm_accuracy * 100:.2f}%")
    print(f"  ANN           : {ann_accuracy * 100:.2f}%")
    print(f"  Anomaly check | mean={mean_v:.2f}V std={std_v:.2f}V")
    print("  7 visualizations generated in ./outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
