import csv
import numpy as np
from esn import EchoStateNetwork

CSV_PATH = "traffic_counts.csv"
MODEL_OUT = "esn_model.npz"
NORM_OUT = "norm_params.npz"

# Predict "congestion soon" after this many samples (interval=1s => 5 sec)
HORIZON_STEPS = 5

# IMPORTANT: your dataset is small, keep washout small
WASHOUT = 5

def load_csv(path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    ns = np.array([float(x["ns"]) for x in rows], dtype=np.float32)
    ew = np.array([float(x["ew"]) for x in rows], dtype=np.float32)
    total = np.array([float(x["total"]) for x in rows], dtype=np.float32)
    return ns, ew, total

def make_features(ns, ew, total):
    # features: [ns, ew, total, rate, ma3]
    rate = np.concatenate([[0.0], np.diff(total)])
    ma3 = np.convolve(total, np.ones(3)/3, mode="same")
    X = np.stack([ns, ew, total, rate, ma3], axis=1)

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd
    return Xn.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)

def make_labels(total, threshold):
    y = np.zeros_like(total, dtype=np.float32)
    for t in range(len(total) - HORIZON_STEPS):
        y[t] = 1.0 if total[t + HORIZON_STEPS] >= threshold else 0.0
    return y.reshape(-1, 1)

def main():
    ns, ew, total = load_csv(CSV_PATH)
    if len(total) < 40:
        print(f"[WARN] Only {len(total)} samples. For better learning, collect 60–120 sec data.")

    # AUTO threshold: top 25% counts => "congested"
    threshold = float(np.percentile(total, 75))
    # safety clamp so it's not too low
    threshold = max(threshold, float(np.mean(total) + 0.5*np.std(total)))

    X, mu, sd = make_features(ns, ew, total)
    Y = make_labels(total, threshold)

    # cut last horizon
    cut = len(total) - HORIZON_STEPS
    X = X[:cut]
    Y = Y[:cut]

    pos_rate = float(Y.mean()) if len(Y) else 0.0
    print(f"Auto THRESHOLD = {threshold:.2f}")
    print(f"Positive label rate = {pos_rate*100:.1f}% (should NOT be 0%)")

    if pos_rate == 0.0 or pos_rate == 1.0:
        print("[ERROR] Labels are all same. Collect more data or change threshold logic.")
        return

    esn = EchoStateNetwork(
        input_dim=X.shape[1],
        reservoir_size=200,
        spectral_radius=0.9,
        sparsity=0.10,
        leak_rate=0.30,
        ridge_alpha=1e-2,
        seed=42,
    )

    esn.fit(X, Y, washout=WASHOUT)

    # Save
    esn.save(MODEL_OUT)
    np.savez(NORM_OUT, mu=mu, sd=sd)

    # sanity check
    esn.reset_state()
    probs = []
    for t in range(X.shape[0]):
        probs.append(esn.predict_proba(X[t])[0])
    probs = np.array(probs)

    print("✅ Trained ESN successfully.")
    print(f"Saved: {MODEL_OUT}, {NORM_OUT}")
    print(f"Prob range: min={probs.min():.3f}, max={probs.max():.3f}, mean={probs.mean():.3f}")

if __name__ == "__main__":
    main()