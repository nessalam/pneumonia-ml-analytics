import os
import json
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks


# -----------------------
# Config
# -----------------------
DATA_DIR = "data"
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
IMAGES_PATH = os.path.join(DATA_DIR, "image_data.npy")

ARTIFACT_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
REPORT_DIR = os.path.join(ARTIFACT_DIR, "reports")
DB_PATH = os.path.join(ARTIFACT_DIR, "prediction_logs.sqlite")
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)


# -----------------------
# Helpers
# -----------------------
def build_model() -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(64,64,3)),

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_prediction_log_df(model, X, y_true, split_name: str, start_scan_id: int) -> pd.DataFrame:
    """
    Build a monitoring-style log table that matches the SQL file schema:
      prediction_logs(scan_id, split_name, predicted_label, true_label, confidence, created_at)
    """
    probs = model.predict(X, verbose=0).flatten()
    predicted_label = (probs >= 0.5).astype(int)

    # confidence in the predicted class, not just the positive class
    confidence = np.where(predicted_label == 1, probs, 1 - probs)

    n = len(y_true)
    # Create a realistic spread of timestamps so daily-trend SQL returns useful output
    timestamps = pd.date_range(
        end=pd.Timestamp.now(),
        periods=n,
        freq="H"
    )

    df = pd.DataFrame({
        "scan_id": np.arange(start_scan_id, start_scan_id + n),
        "split_name": [split_name] * n,
        "predicted_label": predicted_label.astype(int),
        "true_label": y_true.astype(int),
        "confidence": confidence.astype(float),
        "created_at": timestamps.astype(str),
    })
    return df


def main():
    print("Loading data...", flush=True)

    meta = pd.read_csv(META_PATH)
    images = np.load(IMAGES_PATH)

    print("Original shapes:")
    print("Metadata:", meta.shape)
    print("Images:", images.shape)

    # Split according to notebook metadata
    train_meta = meta[meta["split"] == "train"].reset_index(drop=True)
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)

    X_train_full = images[train_meta["index"].values]
    y_train_full = train_meta["class"].values.astype(int)

    X_test = images[test_meta["index"].values]
    y_test = test_meta["class"].values.astype(int)

    # Create validation split from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full,
    )

    print("\nAfter split:")
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    # Normalize
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    print("\nBuilding model...", flush=True)
    model = build_model()
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    print("\nTraining model...", flush=True)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    print("\nEvaluating model...", flush=True)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")

    # Save training history
    history_path = os.path.join(REPORT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Save evaluation summary
    eval_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "saved_model": MODEL_PATH,
    }
    eval_summary_path = os.path.join(REPORT_DIR, "evaluation_summary.json")
    with open(eval_summary_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Evaluation summary saved to {eval_summary_path}")

    # -----------------------
    # SQL logging / monitoring
    # -----------------------
    print("\nSaving prediction logs to SQLite...", flush=True)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # Build logs for validation and test so the SQL dashboard has more than one split
    val_logs = make_prediction_log_df(model, X_val, y_val, "validation", 0)
    test_logs = make_prediction_log_df(model, X_test, y_test, "test", len(val_logs))

    logs_df = pd.concat([val_logs, test_logs], ignore_index=True)

    # Save to the exact table name expected by the SQL file
    logs_df.to_sql("prediction_logs", conn, if_exists="replace", index=False)

    # Make sure schema exists and is query-friendly
    conn.commit()
    conn.close()

    print(f"Saved prediction logs to {DB_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()