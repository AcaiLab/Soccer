# Train ordinary classifiers on event windows, not individual frames.

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


# Remove the per-frame suffix so all frames from one event share an id.
def window_id_from_clip_id(clip_id: str) -> str:
    return re.sub(r"_frame\d{2}-\d{2}\.\d+$", "", clip_id)


# sort frames within each event window.
def frame_time_from_clip_id(clip_id: str) -> float:
    match = re.search(r"_frame(\d{2})-(\d{2}\.\d+)$", clip_id)
    if not match:
        return 0.0
    return int(match.group(1)) * 60.0 + float(match.group(2))


# Convert frame-level rows into event-window rows
def build_windows(frame_manifest: pd.DataFrame) -> pd.DataFrame:
    df = frame_manifest.copy()
    df["window_id"] = df["clip_id"].map(window_id_from_clip_id)
    df["frame_time"] = df["clip_id"].map(frame_time_from_clip_id)

    windows = []
    for window_id, group in df.sort_values("frame_time").groupby("window_id", sort=True):
        labels = group["event_label"].unique()
        games = group["game_label"].unique()

        # A valid window should correspond to exactly one event label in one game.
        if len(labels) != 1 or len(games) != 1:
            raise ValueError(f"Window has mixed labels or games: {window_id}")

        windows.append(
            {
                "window_id": window_id,
                "event_label": labels[0],
                "game_label": games[0],
                "frame_count": len(group),
                "image_paths": list(group["image_path"]),
            }
        )

    return pd.DataFrame(windows)


# turn per-frame embeddings into fixed-size window features.
def make_window_features(
    windows: pd.DataFrame,
    path_to_embedding_row: dict[str, int],
    frame_embeddings: np.ndarray,
    sampled_frames: int = 8,
) -> dict[str, np.ndarray]:
    mean_std_features = []
    sampled_concat_features = []

    for paths in windows["image_paths"]:
        rows = np.array([path_to_embedding_row[path] for path in paths])
        sequence = frame_embeddings[rows]

        # Pooling uses all frames and is robust to variable window lengths.
        mean = sequence.mean(axis=0)
        std = sequence.std(axis=0)
        mean_std_features.append(np.concatenate([mean, std]).astype(np.float32))

        # Fixed-position sampling preserves rough temporal order.
        if len(sequence) == 1:
            sampled = np.repeat(sequence, sampled_frames, axis=0)
        else:
            positions = np.linspace(0, len(sequence) - 1, sampled_frames).round().astype(int)
            sampled = sequence[positions]
        sampled_concat_features.append(sampled.reshape(-1).astype(np.float32))

    return {
        "mean_std_pool": np.vstack(mean_std_features),
        f"sample{sampled_frames}_concat": np.vstack(sampled_concat_features),
    }


# basleines
def make_classifiers(seed: int, include_linear_svm: bool = True) -> list[tuple[str, object]]:
    classifiers: list[tuple[str, object]] = [
        ("majority", DummyClassifier(strategy="most_frequent")),
        ("knn5_cosine", KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")),
        (
            "logreg_balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            ),
        ),
    ]

    if include_linear_svm:
        classifiers.append(
            (
                "linear_svm_balanced",
                make_pipeline(
                    StandardScaler(),
                    LinearSVC(max_iter=6000, class_weight="balanced", dual="auto"),
                ),
            )
        )

    classifiers.append(
        (
            "random_forest_balanced",
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    )
    return classifiers


def evaluate_window_classifiers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_frame_weights: np.ndarray,
    labels: np.ndarray,
    seed: int,
    include_linear_svm: bool = True,
) -> list[dict[str, object]]:
    rows = []
    for predictor_name, classifier in make_classifiers(seed, include_linear_svm):
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)

        rows.append(
            {
                "predictor": predictor_name,
                "window_accuracy": accuracy_score(y_test, predictions),
                "frame_weighted_accuracy": accuracy_score(
                    y_test,
                    predictions,
                    sample_weight=test_frame_weights,
                ),
                "macro_f1": f1_score(
                    y_test,
                    predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
                "weighted_f1": f1_score(
                    y_test,
                    predictions,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                ),
            }
        )

    return rows


def run_frame_window_workflow(
    frame_manifest: pd.DataFrame,
    frame_embeddings_by_name: dict[str, np.ndarray],
    path_to_embedding_row_by_name: dict[str, dict[str, int]],
    train_games: list[str],
    test_games: list[str],
    sampled_frames: int = 8,
    seed: int = 7,
) -> pd.DataFrame:
    windows = build_windows(frame_manifest)
    train_windows = windows[windows["game_label"].isin(train_games)].copy()
    test_windows = windows[windows["game_label"].isin(test_games)].copy()

    # Keep only test labels that exist in training.
    known_labels = set(train_windows["event_label"])
    test_windows = test_windows[test_windows["event_label"].isin(known_labels)].copy()

    selected_windows = pd.concat(
        [
            train_windows.assign(split_name="train"),
            test_windows.assign(split_name="test"),
        ],
        ignore_index=True,
    )

    label_encoder = LabelEncoder().fit(sorted(train_windows["event_label"].unique()))
    y_train = label_encoder.transform(train_windows["event_label"])
    y_test = label_encoder.transform(test_windows["event_label"])
    label_ids = np.arange(len(label_encoder.classes_))
    test_frame_weights = test_windows["frame_count"].to_numpy()

    train_idx = np.flatnonzero(selected_windows["split_name"].to_numpy() == "train")
    test_idx = np.flatnonzero(selected_windows["split_name"].to_numpy() == "test")

    all_results = []
    for embedding_name, frame_embeddings in frame_embeddings_by_name.items():
        window_feature_sets = make_window_features(
            selected_windows,
            path_to_embedding_row_by_name[embedding_name],
            frame_embeddings,
            sampled_frames=sampled_frames,
        )

        for feature_name, x_all in window_feature_sets.items():
            # Skip linear SVM for high-dimensional temporal concatenation.
            include_linear_svm = not feature_name.startswith("sample")
            rows = evaluate_window_classifiers(
                x_all[train_idx],
                y_train,
                x_all[test_idx],
                y_test,
                test_frame_weights,
                label_ids,
                seed,
                include_linear_svm=include_linear_svm,
            )

            for row in rows:
                row.update(
                    {
                        "embedding": embedding_name,
                        "window_feature": feature_name,
                        "train_windows": len(train_windows),
                        "test_windows": len(test_windows),
                        "test_frames": int(test_frame_weights.sum()),
                    }
                )
            all_results.extend(rows)

    return pd.DataFrame(all_results).sort_values(
        ["frame_weighted_accuracy", "macro_f1"],
        ascending=False,
    )


# for reporting
def summarize_windows(windows: pd.DataFrame) -> dict[str, object]:
    return {
        "num_windows": len(windows),
        "num_games": windows["game_label"].nunique(),
        "num_labels": windows["event_label"].nunique(),
        "label_counts": dict(Counter(windows["event_label"])),
        "frames_per_window": windows["frame_count"].describe().to_dict(),
    }
