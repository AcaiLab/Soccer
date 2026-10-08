"""End-to-end runs of the analysis scripts on small synthetic data.

The features have a weak label signal plus a strong per-game offset, like the
real embeddings, and use the 30 games of configs/games.csv. Scripts that need
torch are skipped when it is not installed (CI runs without it).
"""

import numpy as np
import pandas as pd
import pytest

from soccer.annotations import match_minute, seconds_to_tag

LABELS = ["corner", "clearance", "goal", "substitution", "unknown"]
GAMES = [f"{n:02d}" for n in range(1, 31)]
EVENTS_PER_GAME, T, D = 15, 6, 16


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    label_vec = {lab: rng.normal(size=D) for lab in LABELS}
    games = pd.read_csv("configs/games.csv", dtype=str)
    match_of = dict(zip(games.game_num, games.match_id, strict=True))

    feats, masks, labels, gids, halves, tags, gf_rows = [], [], [], [], [], [], []
    for g in GAMES:
        offset = rng.normal(scale=3, size=D)
        for e in range(EVENTS_PER_GAME):
            lab = LABELS[e % len(LABELS)]
            half = 1 + e % 2
            seconds = 60 * (e + 1)
            n = rng.integers(2, T + 1)
            x = np.zeros((T, D), np.float32)
            x[:n] = label_vec[lab] + offset + rng.normal(scale=0.5, size=(n, D))
            x[:n] /= np.linalg.norm(x[:n], axis=1, keepdims=True)
            m = np.zeros(T, bool)
            m[:n] = True
            feats.append(x)
            masks.append(m)
            labels.append(lab)
            gids.append(g)
            halves.append(str(half))
            tags.append(seconds_to_tag(seconds))
            gf_rows.append({"match_id": match_of[g], "league": "england_epl_2021-2022", "official_split": "train",
                            "half": half, "minute": match_minute(seconds, half), "label": lab,
                            "category": "other", "prev_1": "corner", "prev_2": "none", "prev_3": "none",
                            "prev_1_cat": "attack", "prev_2_cat": "none", "prev_3_cat": "none",
                            "recent_corners": e % 3, "recent_shots": 0, "recent_fouls": 1, "recent_cards": 0,
                            "recent_attacks": 1, "recent_stoppages": 0, "text": ""})

    temporal = root / "temporal"
    temporal.mkdir()
    np.savez_compressed(temporal / "clip_temporal_padded.npz", features=np.stack(feats), masks=np.stack(masks),
                        labels=np.array(labels), games=np.array(gids), halves=np.array(halves),
                        event_tags=np.array(tags))
    flat_dir = root / "flat"
    flat_dir.mkdir()
    np.save(flat_dir / "m_features.npy", np.stack(feats)[:, 0])
    pd.DataFrame({"label": labels}).to_csv(flat_dir / "metadata.csv", index=False)
    gf = pd.DataFrame(gf_rows)
    gf.to_csv(root / "game_flow.csv", index=False)
    other = gf.assign(league="spain_laliga_2021-2022", match_id=gf.match_id + "_es")
    pd.concat([gf, other]).to_csv(root / "game_flow_two_leagues.csv", index=False)
    return root


def test_cluster_analysis(data):
    import cluster_analysis

    out = data / "clusters"
    cluster_analysis.main(["--features-dir", str(data / "temporal"), "--flat-features",
                           str(data / "flat" / "m_features.npy"), "--out-dir", str(out)])
    metrics = pd.read_csv(out / "clustering_metrics.csv")
    assert {"ARI", "Purity", "Purity_random"} <= set(metrics.columns) and len(metrics) == 4
    assert (out / "D_colorings_game_half.png").is_file() and (out / "per_game" / "game_30.png").is_file()


def test_within_game_reports_random_baseline(data):
    import within_game_clustering

    out = data / "within"
    within_game_clustering.main(["--features-dir", str(data / "temporal"), "--out-dir", str(out)])
    metrics = pd.read_csv(out / "within_game_metrics.csv")
    assert len(metrics) == 30 and (metrics["Purity_random"] > 0).all()


def test_event_classification_selects_on_validation(data):
    import event_classification

    out = data / "cls"
    event_classification.main(["--dino-features", str(data / "temporal"),
                               "--contrastive-features", str(data / "none"), "--out-dir", str(out)])
    report = (out / "classification_report.txt").read_text()
    assert "Chosen on validation" in report and "majority-class baseline" in report
    results = pd.read_csv(out / "classification_results.csv")
    assert "Val_accuracy" in results and not any("Test" in c for c in results.columns)


def test_multimodal_classification(data):
    import multimodal_classification

    out = data / "mm"
    multimodal_classification.main(["--dino-features", str(data / "temporal"),
                                    "--game-flow-features", str(data / "game_flow.csv"), "--out-dir", str(out)])
    report = (out / "multimodal_classification_report.txt").read_text()
    assert "Chosen on validation" in report
    # every synthetic event has an annotation partner: 6 test games x (15 - 3 "unknown") events
    assert "(72 matched events)" in report
    assert len(pd.read_csv(out / "multimodal_results.csv")) == 9


def test_temporal_event_classifier_cross_league(data):
    import temporal_event_classifier

    out = data / "annot"
    temporal_event_classifier.main(["--features", str(data / "game_flow_two_leagues.csv"),
                                    "--out-dir", str(out), "--cross-league"])
    assert len(pd.read_csv(out / "cross_league_results.csv")) == 2


def test_contrastive_train_then_evaluate(data):
    pytest.importorskip("torch")
    import contrastive_training
    import evaluate_contrastive

    cont = data / "contrastive"
    contrastive_training.main(["--features-dir", str(data / "temporal"), "--out-dir", str(cont),
                               "--epochs", "2", "--device", "cpu"])
    saved = np.load(cont / "contrastive_embeddings.npz")
    assert set(saved["train_games"]) == set(GAMES[:20])

    out = data / "cont_eval"
    evaluate_contrastive.main(["--dino-features", str(data / "temporal"), "--contrastive-features", str(cont),
                               "--out-dir", str(out)])
    metrics = pd.read_csv(out / "contrastive_metrics.csv")
    assert set(metrics["Group"]) == {"held-out games", "train games"}


def test_contrastive_refuses_test_games(data):
    pytest.importorskip("torch")
    import contrastive_training

    with pytest.raises(ValueError, match="both train and test"):
        contrastive_training.main(["--features-dir", str(data / "temporal"), "--out-dir", str(data / "x"),
                                   "--train-games", "01", "25", "--epochs", "1", "--device", "cpu"])


def test_sequence_classifier(data):
    pytest.importorskip("torch")
    import sequence_classifier

    out = data / "seq"
    sequence_classifier.main(["--features", str(data / "temporal"), "--out-dir", str(out), "--epochs", "2",
                              "--hidden-dim", "16", "--device", "cpu"])
    assert "Chosen on validation" in (out / "sequence_classification_report.txt").read_text()
