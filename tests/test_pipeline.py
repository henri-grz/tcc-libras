"""Testes minimos das partes que, se quebrarem em silencio, invalidam o TCC.

Rodar com:
    python -m pytest tests -q
ou, sem pytest:
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import standardize  # noqa: E402
from src.extract_landmarks import parse_filename  # noqa: E402
from src.landmarks_spec import (  # noqa: E402
    COORDS,
    N_HAND,
    FeatureSpec,
    build_frame_matrix,
    resample_to_length,
)
from src.splits import assert_no_group_leakage, grouped_train_val_test, loso_folds  # noqa: E402

SPEC = FeatureSpec(pose_indices=tuple(range(25)))


def test_feature_dim_and_blocks() -> None:
    assert SPEC.dim == 25 * 3 + 2 * 21 * 3 + 2 == 203
    blocks = SPEC.block_slices()
    assert set(blocks) == {"pose", "left_hand", "right_hand", "flags"}
    total = sum(s.stop - s.start for s in blocks.values())
    assert total == SPEC.dim
    assert len(SPEC.slice_names()) == SPEC.dim


def test_missing_hand_becomes_zero_plus_flag() -> None:
    """Mao ausente nao pode virar NaN nem deslocar os outros landmarks."""
    n = 4
    pose = np.zeros((n, 33, COORDS), np.float32)
    pose[:, 11] = (0.4, 0.4, 0.0)
    pose[:, 12] = (0.6, 0.4, 0.0)
    pose[:, 23] = (0.45, 0.8, 0.0)
    pose[:, 24] = (0.55, 0.8, 0.0)

    left = np.full((n, N_HAND, COORDS), np.nan, np.float32)  # ausente
    right = np.full((n, N_HAND, COORDS), 0.5, np.float32)

    m = build_frame_matrix(pose, left, right, SPEC, normalize=True)
    assert m.shape == (n, SPEC.dim)
    assert np.isfinite(m).all(), "nenhum NaN pode sobrar no vetor final"

    blocks = SPEC.block_slices()
    assert np.allclose(m[:, blocks["left_hand"]], 0.0)
    flags = m[:, blocks["flags"]]
    assert np.allclose(flags[:, 0], 0.0), "flag da mao esquerda deve ser 0"
    assert np.allclose(flags[:, 1], 1.0), "flag da mao direita deve ser 1"


def test_normalization_is_translation_invariant() -> None:
    """Deslocar o sinalizador no quadro nao pode mudar a representacao."""
    n = 6
    pose = np.zeros((n, 33, COORDS), np.float32)
    pose[:, 11] = (0.4, 0.4, 0.0)
    pose[:, 12] = (0.6, 0.4, 0.0)
    pose[:, 23] = (0.45, 0.8, 0.0)
    pose[:, 24] = (0.55, 0.8, 0.0)
    hand = np.full((n, N_HAND, COORDS), 0.5, np.float32)

    a = build_frame_matrix(pose, hand, hand, SPEC, normalize=True)
    shift = np.array([0.1, -0.05, 0.0], np.float32)
    b = build_frame_matrix(pose + shift, hand + shift, hand + shift, SPEC, normalize=True)
    assert np.allclose(a, b, atol=1e-5), "normalizacao deve ser invariante a translacao"


def test_resample_length() -> None:
    src = np.random.RandomState(0).rand(97, 7).astype(np.float32)
    for t in (8, 32, 200):
        assert resample_to_length(src, t).shape == (t, 7)
    assert resample_to_length(np.zeros((0, 7), np.float32), 16).shape == (16, 7)
    assert resample_to_length(np.ones((1, 7), np.float32), 16).shape == (16, 7)


def test_standardize_uses_train_stats_only() -> None:
    """Se as estatisticas viessem do conjunto todo, isso seria a falha C2.1."""
    rng = np.random.RandomState(0)
    train = rng.normal(0, 1, size=(20, 10, 4)).astype(np.float32)
    test = rng.normal(50, 1, size=(5, 10, 4)).astype(np.float32)
    tr_s, te_s = standardize(train, test)
    assert abs(float(tr_s.reshape(-1, 4).mean())) < 1e-4, "treino deve ficar centrado"
    assert float(te_s.mean()) > 5, "teste NAO deve ser recentrado nas proprias estatisticas"


def test_group_leakage_guard() -> None:
    groups = np.array(["v1", "v1", "v2", "v2", "v3", "v3"])
    assert_no_group_leakage(groups, np.array([0, 1]), np.array([2, 3]))
    try:
        assert_no_group_leakage(groups, np.array([0, 1]), np.array([1, 2]))
    except AssertionError:
        pass
    else:
        raise AssertionError("a guarda deveria ter detectado o vazamento")


def test_splits_are_group_disjoint() -> None:
    n_classes, n_signers = 5, 3
    y, groups, signers = [], [], []
    for c in range(n_classes):
        for s in range(1, n_signers + 1):
            y.append(c)
            groups.append(f"c{c}_s{s}")
            signers.append(str(s))
    y = np.array(y)
    groups = np.array(groups)
    signers = np.array(signers)

    split = grouped_train_val_test(y, groups, test_size=0.3, val_size=0.3, seed=42)
    assert_no_group_leakage(groups, split.train, split.test)
    assert_no_group_leakage(groups, split.train, split.val)
    assert_no_group_leakage(groups, split.val, split.test)

    folds = loso_folds(y, groups, signers, val_size=0.3, seed=42)
    assert len(folds) == n_signers
    for fold in folds:
        assert len(set(signers[fold.test])) == 1, "cada fold testa um unico sinalizador"
        assert not set(signers[fold.test]) & set(signers[fold.train])


def test_filename_parser() -> None:
    pattern = r"^(?P<sinal>[A-Za-zÀ-ÿ]+)[ _-]+(?P<articulador>[0-9A-Za-z]+)"
    for name in ("Abacaxi 1", "Abacaxi_1", "Abacaxi-1"):
        assert parse_filename(name, pattern) == ("Abacaxi", "1")
    assert parse_filename("12345", pattern) == (None, None)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FALHA {name}: {exc}")
    print(f"\n{'todos os testes passaram' if not failures else f'{failures} falha(s)'}")
    sys.exit(1 if failures else 0)
