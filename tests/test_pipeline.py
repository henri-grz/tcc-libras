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
from src.splits import (  # noqa: E402
    assert_no_group_leakage,
    grouped_train_val_test,
    loso_folds,
    protocol_train_val,
)

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


def test_protocol_train_val() -> None:
    """Revisado: treino = todos os outros sinalizadores, sem validacao. Original: intacto."""
    y = np.repeat(np.arange(5), 3)
    signers = np.tile(np.array(["1", "2", "3"]), 5)
    groups = np.array([f"c{c}_s{s}" for c, s in zip(y, signers)])
    for fold in loso_folds(y, groups, signers, val_size=0.3, seed=42):
        tr, val = protocol_train_val(fold, "revisado")
        assert len(val) == 0, "protocolo revisado nao tem validacao"
        assert sorted(tr.tolist()) == sorted(np.where(signers != signers[fold.test[0]])[0].tolist())
        assert_no_group_leakage(groups, tr, fold.test)

        tr, val = protocol_train_val(fold, "original")
        assert np.array_equal(tr, fold.train) and np.array_equal(val, fold.val)


def test_fixed_epochs_keeps_final_weights() -> None:
    """monitor="none" treina todas as epocas e nao restaura pesos antigos."""
    import torch

    from src.train import train_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(8, 4, 3)).astype(np.float32)
    y = np.arange(8) % 2
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
    hist = train_model(model, X, y, epochs=7, monitor="none", device=torch.device("cpu"),
                       verbose=False)
    assert hist.stopped_epoch == 6 and hist.best_epoch == 6
    assert len(hist.train_loss) == 7


def test_folds_por_coluna_nao_compartilham_pessoas() -> None:
    """A coluna "articulador" nao e a pessoa (data/pessoas.csv). Os folds do exp05 so
    sao independentes de sinalizador se nenhuma pessoa do teste aparecer no treino.
    Verificado nas duas configuracoes: 11 sinais (default) e 7 sinais."""
    from src.config import PROJECT_ROOT, load_config
    from src.dataset import load_metadata, load_pessoas

    for cfg_path in (None, PROJECT_ROOT / "configs" / "7sinais.yaml"):
        cfg = load_config(cfg_path)
        pessoas = load_pessoas(cfg)
        if pessoas is None or not cfg.metadata_csv.exists():
            print("     (pulado: sem data/pessoas.csv ou sem cache)")
            return
        meta = load_metadata(cfg)
        if not set(meta["video_id"]) <= set(pessoas):
            print("     (pulado: cache nao corresponde ao mapeamento, ex.: sintetico)")
            return
        if cfg_path is not None:
            assert len(meta) == 21, f"7 sinais: esperados 21 videos, ha {len(meta)}"
        y = meta["sinal"].astype("category").cat.codes.to_numpy()
        groups = meta["video_id"].to_numpy()
        signers = meta["articulador"].astype(str).to_numpy()
        pessoa = np.array([pessoas[v] for v in groups])
        for fold in loso_folds(y, groups, signers, val_size=0.2, seed=42):
            treino = np.concatenate([fold.train, fold.val])
            comum = set(pessoa[fold.test]) & set(pessoa[treino])
            assert not comum, f"{fold.name}: pessoas no treino e no teste: {sorted(comum)}"


def test_split_do_pipeline_original_igual_ao_de_prestes() -> None:
    """O split no nivel do ponto e o `train_test_split` do original (Apendice B, p. 61),
    que arredonda o teste para cima: com 7 sinais (293.733 pontos), 29.374 no teste."""
    from sklearn.model_selection import train_test_split

    from src.splits import leaky_pointlevel_split

    for n in (293_733, 182_166, 489_129):
        tr, te = leaky_pointlevel_split(n, 0.1, 42)
        sk_tr, sk_te = train_test_split(np.arange(n), test_size=0.1, random_state=42)
        assert np.array_equal(tr, sk_tr) and np.array_equal(te, sk_te)
    assert len(leaky_pointlevel_split(293_733, 0.1, 42)[1]) == 29_374


def test_filename_parser() -> None:
    pattern = r"^(?P<sinal>[A-Za-zÀ-ÿ]+)[ _-]+(?P<articulador>[0-9A-Za-z]+)"
    for name in ("Abacaxi 1", "Abacaxi_1", "Abacaxi-1"):
        assert parse_filename(name, pattern) == ("Abacaxi", "1")
    assert parse_filename("12345", pattern) == (None, None)

def test_simulacao_c3() -> None:
    from src.baseline_original import sobreviventes_c3

    sete = ["Abacaxi", "Abraco", "Cafe", "Casa", "Esperar", "Livro", "Oi"]
    ids = [f"{s}_{i}" for s in sete for i in (1, 2, 3)]
    assert sobreviventes_c3(ids) == sorted([
        "Abacaxi_1", "Abacaxi_2", "Abraco_2", "Abraco_3", "Cafe_1", "Cafe_2",
        "Casa_2", "Casa_3", "Esperar_1", "Esperar_2", "Livro_2", "Livro_3",
        "Oi_1", "Oi_2",
    ])
    onze = sete + ["Ano", "Escola", "Meia", "Ocupado"]
    assert len(sobreviventes_c3([f"{s}_{i}" for s in onze for i in (1, 2, 3)])) == 22


def test_ordem_dos_pontos_do_pipeline_original() -> None:
    from src.baseline_original import pontos_do_video

    vazio = np.full((2, 21, 3), np.nan, np.float32)
    esq, dir_ = vazio.copy(), vazio.copy()
    esq[0], dir_[1] = 1.0, 2.0  # mao esquerda so no frame 0, direita so no frame 1
    pose = np.stack([np.full((33, 3), 3.0), np.full((33, 3), 4.0)]).astype(np.float32)
    arrays = {"left_hand": esq, "right_hand": dir_, "pose": pose}
    assert pontos_do_video(arrays, "por_frame")[:, 0].tolist() == (
        [1.0] * 21 + [3.0] * 33 + [2.0] * 21 + [4.0] * 33
    )
    assert pontos_do_video(arrays, "por_video")[:, 0].tolist() == (
        [1.0] * 21 + [2.0] * 21 + [3.0] * 33 + [4.0] * 33
    )


def test_pipeline_original_pode_manter_pesos_da_ultima_epoca() -> None:
    """restore_best=False termina com os mesmos pesos que monitor="none"."""
    import torch

    from src.train import train_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(8, 4, 3)).astype(np.float32)
    y = np.arange(8) % 2
    finais = []
    for kw in (
        {"monitor": "none"},
        {"monitor": "train_loss", "patience": 100, "restore_best": False},
    ):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
        train_model(model, X, y, epochs=5, device=torch.device("cpu"), verbose=False, **kw)
        finais.append([p.detach().clone() for p in model.parameters()])
    assert all(torch.equal(a, b) for a, b in zip(*finais))


def test_config_7sinais_so_difere_no_previsto() -> None:
    import yaml

    from src.config import PROJECT_ROOT

    def folhas(d: dict, pre: tuple = ()):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from folhas(v, pre + (k,))
            else:
                yield pre + (k,), v

    def ler(nome: str) -> dict:
        with open(PROJECT_ROOT / "configs" / nome, encoding="utf-8") as f:
            return dict(folhas(yaml.safe_load(f)))

    base, sete = ler("default.yaml"), ler("7sinais.yaml")
    previstas = {("classes",), ("paths", "results"), ("baseline_original", "epochs")}
    difs = {k for k in base.keys() | sete.keys() if base.get(k) != sete.get(k)}
    assert difs <= previstas, f"diferencas nao previstas: {sorted(difs - previstas)}"


def test_aumento_neutro_e_identidade() -> None:
    import torch

    from src.augment import ParamsAumento, aumentar_lote

    x = torch.randn(4, 32, SPEC.dim)
    neutro = ParamsAumento(corte_temporal_max=0.0, rotacao_graus=0.0, escala=(1.0, 1.0),
                           ruido_std=0.0, aspecto=0.75)
    y = aumentar_lote(x, SPEC, neutro, torch.Generator().manual_seed(0))
    assert torch.allclose(x, y, atol=1e-6)


def test_aumento_preserva_ausentes_e_flags() -> None:
    import torch

    from src.augment import ParamsAumento, aumentar_lote

    blocos = SPEC.block_slices()
    x = torch.randn(3, 32, SPEC.dim)
    x[:, :, blocos["left_hand"]] = 0.0          # mao esquerda ausente
    x[:, :, blocos["flags"]] = torch.tensor([0.0, 1.0])
    p = ParamsAumento(corte_temporal_max=0.0, aspecto=0.75)  # sem corte: flags iguais
    y = aumentar_lote(x, SPEC, p, torch.Generator().manual_seed(0))
    assert y.shape == x.shape
    assert torch.equal(y[:, :, blocos["left_hand"]], x[:, :, blocos["left_hand"]])
    assert torch.equal(y[:, :, blocos["flags"]], x[:, :, blocos["flags"]])
    assert not torch.allclose(y[:, :, blocos["pose"]], x[:, :, blocos["pose"]])


def test_rotacao_preserva_distancias_isotropicas() -> None:
    """Girar as fracoes de largura e de altura diretamente deformaria o corpo no quadro
    1024 x 768. Com y convertido pelo aspecto, a rotacao preserva as distancias reais."""
    import torch

    from src.augment import ParamsAumento, aumentar_lote

    x = torch.randn(2, 8, SPEC.dim)
    p = ParamsAumento(corte_temporal_max=0.0, rotacao_graus=30.0, escala=(1.0, 1.0),
                      ruido_std=0.0, aspecto=0.75)
    y = aumentar_lote(x, SPEC, p, torch.Generator().manual_seed(1))
    n = SPEC.block_slices()["right_hand"].stop
    em_pixels = torch.tensor([1.0, 0.75])  # y na unidade de x
    a = x[:, :, :n].reshape(2, 8, -1, 3)[..., :2] * em_pixels
    b = y[:, :, :n].reshape(2, 8, -1, 3)[..., :2] * em_pixels
    assert torch.allclose(a.norm(dim=-1), b.norm(dim=-1), atol=1e-5)
    assert not torch.allclose(a, b)


def test_aumento_deterministico_por_semente() -> None:
    import torch

    from src.augment import ParamsAumento, aumentar_lote

    x = torch.randn(3, 32, SPEC.dim)
    p = ParamsAumento(aspecto=0.75)
    a = aumentar_lote(x, SPEC, p, torch.Generator().manual_seed(7))
    b = aumentar_lote(x, SPEC, p, torch.Generator().manual_seed(7))
    c = aumentar_lote(x, SPEC, p, torch.Generator().manual_seed(8))
    assert torch.equal(a, b), "a mesma semente deve gerar o mesmo aumento"
    assert not torch.allclose(a, c), "sementes diferentes devem gerar aumentos diferentes"


def test_batch_transform_so_no_treino() -> None:
    import torch

    from src.train import predict, train_model

    chamadas = []

    def transform(xb):
        chamadas.append(len(xb))
        return xb

    rng = np.random.RandomState(0)
    X = rng.normal(size=(8, 4, 3)).astype(np.float32)
    y = np.arange(8) % 2
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
    train_model(model, X, y, epochs=3, batch_size=4, monitor="none",
                device=torch.device("cpu"), verbose=False, batch_transform=transform)
    assert len(chamadas) == 6  # 3 epocas x 2 lotes
    predict(model, X, device=torch.device("cpu"))
    assert len(chamadas) == 6, "predict nao pode aplicar o aumento"


def test_standardization_stats_igual_ao_standardize() -> None:
    from src.dataset import standardization_stats

    rng = np.random.RandomState(0)
    tr = rng.normal(5, 2, size=(6, 4, 3)).astype(np.float32)
    mean, std = standardization_stats(tr)
    (tr_s,) = standardize(tr)
    assert np.allclose(tr_s, (tr - mean) / std, atol=1e-5)


def test_aumento_neutro_reproduz_treino_padronizado() -> None:
    """O caminho com_aumento do exp07 (X bruto, aumento e padronizacao no lote) com
    intensidades zero treina como o sem_aumento (X padronizado antes). Entao a unica
    diferenca entre as condicoes e o aumento."""
    import torch

    from src.augment import ParamsAumento, aumentar_lote
    from src.dataset import standardization_stats
    from src.train import train_model

    rng = np.random.RandomState(0)
    X = rng.normal(2, 3, size=(8, 4, SPEC.dim)).astype(np.float32)
    y = np.arange(8) % 2
    neutro = ParamsAumento(corte_temporal_max=0.0, rotacao_graus=0.0, escala=(1.0, 1.0),
                           ruido_std=0.0, aspecto=0.75)
    mean, std = standardization_stats(X)
    mean_t, std_t = torch.as_tensor(mean), torch.as_tensor(std)
    gen = torch.Generator().manual_seed(0)

    finais = []
    for X_in, transform in (
        (standardize(X)[0], None),
        (X, lambda xb: (aumentar_lote(xb, SPEC, neutro, gen) - mean_t) / std_t),
    ):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4 * SPEC.dim, 2))
        train_model(model, X_in, y, epochs=3, batch_size=4, monitor="none",
                    device=torch.device("cpu"), verbose=False, batch_transform=transform)
        finais.append([p.detach().clone() for p in model.parameters()])
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(*finais))


def test_checkpoint_ida_e_volta() -> None:
    """O modelo carregado recebe landmarks brutos e monta a mesma entrada que os
    experimentos montam (build_frame_matrix, reamostragem, colunas da ablacao,
    padronizacao do treino), e devolve as mesmas logits. Vetor inteiro e so maos."""
    import tempfile

    import torch

    from src.checkpoint import load_checkpoint, save_checkpoint
    from src.dataset import standardization_stats
    from src.models import build_model

    rng = np.random.RandomState(0)
    T, n_frames = 8, 13

    def video() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = rng.uniform(0.2, 0.8, size=(n_frames, 33, COORDS)).astype(np.float32)
        left = rng.uniform(0.2, 0.8, size=(n_frames, N_HAND, COORDS)).astype(np.float32)
        right = rng.uniform(0.2, 0.8, size=(n_frames, N_HAND, COORDS)).astype(np.float32)
        left[::3] = np.nan  # mao ausente em alguns frames
        return pose, left, right

    videos = [video() for _ in range(6)]
    X = np.stack([
        resample_to_length(build_frame_matrix(*v, SPEC, normalize=True), T) for v in videos
    ])
    slices = SPEC.block_slices()

    for blocks in (None, ["left_hand", "right_hand", "flags"]):
        nomes = blocks or list(slices)
        cols = np.concatenate([np.arange(slices[b].start, slices[b].stop) for b in nomes])
        X_b = X[:, :, cols]
        mean, std = standardization_stats(X_b)
        (X_s,) = standardize(X_b)

        torch.manual_seed(0)
        model = build_model("lstm_corrected", input_size=len(cols), num_classes=3).eval()
        with tempfile.TemporaryDirectory() as d:
            path = save_checkpoint(
                Path(d) / "m.pth", model, "lstm_corrected", ["a", "b", "c"], SPEC, blocks,
                sequence_length=T, normalize=True, mean=mean, std=std,
                extracao={"frame_width": 1024, "frame_height": 768}, info={},
            )
            carregado = load_checkpoint(path)

        for i, v in enumerate(videos):
            entrada = carregado.preparar(*v)
            assert np.allclose(entrada, X_s[i], atol=1e-5)
            with torch.no_grad():
                esperado = model(torch.from_numpy(X_s[i : i + 1]))[0].softmax(0).numpy()
            assert np.allclose(carregado.probabilidades(*v), esperado, atol=1e-5)
        assert carregado.prever(*videos[0])[0] in {"a", "b", "c"}


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
