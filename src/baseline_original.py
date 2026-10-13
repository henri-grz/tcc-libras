"""Reproducao fiel do pipeline de Prestes (2024).

Este modulo replica de proposito o pipeline original, inclusive os pontos discutidos
no diagnostico (docs/00_diagnostico_baseline.md). Nao corrigir nada aqui: e o pipeline
que os experimentos 01 a 04 reproduzem e reavaliam.

Replicado:
- uma amostra por landmark individual, `input_size = 3` (falha C1)
- ordem de leitura: primeiro `hand_landmarks`, depois `pose_landmarks`, como no
  Apendice B p. 60-61
- `StandardScaler` ajustado no dataset completo antes do split (falha C2.1)
- `train_test_split` com shuffle no nivel do ponto (falha C2.2)
- `create_sequences` com janela deslizante de 200 e rotulo do ultimo ponto (falha C2.3)

Diferenca deliberada: os dados vem do cache `.npz`, nao dos JSON agregados. Por padrao,
o descarte de videos da falha C3 (Apendice A) nao e reproduzido: entram todos os videos
da configuracao. No ambiente do autor, a C3 deixava 2 videos por sinal, com rotulos
corretos por causa da listagem alfabetica, mas quais videos sobravam depende de nomes de
arquivo que o trabalho nao informa. Com `--c3` no exp01, o descarte e simulado
(`sobreviventes_c3`) sob a suposicao de nomes descrita em
docs/05_protocolo_experimento1.md. Ver docs/00_diagnostico_baseline.md, C3.

Duas escolhas vem da configuracao (`baseline_original` no YAML):
- `ordem_pontos`: ordem em que os pontos de cada video entram no dataset
  (`pontos_do_video`). O original percorre frame a frame (`por_frame`).
- `pesos_finais`: o original fica com os pesos da ultima epoca (`ultima_epoca`).
As duas configuracoes usam os valores do original. As rodadas de 11 sinais anteriores
a 13/10/2026 (results/obsoleto/) usaram `por_video` e `melhor_epoca`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .dataset import load_metadata, load_video_arrays
from .splits import leaky_pointlevel_split

ORDENS_PONTOS = ("por_video", "por_frame")


@dataclass
class PointLevelData:
    """Dataset no formato do pipeline original: uma linha por landmark."""

    X: np.ndarray  # (n_points, 3)
    y: np.ndarray  # (n_points,) int
    groups: np.ndarray  # (n_points,) video_id, para o exp04
    class_names: list[str]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def describe(self) -> str:
        return (
            f"pontos: {len(self):,}\n"
            f"videos: {len(set(self.groups.tolist()))}\n"
            f"classes: {len(self.class_names)}\n"
            f"pontos por classe: "
            f"{pd.Series(self.y).value_counts().sort_index().to_dict()}"
        )


def pontos_do_video(arrays: dict[str, np.ndarray], ordem: str) -> np.ndarray:
    """Pontos 3D de um video na ordem em que entram no dataset do pipeline original.

    - "por_frame": em cada frame, mao esquerda, mao direita e pose, como o laco do
      Apendice B (p. 60-61). E a ordem do original.
    - "por_video": todos os frames da mao esquerda, depois da direita, depois da pose.
      Ordem das rodadas de 11 sinais anteriores a 13/10/2026 (results/obsoleto/).

    Landmark ausente era omitido no original; aqui saem as linhas com NaN.
    """
    partes = [arrays[n] for n in ("left_hand", "right_hand", "pose")]  # (F, K, 3)
    if ordem == "por_frame":
        flat = np.concatenate(partes, axis=1).reshape(-1, 3)
    elif ordem == "por_video":
        flat = np.concatenate([p.reshape(-1, 3) for p in partes], axis=0)
    else:
        raise ValueError(f"ordem_pontos desconhecida: {ordem}. disponiveis: {ORDENS_PONTOS}")
    return flat[np.isfinite(flat).all(axis=1)].astype(np.float32)


def restaurar_melhor_epoca(bo: dict) -> bool:
    """Le `baseline_original.pesos_finais`. Ausente = ultima epoca, como o original."""
    valor = bo.get("pesos_finais", "ultima_epoca")
    if valor not in ("melhor_epoca", "ultima_epoca"):
        raise ValueError(f"pesos_finais desconhecido: {valor}")
    return valor == "melhor_epoca"


def simular_c3(video_ids: list[str]) -> dict[str, list[str]]:
    """Simula `processa()` do Apendice A (p. 50) sobre a listagem dos videos.

    Supoe arquivos "Sinal N.mov", com N = coluna da V-LIBRASIL, listados em ordem
    alfabetica como no NTFS. Devolve {nome do JSON: videos gravados nele}. Um JSON
    gravado de novo com o mesmo nome substitui o anterior, como o open(..., "w").
    """
    nomes = sorted(video_ids, key=lambda v: v.replace("_", " ").casefold())
    classes = sorted(v.split("_")[0] for v in nomes)  # com repeticao, como clas_Arq
    arquivos: dict[str, list[str]] = {}
    todos: list[str] = []
    for num, vid in enumerate(nomes):
        todos.append(vid)
        if len(todos) == 2:
            arquivos[classes[num]] = todos
            todos = []
    return arquivos


def sobreviventes_c3(video_ids: list[str]) -> list[str]:
    """Videos que entram no treino do original depois da C3.

    Falha se algum JSON misturar sinais ou se faltar JSON para algum sinal: nesses casos
    os rotulos do original sairiam trocados e a simulacao nao serviria como reproducao.
    """
    arquivos = simular_c3(video_ids)
    sinais = {v.split("_")[0] for v in video_ids}
    if set(arquivos) != sinais:
        raise ValueError(f"C3: sinais sem JSON: {sorted(sinais - set(arquivos))}")
    for nome, vids in arquivos.items():
        if any(v.split("_")[0] != nome for v in vids):
            raise ValueError(f"C3: {nome}.json mistura sinais: {vids}")
    return sorted(v for vids in arquivos.values() for v in vids)


def build_pointlevel(
    cfg: Config,
    video_ids: list[str] | None = None,
    label_permutation: dict[str, str] | None = None,
) -> PointLevelData:
    """Monta o dataset no nivel do ponto, como no Apendice B.

    Args:
        video_ids: restringe aos videos indicados (usado pelo exp04).
        label_permutation: mapa video_id -> sinal, para embaralhar rotulos no nivel do
            video (usado pelo exp02).
    """
    meta = load_metadata(cfg)
    if video_ids is not None:
        meta = meta[meta["video_id"].isin(video_ids)]

    class_names = [c for c in cfg.classes if c in set(meta["sinal"])]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    ordem = cfg.baseline_original.get("ordem_pontos", "por_frame")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gs: list[np.ndarray] = []

    for row in meta.itertuples(index=False):
        arrays = load_video_arrays(cfg, row.npz)
        sinal = label_permutation.get(row.video_id, row.sinal) if label_permutation else row.sinal
        label = class_to_idx[sinal]

        # Maos primeiro, depois pose, na ordem `ordem_pontos` (ver pontos_do_video).
        pts = pontos_do_video(arrays, ordem)
        if pts.size == 0:
            continue
        xs.append(pts)
        ys.append(np.full(len(pts), label, dtype=np.int64))
        gs.append(np.full(len(pts), row.video_id, dtype=object))

    return PointLevelData(
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        groups=np.concatenate(gs, axis=0),
        class_names=class_names,
    )


def create_sequences(
    data: np.ndarray, labels: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Replica exata de `create_sequences` do Apendice B, p. 61.

    Janela deslizante de passo 1, rotulo do ultimo ponto da janela. Aplicada sobre um
    array ja embaralhado pelo split, e por isso cada "sequencia" mistura classes.
    """
    sequences, sequence_labels = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i : i + sequence_length])
        sequence_labels.append(labels[i + sequence_length - 1])
    return np.array(sequences, dtype=np.float32), np.array(sequence_labels, dtype=np.int64)


def _fit_scaler_on_all(X: np.ndarray) -> np.ndarray:
    """Replica `scaler.fit_transform(X)` aplicado ao dataset completo (falha C2.1)."""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(X).astype(np.float32)


def leaky_pipeline(
    pld: PointLevelData,
    sequence_length: int = 200,
    test_size: float = 0.1,
    random_state: int = 42,
    max_sequences: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """O pipeline original, com todas as falhas preservadas.

    Ordem exata do original: padroniza tudo, embaralha pontos, divide, e so depois cria
    as sequencias.

    Args:
        max_sequences: subamostra as janelas para caber na memoria. O original gerava
            uma janela por ponto, o que com milhoes de pontos e impraticavel. A
            subamostragem e uniforme e nao altera a natureza do vazamento.
    """
    X = _fit_scaler_on_all(pld.X)
    train_idx, test_idx = leaky_pointlevel_split(len(X), test_size, random_state)

    X_train_seq, y_train_seq = create_sequences(X[train_idx], pld.y[train_idx], sequence_length)
    X_test_seq, y_test_seq = create_sequences(X[test_idx], pld.y[test_idx], sequence_length)

    if max_sequences:
        X_train_seq, y_train_seq = _subsample(X_train_seq, y_train_seq, max_sequences, random_state)
        X_test_seq, y_test_seq = _subsample(
            X_test_seq, y_test_seq, max(max_sequences // 5, 1), random_state
        )

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq


def grouped_pipeline(
    cfg: Config,
    train_videos: list[str],
    test_videos: list[str],
    sequence_length: int = 200,
    max_sequences: int | None = None,
    random_state: int = 42,
    label_permutation: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mesma representacao do pipeline original, mas com split agrupado por video (exp04).

    A unica coisa que muda em relacao a `leaky_pipeline` e o split: os videos de teste
    nunca aparecem no treino, a padronizacao usa apenas estatisticas do treino, e as
    janelas sao contiguas dentro de cada conjunto. Isolar essa unica variavel e o que
    permite atribuir a queda de acuracia ao vazamento.
    """
    from sklearn.preprocessing import StandardScaler

    tr = build_pointlevel(cfg, video_ids=train_videos, label_permutation=label_permutation)
    te = build_pointlevel(cfg, video_ids=test_videos, label_permutation=label_permutation)

    scaler = StandardScaler().fit(tr.X)  # so o treino
    X_tr = scaler.transform(tr.X).astype(np.float32)
    X_te = scaler.transform(te.X).astype(np.float32)

    X_train_seq, y_train_seq = create_sequences(X_tr, tr.y, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_te, te.y, sequence_length)

    if max_sequences:
        X_train_seq, y_train_seq = _subsample(X_train_seq, y_train_seq, max_sequences, random_state)
        X_test_seq, y_test_seq = _subsample(
            X_test_seq, y_test_seq, max(max_sequences // 5, 1), random_state
        )

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq


def _subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def random_label_permutation(cfg: Config, seed: int = 42) -> dict[str, str]:
    """Sorteia um sinal aleatorio para cada video, mantendo o agrupamento (exp02).

    O ponto do teste: os rotulos passam a nao ter relacao nenhuma com o conteudo dos
    videos. Se o pipeline original ainda assim atingir acuracia alta, isso so pode vir
    de memorizacao de coordenadas que aparecem nos dois lados do split, ou seja,
    vazamento. E o teste classico de rotulo aleatorio.
    """
    meta = load_metadata(cfg)
    rng = np.random.RandomState(seed)
    classes = [c for c in cfg.classes if c in set(meta["sinal"])]
    return {
        vid: classes[int(rng.randint(len(classes)))]
        for vid in meta["video_id"].tolist()
    }
