"""EXP07 - Aumento de dados no pipeline corrigido.

Pergunta: aumentar os dados de treino com transformacoes dos landmarks melhora o
reconhecimento em sinalizadores que o modelo nunca viu?

Comparacao pareada, em cada seed e em cada fold, entre duas condicoes iguais em tudo
menos nas transformacoes:
- sem_aumento: identico ao exp05, protocolo revisado do docs/04
- com_aumento: cada lote de treino recebe uma versao nova de cada video (corte
  temporal, rotacao, escala e ruido, ver src/augment.py)

Com a mesma seed, as duas condicoes tem os mesmos pesos iniciais, a mesma ordem de
lotes e o mesmo dropout. O aumento usa um gerador proprio, que nao consome o RNG
global. O aumento e aplicado so ao treino, dentro do fold, depois da separacao.

Nas seeds do exp05, a condicao sem_aumento deve repetir as matrizes de confusao do
exp05 (verificacao de integridade).

Registro previo, criterio de leitura e vedacoes em docs/06_protocolo_exp07_aumento.md.

Uso:
    python -m experiments.exp07_aumento_dados --smoke
    python -m experiments.exp07_aumento_dados
    python -m experiments.exp07_aumento_dados --config configs/7sinais.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import (  # noqa: E402
    banner,
    base_parser,
    salvar_modelo_final,
    stopping_kwargs,
)
from experiments.diag_exp05 import binom_sf  # noqa: E402

from src.augment import ParamsAumento, aumentar_lote
from src.config import load_config
from src.dataset import build_dataset, standardization_stats, standardize
from src.evaluate import (
    aggregate_runs,
    compute_metrics,
    plot_confusion_matrix,
    save_result,
)
from src.landmarks_spec import FeatureSpec
from src.models import build_model
from src.seeding import set_seed
from src.splits import assert_no_group_leakage, loso_folds, protocol_train_val
from src.train import get_device, predict, train_model

NAME = "exp07_aumento_dados"
MODEL = "lstm_corrected"
PROTOCOLO = "revisado"
CONDICOES = ("sem_aumento", "com_aumento")
EXP05_JSON = "exp05_pipeline_corrigido_loso_revisado.json"
ALFA = 0.05
# Com menos diferencas nao nulas, p < 0,05 e impossivel no Wilcoxon bilateral:
# 5 de 5 do mesmo lado ja dao p = 0,0625.
MIN_NAO_NULAS = 6


def transformacao(spec: FeatureSpec, params: ParamsAumento, gen: torch.Generator,
                  mean: np.ndarray, std: np.ndarray):
    """Aumento seguido da padronizacao com as estatisticas do treino sem aumento."""
    mean_t = torch.as_tensor(mean, dtype=torch.float32)
    std_t = torch.as_tensor(std, dtype=torch.float32)

    def aplicar(xb: torch.Tensor) -> torch.Tensor:
        return (aumentar_lote(xb, spec, params, gen) - mean_t) / std_t

    return aplicar


def acertos(run: dict) -> int:
    return int(np.trace(np.asarray(run["confusion_matrix"])))


def resumo_por_condicao(runs: list[dict], seeds: list[int], chance: float) -> dict:
    """Agregados por condicao e acertos somados nos folds de cada seed."""
    out: dict[str, dict] = {}
    for cond in CONDICOES:
        rs = [r for r in runs if r["condicao"] == cond]
        hits, totais, p_binom = [], [], []
        for seed in seeds:
            do_seed = [r for r in rs if r["seed"] == seed]
            k = sum(acertos(r) for r in do_seed)
            n = sum(r["n_test"] for r in do_seed)
            hits.append(k)
            totais.append(n)
            p_binom.append(binom_sf(k, n, chance))
        acc = [k / n for k, n in zip(hits, totais)]
        out[cond] = {
            "aggregate": aggregate_runs(rs),
            "acertos_por_seed": hits,
            "n_teste_por_seed": totais,
            "acuracia_agregada_por_seed": acc,
            "acuracia_agregada_media": float(np.mean(acc)),
            "p_binomial_por_seed": p_binom,
            "seeds_acima_do_acaso": int(sum(p < ALFA for p in p_binom)),
            "train_acc_final_media": float(np.mean([r["train_acc_final"] for r in rs])),
            "train_acc_limpo_media": float(np.mean([r["train_acc_limpo"] for r in rs])),
        }
    return out


def comparar(resumo: dict, smoke: bool) -> dict:
    """Wilcoxon pareado sobre as diferencas por seed (com menos sem) e leitura (E7)."""
    sem = np.asarray(resumo["sem_aumento"]["acuracia_agregada_por_seed"])
    com = np.asarray(resumo["com_aumento"]["acuracia_agregada_por_seed"])
    k_dif = (np.asarray(resumo["com_aumento"]["acertos_por_seed"])
             - np.asarray(resumo["sem_aumento"]["acertos_por_seed"]))
    dif = com - sem
    nao_nulas = dif[k_dif != 0]
    out = {
        "diferencas_pp": [float(d * 100) for d in dif],
        "diferenca_media_pp": float(dif.mean() * 100),
        "seeds_com_ganho": int((k_dif > 0).sum()),
        "seeds_com_empate": int((k_dif == 0).sum()),
        "seeds_com_perda": int((k_dif < 0).sum()),
        "n_diferencas_nao_nulas": int(len(nao_nulas)),
        "wilcoxon": None,
    }
    if smoke:
        out["leitura"] = "nao aplicavel (smoke)"
        return out
    if len(nao_nulas) < MIN_NAO_NULAS:
        out["wilcoxon"] = {"p": None, "motivo": f"menos de {MIN_NAO_NULAS} diferencas nao nulas"}
        out["leitura"] = "inconclusivo"
        return out

    res = wilcoxon(nao_nulas, zero_method="wilcox", alternative="two-sided")
    p = float(res.pvalue)
    empates = len(np.unique(np.abs(nao_nulas))) < len(nao_nulas)
    out["wilcoxon"] = {
        "estatistica": float(res.statistic),
        "p": p,
        "zero_method": "wilcox",
        "alternative": "two-sided",
        "metodo": (
            "auto: permutacao exaustiva (ha empates entre as diferencas)" if empates
            else "auto: distribuicao exata (sem empates)"
        ),
        "scipy": scipy.__version__,
    }
    if p < ALFA and dif.mean() > 0:
        out["leitura"] = "melhora"
    elif p < ALFA and dif.mean() < 0:
        out["leitura"] = "piora"
    else:
        out["leitura"] = "sem diferenca detectada"
    return out


def verificar_integridade(runs: list[dict], results_dir: Path, class_names: list[str]) -> dict:
    """Compara sem_aumento com o exp05 nas seeds e folds presentes nos dois."""
    path = results_dir / EXP05_JSON
    if not path.exists():
        return {"status": "sem referencia", "arquivo": str(path)}
    with open(path, encoding="utf-8") as f:
        ref = json.load(f)
    if ref.get("smoke"):
        return {"status": "sem referencia", "motivo": "o JSON do exp05 e de smoke"}
    if ref["protocolo"]["classes"] != class_names:
        return {"status": "sem referencia", "motivo": "classes diferentes das do exp05"}

    alvo = {(r["seed"], r["fold"]): r for r in ref["resultados"][MODEL]["runs"]}
    comparados, divergencias = 0, []
    for r in runs:
        e5 = alvo.get((r["seed"], r["fold"]))
        if r["condicao"] != "sem_aumento" or e5 is None:
            continue
        comparados += 1
        if r["confusion_matrix"] != e5["confusion_matrix"]:
            divergencias.append({
                "seed": r["seed"], "fold": r["fold"],
                "acuracia_exp07": r["accuracy"], "acuracia_exp05": e5["accuracy"],
            })
    if divergencias:
        status = "diverge"
    else:
        status = "confere" if comparados else "sem execucoes em comum"
    return {"status": status, "comparados": comparados, "divergencias": divergencias}


def referencia_1nn(results_dir: Path) -> dict | None:
    """Acuracia do 1-NN sem treino (diag_exp05), so para contexto. Nao e rodado aqui."""
    path = results_dir / "diag_exp05.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        knn = json.load(f).get("diag3_knn_loso", {})
    return {nome: {"acertos": r["acertos"], "n": r["n"], "acuracia": r["acuracia"]}
            for nome, r in knn.items()}


def main() -> None:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr_cfg = cfg.training
    params = ParamsAumento.from_config(cfg)
    seeds = args.seeds or [int(s) for s in cfg.raw["aumento"]["seeds"]]
    if args.smoke:
        seeds = seeds[:2]
    epochs = 5 if args.smoke else int(tr_cfg["epochs"])
    parada = stopping_kwargs(PROTOCOLO, tr_cfg, epochs)
    device = get_device(prefer_cuda=not args.cpu)

    banner(
        "EXP07 - aumento de dados no pipeline corrigido",
        f"condicoes: {', '.join(CONDICOES)} | protocolo de treino: {PROTOCOLO}",
    )

    samples = build_dataset(cfg)
    print(samples.describe())
    print(f"\naumento: {params}")

    # A mesma semente de folds do exp05 (a primeira de training.seeds): a ordem dos
    # indices de treino depende dela, e sem isso a verificacao de integridade falha.
    folds = loso_folds(
        samples.y, samples.groups, samples.signers,
        val_size=float(tr_cfg["val_size"]), seed=int(tr_cfg["seeds"][0]),
    )
    if not folds:
        raise SystemExit("nenhum fold LOSO gerado")
    print(f"\nfolds: {len(folds)} | seeds: {seeds} | epocas: {epochs}\n")

    runs: list[dict] = []
    first_cm: list[list[int]] | None = None

    for seed in seeds:
        for fold in folds:
            for condicao in CONDICOES:
                set_seed(seed)
                tr_idx, val_idx = protocol_train_val(fold, PROTOCOLO)
                assert_no_group_leakage(samples.groups, tr_idx, fold.test)
                assert_no_group_leakage(samples.groups, tr_idx, val_idx)

                X_tr, X_val, X_te = standardize(
                    samples.X[tr_idx], samples.X[val_idx], samples.X[fold.test]
                )
                y_tr = samples.y[tr_idx]
                y_val = samples.y[val_idx]
                y_te = samples.y[fold.test]

                model = build_model(
                    MODEL, input_size=samples.dim, num_classes=len(samples.class_names)
                )
                treino = {
                    "batch_size": int(tr_cfg["batch_size"]),
                    "learning_rate": float(tr_cfg["learning_rate"]),
                    "device": device,
                    "verbose": False,
                    **parada,
                }
                if condicao == "sem_aumento":
                    hist = train_model(model, X_tr, y_tr, X_val, y_val, **treino)
                else:
                    # O treino recebe os videos sem padronizar: o aumento atua no espaco
                    # normalizado pelos ombros, e a padronizacao vem depois, no transform.
                    mean, std = standardization_stats(samples.X[tr_idx])
                    gen = torch.Generator().manual_seed(seed)
                    hist = train_model(
                        model, samples.X[tr_idx], y_tr, X_val, y_val,
                        batch_transform=transformacao(samples.spec, params, gen, mean, std),
                        **treino,
                    )

                y_pred = predict(model, X_te, device=device)
                metrics = compute_metrics(y_te, y_pred, samples.class_names)
                # Acerto nos videos de treino originais, sem aumento e sem dropout: a
                # mesma medida nas duas condicoes. O train_acc_final de com_aumento e
                # medido nos lotes aumentados.
                train_acc_limpo = float(np.mean(predict(model, X_tr, device=device) == y_tr))
                print(
                    f"  seed={seed:<5} {fold.name:<17} {condicao:<12} "
                    f"acc={metrics['accuracy'] * 100:6.2f}%  "
                    f"acc_treino_final={hist.train_acc[-1] * 100:6.2f}%  "
                    f"acc_treino_limpo={train_acc_limpo * 100:6.2f}%"
                )
                runs.append({
                    "seed": seed, "fold": fold.name, "condicao": condicao,
                    "train_acc_final": float(hist.train_acc[-1]),
                    "train_acc_limpo": train_acc_limpo,
                    "video_ids_teste": samples.groups[fold.test].tolist(),
                    "predicoes": [samples.class_names[int(i)] for i in y_pred],
                    **metrics,
                })
                if first_cm is None and condicao == "com_aumento":
                    first_cm = metrics["confusion_matrix"]

    chance = 1.0 / len(samples.class_names)
    resumo = resumo_por_condicao(runs, seeds, chance)
    comparacao = comparar(resumo, bool(args.smoke))
    integridade = (
        {"status": "nao aplicavel (smoke)"} if args.smoke
        else verificar_integridade(runs, cfg.results_dir, samples.class_names)
    )
    knn = referencia_1nn(cfg.results_dir)

    if first_cm is not None:
        plot_confusion_matrix(
            first_cm, samples.class_names,
            cfg.results_dir / f"{NAME}_confusao.png",
            title=f"EXP07 com aumento (seed {seeds[0]}, fold 1)",
        )

    banner("EXP07 conclusao")
    print(f"nivel de chance: {chance * 100:.2f}%\n")
    print(
        f"{'condicao':<12} {'acuracia (seed x fold)':>24} {'acertos/seed':>13} "
        f"{'balanceada':>11} {'treino final':>13} {'treino limpo':>13} {'acima do acaso':>15}"
    )
    for cond in CONDICOES:
        r = resumo[cond]
        a = r["aggregate"]["accuracy"]
        n = r["n_teste_por_seed"][0]
        print(
            f"{cond:<12} {a['mean'] * 100:14.2f}% +- {a['std'] * 100:5.2f} "
            f"{np.mean(r['acertos_por_seed']):8.2f}/{n:<4} "
            f"{r['aggregate']['balanced_accuracy']['mean'] * 100:10.2f}% "
            f"{r['train_acc_final_media'] * 100:12.2f}% "
            f"{r['train_acc_limpo_media'] * 100:12.2f}% "
            f"{r['seeds_acima_do_acaso']:>8}/{len(seeds)}"
        )
    w = comparacao["wilcoxon"]
    print(
        f"\ndiferenca media: {comparacao['diferenca_media_pp']:+.2f} pp | "
        f"seeds com ganho/empate/perda: {comparacao['seeds_com_ganho']}/"
        f"{comparacao['seeds_com_empate']}/{comparacao['seeds_com_perda']} | "
        f"Wilcoxon p = {w['p'] if w else None}"
    )
    print(f"leitura: {comparacao['leitura']}")
    if knn:
        print("referencia 1-NN sem treino (diag_exp05): " + ", ".join(
            f"{nome} {r['acertos']}/{r['n']} = {r['acuracia'] * 100:.2f}%"
            for nome, r in knn.items()
        ))
    print(f"integridade (sem_aumento contra o exp05): {integridade['status']}")
    for d in integridade.get("divergencias", []):
        print(
            f"  DIVERGE seed={d['seed']} {d['fold']}: exp07 {d['acuracia_exp07'] * 100:.2f}% "
            f"contra exp05 {d['acuracia_exp05'] * 100:.2f}%"
        )

    # Um modelo final por condicao, com todos os videos e a primeira seed. Roda depois
    # da analise e nao entra nela (docs/06).
    banner("EXP07 modelos finais (todos os videos)")
    def com_aumento(mean: np.ndarray, std: np.ndarray):
        gen = torch.Generator().manual_seed(seeds[0])
        return transformacao(samples.spec, params, gen, mean, std)

    modelos_finais: dict[str, str] = {}
    for condicao in CONDICOES:
        modelo = salvar_modelo_final(
            cfg, samples, f"{NAME}_{condicao}", MODEL, seed=seeds[0],
            treino={
                "batch_size": int(tr_cfg["batch_size"]),
                "learning_rate": float(tr_cfg["learning_rate"]),
                "device": device,
                "verbose": False,
                **parada,
            },
            aggregate=resumo[condicao]["aggregate"],
            info={"experimento": NAME, "protocolo": PROTOCOLO, "condicao": condicao,
                  "resultado": f"{NAME}.json"},
            smoke=bool(args.smoke),
            transform_factory=com_aumento if condicao == "com_aumento" else None,
        )
        modelos_finais[condicao] = modelo.relative_to(cfg.results_dir).as_posix()

    save_result(
        NAME,
        {
            "descricao": "aumento de dados no pipeline corrigido, comparacao pareada",
            "registro_previo": "docs/06_protocolo_exp07_aumento.md",
            "smoke": bool(args.smoke),
            "modelo": MODEL,
            "protocolo": {
                "treino": PROTOCOLO,
                "parada": parada,
                "revisao": "docs/04_revisao_protocolo_exp05.md",
                "folds_loso": len(folds),
                "seed_dos_folds": int(tr_cfg["seeds"][0]),
                "seeds": seeds,
                "epochs": epochs,
                "sequence_length": samples.seq_len,
                "feature_dim": samples.dim,
                "classes": samples.class_names,
            },
            "aumento": {
                "corte_temporal_max": params.corte_temporal_max,
                "rotacao_graus": params.rotacao_graus,
                "escala": list(params.escala),
                "ruido_std": params.ruido_std,
                "aspecto": params.aspecto,
            },
            "versoes": {"torch": torch.__version__, "scipy": scipy.__version__,
                        "numpy": np.__version__},
            "chance_level": chance,
            "resumo": resumo,
            "comparacao": comparacao,
            "integridade": integridade,
            "referencia_1nn_diag_exp05": knn,
            "modelos_finais": modelos_finais,
            "runs": runs,
        },
        cfg.results_dir,
    )


if __name__ == "__main__":
    main()
