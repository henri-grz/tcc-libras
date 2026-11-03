"""Demo do modelo treinado para a apresentacao do TCC (Gradio, roda local).

Duas abas:
- Video: envia um video de um sinal, mostra o video com os landmarks do MediaPipe e as
  classes mais provaveis. Avisa quando o video fez parte do treino do modelo.
- Webcam: landmarks ao vivo e predicao sobre uma janela deslizante em segundos.

Nao e resultado do TCC. O modelo exibido e o final dos experimentos, treinado com
todos os videos; o desempenho esperado em uma pessoa nova e a acuracia LOSO, mostrada
na tela. O modulo de tempo real segue em trabalhos futuros (CLAUDE.md, secao 3).

Uso:
    python -m app.demo
    python -m app.demo --camera 1
    python -m app.demo --config configs/7sinais.yaml
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.processamento import (  # noqa: E402
    JANELA_PADRAO_S,
    MIN_FRACAO_MAOS,
    JanelaDeslizante,
    ProcessadorFrames,
    empilhar,
    fracao_com_mao,
    recortar_aspecto,
    video_do_treino,
)
from src.checkpoint import ModeloTreinado, load_checkpoint  # noqa: E402
from src.config import Config, load_config  # noqa: E402

INTERVALO_PREDICAO_S = 0.5
LARGURA_WEBCAM = 640

CSS = """
.cabecalho h1 { margin-bottom: 0.2rem; }
.cabecalho p { margin-top: 0; opacity: 0.8; }
.pipeline { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; margin: 0.6rem 0 0.2rem; }
.pipeline span { padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.85rem;
                 background: var(--color-accent-soft); border: 1px solid var(--border-color-accent); }
.pipeline b { opacity: 0.5; }
.aviso-treino { padding: 0.8rem 1rem; border-radius: 0.6rem; border-left: 4px solid #d97706;
                background: rgba(217, 119, 6, 0.10); }
"""

# Textos da tela com acentos: sao para a banca, nao para o codigo.
CABECALHO = """
<div class="cabecalho">
  <h1>Reconhecimento de sinais isolados de LIBRAS</h1>
  <p>Demonstração do modelo do pipeline corrigido. {n} sinais da V-LIBRASIL: {sinais}.</p>
  <div class="pipeline">
    <span>vídeo</span><b>&rarr;</b><span>MediaPipe Holistic: pose e mãos</span><b>&rarr;</b>
    <span>normalização pelos ombros</span><b>&rarr;</b><span>{t} frames</span><b>&rarr;</b>
    <span>LSTM</span><b>&rarr;</b><span>sinal</span>
  </div>
</div>
"""


class Modelos:
    """Arquivos .pth de `<results>/modelos/` (sem os de smoke), carregados sob demanda."""

    def __init__(self, cfg: Config):
        pasta = cfg.results_dir / "modelos"
        self.caminhos = {
            p.stem: p for p in sorted(pasta.glob("*.pth")) if not p.stem.endswith("_smoke")
        }
        self._cache: dict[str, ModeloTreinado] = {}

    def padrao(self) -> str | None:
        exp05 = [n for n in self.caminhos if n.startswith("exp05_")]
        return (exp05 or list(self.caminhos) or [None])[0]

    def __getitem__(self, nome: str) -> ModeloTreinado:
        if nome not in self.caminhos:
            raise gr.Error("Escolha um modelo da lista.")
        if nome not in self._cache:
            self._cache[nome] = load_checkpoint(self.caminhos[nome])
        return self._cache[nome]


def resumo_modelo(m: ModeloTreinado) -> str:
    info = m.info
    loso = info.get("avaliacao_loso", {}).get("accuracy")
    acc = f"**{loso['mean'] * 100:.1f}% ± {loso['std'] * 100:.1f}**" if loso else "não registrada"
    variante = info.get("ablacao") or info.get("condicao") or info.get("protocolo", "")
    return (
        f"**Modelo:** {info.get('experimento', '?')} ({variante})  \n"
        f"**Treino:** {len(info.get('videos_treino', []))} vídeos, "
        f"{info.get('epocas_treinadas', '?')} épocas  \n"
        f"**Acurácia esperada em pessoa nova (LOSO):** {acc} "
        f"(acaso: {100 / len(m.class_names):.1f}%)"
    )


def probabilidades_label(m: ModeloTreinado, probs) -> dict[str, float]:
    return {c: float(p) for c, p in zip(m.class_names, probs)}


def construir(cfg: Config, camera: int) -> gr.Blocks:
    modelos = Modelos(cfg)
    if not modelos.caminhos:
        raise SystemExit(
            f"nenhum modelo em {cfg.results_dir / 'modelos'}. Rode o exp05 antes "
            "(python -m experiments.exp05_pipeline_corrigido_loso)."
        )

    def analisar_video(caminho: str | None, nome_modelo: str, progress=gr.Progress()):
        if not caminho:
            raise gr.Error("Envie um vídeo.")
        m = modelos[nome_modelo]
        cap = cv2.VideoCapture(caminho)
        if not cap.isOpened():
            raise gr.Error("Não foi possível abrir o vídeo.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        saida = Path(tempfile.mkdtemp(prefix="libras_demo_")) / "landmarks.webm"
        writer = None
        quadros = []
        try:
            with ProcessadorFrames(m.extracao) as proc:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    arrays, desenho = proc.processar(recortar_aspecto(frame))
                    if writer is None:
                        h, w = desenho.shape[:2]
                        writer = cv2.VideoWriter(
                            str(saida), cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h)
                        )
                    writer.write(cv2.cvtColor(desenho, cv2.COLOR_RGB2BGR))
                    quadros.append(arrays)
                    progress(min(len(quadros) / total, 1.0), desc="MediaPipe")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        if not quadros:
            raise gr.Error("Nenhum quadro lido do vídeo.")

        probs = m.probabilidades(*empilhar(quadros))
        melhor = int(probs.argmax())
        texto = (
            f"### {m.class_names[melhor]} ({probs[melhor] * 100:.1f}%)\n"
            f"{len(quadros)} quadros, {len(quadros) / fps:.1f} s, "
            f"mãos detectadas em {fracao_com_mao(quadros) * 100:.0f}% dos quadros."
        )
        treino = video_do_treino(Path(caminho), m.info.get("videos_treino", []), cfg.videos_dir)
        if treino:
            texto += (
                f'\n\n<div class="aviso-treino"><b>Este vídeo fez parte do treino</b> '
                f"({treino}). O modelo já o viu, então o acerto aqui mede memorização, "
                f"não reconhecimento de um sinalizador novo.</div>"
            )
        return str(saida), probabilidades_label(m, probs), texto

    def webcam(nome_modelo: str, janela_s: float):
        m = modelos[nome_modelo]
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise gr.Error(f"Não foi possível abrir a webcam {camera}. Tente --camera 1.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        janela = JanelaDeslizante(janela_s)
        tempos: list[float] = []
        ultima = 0.0
        try:
            with ProcessadorFrames(m.extracao, largura_exibicao=LARGURA_WEBCAM) as proc:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        raise gr.Error("A webcam parou de enviar quadros.")
                    t = time.monotonic()
                    arrays, desenho = proc.processar(recortar_aspecto(frame))
                    janela.adicionar(t, arrays)
                    tempos = [x for x in tempos if t - x < 1.0] + [t]
                    # Espelhado so na exibicao: o modelo recebe o quadro como no treino.
                    imagem = cv2.flip(desenho, 1)

                    if t - ultima < INTERVALO_PREDICAO_S:
                        yield imagem, gr.skip(), gr.skip()
                        continue
                    ultima = t
                    if janela.duracao() < 0.8 * janela.segundos:
                        label = {}
                        status = f"Preenchendo a janela: {janela.duracao():.1f} de {janela_s:.1f} s"
                    elif fracao_com_mao(janela.quadros()) < MIN_FRACAO_MAOS:
                        label = {}
                        status = "Aguardando sinal: nenhuma mão na janela"
                    else:
                        probs = m.probabilidades(*empilhar(janela.quadros()))
                        label = probabilidades_label(m, probs)
                        status = f"Janela de {janela.duracao():.1f} s"
                    yield imagem, label, f"{status} | {len(tempos)} fps"
        finally:
            cap.release()

    padrao = modelos.padrao()
    m_padrao = modelos[padrao]
    with gr.Blocks(title="LIBRAS: demo do modelo") as demo:
        gr.HTML(CABECALHO.format(
            n=len(m_padrao.class_names), sinais=", ".join(m_padrao.class_names),
            t=m_padrao.sequence_length,
        ))
        with gr.Row():
            seletor = gr.Dropdown(
                list(modelos.caminhos), value=padrao, label="Modelo", scale=2
            )
            info = gr.Markdown(resumo_modelo(m_padrao))
        seletor.change(lambda n: resumo_modelo(modelos[n]), seletor, info)

        with gr.Tabs():
            with gr.Tab("Vídeo"):
                with gr.Row():
                    with gr.Column():
                        entrada = gr.Video(label="Vídeo do sinal", sources=["upload"])
                        analisar = gr.Button("Analisar", variant="primary")
                    with gr.Column():
                        saida_video = gr.Video(label="Landmarks do MediaPipe", autoplay=True,
                                               loop=True, interactive=False)
                with gr.Row():
                    saida_label = gr.Label(label="Sinais mais prováveis", num_top_classes=3)
                    saida_texto = gr.Markdown()
                analisar.click(
                    analisar_video, [entrada, seletor], [saida_video, saida_label, saida_texto]
                )

            with gr.Tab("Webcam"):
                with gr.Row():
                    with gr.Column(scale=3):
                        imagem = gr.Image(label="Webcam", type="numpy", format="jpeg",
                                          interactive=False)
                    with gr.Column(scale=2):
                        webcam_label = gr.Label(label="Sinal na janela", num_top_classes=3)
                        status = gr.Markdown("Clique em **Iniciar** e faça um sinal.")
                        janela = gr.Slider(3, 8, value=JANELA_PADRAO_S, step=0.5,
                                           label="Janela (s)",
                                           info="Mediana da duração dos vídeos de treino: 5,5 s")
                        with gr.Row():
                            iniciar = gr.Button("Iniciar", variant="primary")
                            parar = gr.Button("Parar", variant="stop")
                evento = iniciar.click(webcam, [seletor, janela], [imagem, webcam_label, status])
                parar.click(None, None, None, cancels=[evento])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="caminho do YAML de configuracao")
    ap.add_argument("--camera", type=int, default=0, help="indice da webcam no OpenCV")
    ap.add_argument("--porta", type=int, default=7860)
    args = ap.parse_args()

    demo = construir(load_config(args.config), args.camera)
    demo.queue().launch(
        server_name="127.0.0.1", server_port=args.porta, inbrowser=True, footer_links=[],
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="indigo"), css=CSS,
    )


if __name__ == "__main__":
    main()
