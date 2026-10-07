# Reconhecimento de sinais isolados de LIBRAS: estudo de reprodutibilidade

Codigo do Trabalho de Conclusao de Curso de Henri André Grzegozeski (Bacharelado em Ciencia da
Computacao). Reproduz o pipeline de Prestes (2024), investiga se os resultados
relatados se mantem sob um protocolo de avaliacao sem vazamento de dados, e reavalia o
problema sob um protocolo independente de sinalizador (nenhuma pessoa do teste aparece
no treino).

O diagnostico do trabalho original, com evidencia de codigo, esta em
`docs/00_diagnostico_baseline.md`. Os conceitos usados (generalizacao, vazamento,
unidade de analise) estao em `docs/02_glossario_tecnico.md`.

## Instalacao

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para GPU, instale o PyTorch com CUDA seguindo <https://pytorch.org/get-started/locally/>
em vez do `torch` do requirements.

## Comeco rapido, sem os videos

Da para validar todo o pipeline antes de baixar qualquer dado:

```bash
python -m src.make_synthetic        # cria um cache de landmarks falso
python tests/test_pipeline.py       # testes das partes criticas
make smoke                          # roda os 6 experimentos em modo rapido
```

Nenhum numero gerado a partir do dado sintetico e usado no TCC. E teste de codigo.

## Fluxo real

Os landmarks ja extraidos estao em `data/cache/` (ver secao "Dados"), entao o passo 2
e opcional para quem so quer reproduzir os experimentos.

```bash
# 1. Coloque os videos em data/videos/ com nomes "Sinal_Articulador.mov"
#    exemplo: Abacaxi_1.mov, Abacaxi_2.mov, Abacaxi_3.mov, Abraco_1.mov, ...
#    O numero e a coluna "Articulador" do site da V-LIBRASIL, nao a pessoa
#    (ver "Convencao de nome dos videos"). Ele define os folds do exp05 e do exp06.

# 2. Extraia os landmarks. Roda UMA VEZ e alimenta todos os experimentos.
make extract

# 3. Reproducao do baseline e experimentos-controle
make controles

# 4. Resultado principal e ablacao
make exp05
make exp06
```

Cada experimento grava um JSON em `results/` mais a matriz de confusao em PNG.

## Os experimentos

| Script | Pergunta que responde |
|---|---|
| `exp01_reproduzir_baseline` | O pipeline original reproduz os 70,76% relatados para os 11 sinais? |
| `exp02_rotulos_aleatorios` | Com rotulos sorteados, a acuracia continua alta? Se sim, e vazamento |
| `exp03_ponto_unico` | Um classificador que ve UM ponto 3D chega perto do baseline? Se sim, a LSTM nao contribui |
| `exp04_split_agrupado` | Mesma representacao e mesmo modelo, mudando so o split. Quanto cai? |
| `exp05_pipeline_corrigido_loso` | Qual e o desempenho em sinalizadores que o modelo nunca viu? |
| `exp06_ablacao_features` | Os landmarks de pose agregam sobre os de mao? |

`exp01` a `exp04` formam a reproducao e os controles. `exp05` e o resultado principal.

`diag_exp05` nao e experimento do TCC: e o diagnostico que motivou a revisao do
protocolo de treino do `exp05` e do `exp06` (ver `docs/04_revisao_protocolo_exp05.md`).

## Decisoes de projeto

- **Uma amostra = um video.** Nunca por frame, nunca por landmark.
- **Landmarks extraidos uma unica vez** para `data/cache/`. Todos os experimentos,
  inclusive a reproducao do baseline, leem desse cache.
- **Sem corte manual da posicao neutra**, que nao era reprodutivel. Substituido por
  reamostragem temporal automatica para T frames fixos.
- **NaN explicito para landmark ausente**, convertido em zero mais flag de presenca no
  vetor final.
- **Normalizacao invariante:** origem no ponto medio dos ombros, escala pela distancia
  entre ombros, aplicada por frame antes de qualquer padronizacao estatistica.
- **Split sempre agrupado por video**; para o resultado principal, por sinalizador: 3
  folds, um por coluna da V-LIBRASIL, sem nenhuma pessoa do teste no treino.
- **Epocas fixas definidas a priori no exp05 e no exp06**, sem validacao. A primeira
  rodada usava early stopping na loss de validacao e parou na epoca 1 em todas as
  execucoes. A revisao e a evidencia estao em `docs/04_revisao_protocolo_exp05.md`, e o
  protocolo original continua disponivel com `--protocolo original`.
- **Metricas com o nome correto.** Recall nunca e reportado como acuracia.
- **Seeds fixadas** por `src/seeding.py`; resultado principal e media de 3 seeds com
  desvio padrao.
- **Os mesmos 11 sinais do trabalho original**, para comparabilidade direta.

## Estrutura

```
configs/default.yaml      toda a configuracao, inclusive os parametros do baseline
data/cache/               landmarks extraidos (um .npz por video) e metadata.csv
data/pessoas.csv          mapeamento anonimo video -> pessoa (P1 a P5)
docs/
  00_diagnostico_baseline.md   pontos do trabalho original com evidencia de codigo
  02_glossario_tecnico.md      conceitos com os numeros medidos neste dataset
  03_protocolo_curva_exp01.md  regra do ponto de operacao do exp01, definida a priori
  04_revisao_protocolo_exp05.md  revisao do protocolo de treino do exp05 e do exp06
src/
  config.py               carregamento de configuracao
  seeding.py              fixacao de seed
  landmarks_spec.py       vetor de features por frame, normalizacao, reamostragem
  extract_landmarks.py    MediaPipe Holistic, roda uma vez, gera o cache
  dataset.py              dataset corrigido (uma amostra = um video)
  splits.py               split agrupado por video e por sinalizador
  models.py               LSTMOriginal (replica fiel) e versoes corrigidas
  train.py                laco de treino com early stopping na validacao
  evaluate.py             metricas com nomes corretos, matriz de confusao
  baseline_original.py    reproducao fiel do pipeline original, para exp01 a exp04
  make_synthetic.py       cache falso para teste de fumaca
experiments/              um script por experimento
tests/test_pipeline.py    testes das partes que nao podem quebrar em silencio
results/                  saidas em JSON e PNG
```

## Convencao de nome dos videos

O parser espera `Sinal` seguido de separador e `Articulador`:

```
Abacaxi_1.mov   Abacaxi 2.mov   Abacaxi-3.mov
```

O regex fica em `configs/default.yaml`, chave `extraction.filename_regex`. Arquivos que
nao casam com o padrao, ou cujo sinal nao esta na lista de classes, sao listados e
ignorados na extracao. Confira essa lista: e a defesa contra o problema de rotulagem
descrito em `docs/00_diagnostico_baseline.md` (C3).

O numero depois do sinal e a coluna "Articulador 1/2/3" do site da V-LIBRASIL. **A
coluna nao identifica a pessoa**: nos 33 videos ha 5 pessoas, e o mapeamento anonimo
video -> pessoa (P1 a P5), feito por inspecao visual, esta em `data/pessoas.csv`. Cada
pessoa aparece em uma unica coluna, entao deixar uma coluna de fora nunca coloca no
teste uma pessoa vista no treino. O teste `test_folds_por_coluna_nao_compartilham_pessoas`
verifica isso.

## O que este codigo deliberadamente NAO corrige

`src/baseline_original.py` e `src/models.py::LSTMOriginal` reproduzem o pipeline
original de proposito, inclusive nos pontos discutidos no diagnostico. Sao o baseline
de comparacao. Nao "melhore" esses trechos: sem eles nao ha com o que comparar.

## Dados

Os videos utilizados sao da base **V-LIBRASIL** (RODRIGUES, 2021), 11 sinais x 3
gravacoes, feitas por 5 pessoas. Conforme os termos de uso da base, os videos **nao** sao redistribuidos
neste repositorio e devem ser obtidos diretamente em <https://libras.cin.ufpe.br>.

Este repositorio disponibiliza apenas os landmarks derivados (coordenadas extraidas com
MediaPipe Holistic, em `data/cache/`), para fins de pesquisa e educacao. Uso comercial
nao e permitido. Qualquer publicacao ou apresentacao que utilize esses dados deve citar
a V-LIBRASIL conforme a referencia abaixo.

## Uso de ferramentas de IA

Durante o desenvolvimento foi utilizado um assistente de IA (Claude, da Anthropic) como
ferramenta de apoio a programacao e a revisao da documentacao. A definicao da pergunta
de pesquisa, o desenho dos experimentos, as decisoes metodologicas e a interpretacao dos
resultados sao de responsabilidade do autor, que revisou e validou todo o codigo aqui
publicado.

## Referencias

PRESTES, Willyan Pizarro. *Um sistema de reconhecimento de sinais isolados de LIBRAS
utilizando MediaPipe Holistic e LSTM*. 2024. Trabalho de Conclusao de Curso
(Bacharelado em Sistemas de Informacao) - Faculdade de Ciencias, Universidade Estadual
Paulista (UNESP), Bauru, 2024.

RODRIGUES, Ailton Jose. *V-LIBRASIL: uma base de dados com sinais na Lingua Brasileira
de Sinais (Libras)*. 2021. Dissertacao (Mestrado) - Centro de Informatica, Universidade
Federal de Pernambuco (UFPE), Recife, 2021. Disponivel em: <https://libras.cin.ufpe.br>.
