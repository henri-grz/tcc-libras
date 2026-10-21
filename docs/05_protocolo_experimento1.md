# Protocolo do Experimento 1 de Prestes (7 sinais) e da nova rodada de 11 sinais

**Registrado em 13/10/2026, ANTES de rodar qualquer experimento com os 7 sinais e
ANTES da nova rodada de 11 sinais.** Os smoke de validacao de codigo (3 a 5 epocas,
gravados com `"smoke": true`) nao contam como rodada. Este documento fixa a
configuracao, as variantes e os criterios de leitura antes de ver qualquer numero.
Qualquer alteracao posterior deve ser datada e justificada na secao "Alteracoes
posteriores".

## Motivacao

Ate aqui, todos os experimentos usavam os 11 sinais do Experimento 2 de Prestes (2024),
que relata 70,76%. O TCC passa a reproduzir os **dois** experimentos, cada um com os
proprios sinais:

- **Experimento 1 (82,83%): 7 sinais, 21 videos.** E a reproducao exata nos dados:
  Prestes usou so videos da V-LIBRASIL, e temos todos.
- **Experimento 2 (70,76%): 11 sinais, 33 videos.** E uma reproducao aproximada (ver
  "Por que o Experimento 2 e reproduzido sem os videos do INES").

Ao preparar os 7 sinais, foram identificadas duas diferencas entre a nossa reproducao e
o codigo do Apendice B (pesos finais e ordem dos pontos, ver "Correcoes de
fidelidade"). O codigo de Prestes e um so para os dois experimentos, entao as duas
correcoes valem para as duas configuracoes, e as rodadas de 11 sinais do pipeline
original sao refeitas (ver "Nova rodada de 11 sinais").

Todos os experimentos (exp01 a exp06) rodam nas duas configuracoes. Cada numero nosso e
comparado com o experimento de Prestes que tem os mesmos sinais.

## O Experimento 1 de Prestes

| Item | Valor | Onde no PDF |
|---|---|---|
| Sinais | Abacaxi, Abraco, Cafe, Casa, Esperar, Livro, Oi | p. 29 |
| Videos | 3 por sinal, da V-LIBRASIL (21) | p. 30 |
| Epocas | no maximo 150 (o Experimento 2 usou 200) | p. 32 |
| Parada | early stopping com paciencia 5 na loss de treino; parou na epoca 96 | p. 32; Apendice B, p. 55 |
| Batch e taxa de aprendizado | 32 e 0,001 | p. 32 |
| Teste | 10%, `train_test_split(..., random_state=42)` | p. 33; Apendice B, p. 61 |
| Janela | 200 pontos | Apendice B, p. 62 |
| Resultado | acuracia 82,83%; precisao 83,12%, recall 82,83% e F1 82,85% (ponderados); loss 0,0149 | p. 33 |

O resumo do PDF diz 82,77% e o capitulo 4 diz 82,83%. Adota-se 82,83%, e a
inconsistencia fica registrada.

## Dados

- Os landmarks dos 21 videos ja estao em `data/cache/` e somam 293.733 pontos. Nada e
  re-extraido.
- Pessoas nos videos dos 7 sinais, por coluna da V-LIBRASIL (o `articulador` do
  `metadata.csv`):
  - coluna 1: P1 (Abacaxi, Abraco, Cafe, Casa, Oi) e P5 (Esperar, Livro);
  - coluna 2: P2 (6 videos) e P4 (Abraco);
  - coluna 3: P3 (7 videos).

  Cada pessoa aparece em uma coluna so, entao os folds por coluna (exp05 e exp06) nao
  tem pessoa em comum entre treino e teste.

## Duas configuracoes

Decisao do autor (07/10/2026).

| Configuracao | Comparada com | Parametro | Resultados |
|---|---|---|---|
| 7 sinais, 21 videos | Experimento 1 | `--config configs/7sinais.yaml` | `results/7sinais/` |
| 11 sinais, 33 videos | Experimento 2, sem os videos do INES | nenhum (`configs/default.yaml`) | `results/` |

- Cada configuracao segue o experimento de Prestes correspondente. Quando o PDF informa
  um valor diferente para o Experimento 1, vale o valor do Experimento 1.
- `configs/7sinais.yaml` difere de `configs/default.yaml` so em `paths.results`,
  `classes` e `baseline_original.epochs`. Um teste em `tests/test_pipeline.py` garante
  isso.

## Variantes do exp01 (7 sinais)

- **A, todos os videos (21).** A mesma pratica da reproducao de 11 sinais. Grava
  `results/7sinais/exp01_reproduzir_baseline.json`, o arquivo que o exp04 usa para
  calcular a queda.
- **B, os videos que sobrevivem a C3 (14), flag `--c3`.** E o mais proximo do que
  Prestes treinou de fato. Grava `results/7sinais/exp01_reproduzir_baseline_c3.json`.

A variante B simula `processa()` (Apendice A, p. 50), que grava um JSON a cada 2 videos
com o nome `classes[num]` e sobrescreve arquivos de mesmo nome
(`src/baseline_original.py::simular_c3`). Com os 21 videos listados em ordem
alfabetica, sobram 14 videos, todos com o rotulo certo (182.166 pontos):

> Abacaxi 1 e 2, Abraco 2 e 3, Cafe 1 e 2, Casa 2 e 3, Esperar 1 e 2, Livro 2 e 3, Oi 1 e 2

Isso supoe que os arquivos se chamavam `Sinal N.mov`, com N = coluna da V-LIBRASIL. O
espaco depois do sinal e obrigatorio, porque o codigo extrai o sinal com `split()[0]`. A
suposicao e o numero depois do espaco. Se Prestes nomeou os arquivos de outro jeito, os
sobreviventes sao outros: ha 3^7 = 2.187 combinacoes possiveis de 2 videos por sinal.

A variante B so e rodada com 7 sinais. Com 11 sinais, a C3 deixaria 22 dos 44 videos
originais, mas os 11 videos do INES nao estao disponiveis, entao nao ha como saber quais
sobrariam.

## Correcoes de fidelidade e epocas

Nas duas configuracoes:

- **Pesos finais (`baseline_original.pesos_finais: ultima_epoca`).** O nosso
  `train_model` com `monitor="train_loss"` restaurava os pesos da melhor epoca. O
  codigo de Prestes (Apendice B, p. 55) so faz `break` e fica com os pesos da ultima
  epoca, 5 epocas depois da melhor.
- **Ordem dos pontos (`baseline_original.ordem_pontos: por_frame`).** As rodadas
  anteriores empilhavam, para cada video, todos os frames da mao esquerda, depois todos
  os da direita e depois toda a pose. O original (Apendice B, p. 60-61) percorre frame
  a frame: em cada frame, as maos e depois a pose.
  - Onde ha embaralhamento no nivel do ponto (exp01, exp02, exp03, curva kNN), a ordem
    nao muda nada em media: os pontos sao os mesmos.
  - No exp04, que nao embaralha, ela muda o conteudo das janelas.

Epocas do pipeline original (exp01, exp02, exp04): no maximo 150 com 7 sinais, como no
Experimento 1, e 200 com 11 sinais, como no Experimento 2.

As 200 epocas do exp05 e do exp06 nao sao de Prestes: sao do nosso pipeline corrigido
(docs/04). Por isso ficam iguais nas duas configuracoes.

## Nova rodada de 11 sinais

Decisao do autor (13/10/2026): as rodadas de 11 sinais do pipeline original sao
refeitas com as duas correcoes de fidelidade. Os resultados anteriores, feitos com
`melhor_epoca` e `por_video`, foram movidos para `results/obsoleto/` e deixam de ser
usados no TCC.

| Experimento | O que acontece |
|---|---|
| exp01 (todos os videos), exp02, exp03, exp04, curva kNN | Rodados de novo. Usam `build_pointlevel`, entao a ordem dos pontos os afeta; exp01, exp02 e exp04 tambem usam os pesos finais |
| exp05, exp06, `diag_exp05` | Nao sao rodados de novo. Nao usam o pipeline original, e as duas chaves nao mudam nada neles. Os resultados continuam em `results/` |
| Curva da LSTM (exp01 com 8.000, 32.000 e 128.000 janelas) | Nao e refeita. A curva serviu para escolher o ponto de operacao (docs/03), e o ponto adotado, sem subamostragem, e tambem o do original, que gera uma janela por ponto. A rodada anterior fica em `results/obsoleto/` como registro dessa decisao |

O novo exp01 grava `results/exp01_reproduzir_baseline.json` (sem o sufixo `_completo`
das rodadas anteriores). Para reproduzir as rodadas anteriores, basta usar
`pesos_finais: melhor_epoca` e `ordem_pontos: por_video` no `default.yaml`.

## Ponto de operacao

Sem subamostragem (`max_sequences: 0`, `max_points: 0`), nas duas configuracoes, e sem
curva da LSTM. Com 11 sinais a curva nao chegou a plato (docs/03), e com 7 sinais o
conjunto completo e menor:

| Configuracao | Janelas de treino | Janelas de teste |
|---|---|---|
| 7 sinais, variante A (21 videos) | 264.160 | 29.173 |
| 7 sinais, variante B (14 videos) | 163.749 | 18.017 |
| 11 sinais (33 videos) | 440.016 | 48.713 |

## Protocolo de exp02 a exp06

Com 7 sinais, todos usam os 21 videos (variante A).

- **exp02 e exp04.** Pipeline original com os parametros de cada configuracao (150 ou
  200 epocas, pesos da ultima epoca, ordem por frame), seeds 42, 1337 e 2024. Com 7
  sinais, o exp04 separa 14 videos de treino e 7 de teste por seed.
- **exp03 e curva kNN.** Sem mudanca de codigo; a ordem dos pontos entra pelo
  `build_pointlevel`.
- **exp05 e exp06 (so 7 sinais).** Protocolo revisado do docs/04, sem mudanca:
  - 200 epocas fixas, sem validacao, pesos da ultima epoca;
  - 3 seeds, 3 folds por coluna, T = 32;
  - so `lstm_corrected` no exp05, como na rodada de 11 sinais.

Com 7 sinais, cada fold do exp05 e do exp06 treina com 14 videos. Com batch 16, isso e
1 passo de gradiente por epoca, ou 200 passos no total, contra 400 com 11 sinais. Nada
e ajustado por isso. A acuracia de treino da ultima epoca passa a ser gravada em cada
rodada (`train_acc_final`), para detectar subajuste, e e apenas reportada.

## Criterios de leitura

- O nivel de chance e 1/7 = 14,29% com 7 sinais e 1/11 = 9,09% com 11 sinais.
- **exp05 e exp06 (7 sinais).** Cada seed tem 21 predicoes de teste (3 folds x 7). O
  teste e o binomial unilateral contra 1/7:
  - 6 acertos (28,6%) dao p = 0,068;
  - 7 acertos (33,3%) dao p = 0,0225, e sao o minimo para p < 0,05.
- **exp02.** O vazamento e confirmado se a acuracia media for mais de 1,5 vez a chance
  (criterio que ja esta no codigo).
- **exp01.** A diferenca para 82,83% (7 sinais) ou 70,76% (11 sinais) e reportada junto
  com as diferencas conhecidas:
  - o corte manual da posicao neutra, nao reproduzido (usa-se o video inteiro);
  - as versoes de bibliotecas, que Prestes nao informa (MediaPipe, PyTorch,
    scikit-learn);
  - na variante B, a suposicao de nomes;
  - com 11 sinais, a ausencia dos videos do INES e o descarte da C3, nao replicado.

  Nenhum valor-alvo e usado para escolher configuracao.

## O que fica vedado

Depois de ver resultados, nao se mudam epocas, seeds, `max_sequences`, variante,
suposicao da C3 nem protocolo, em nenhuma das duas configuracoes. Qualquer mudanca
exige uma secao datada em "Alteracoes posteriores", escrita antes de rodar, e todos os
resultados anteriores continuam reportados.

## Por que o Experimento 2 e reproduzido sem os videos do INES

Prestes acrescentou ao Experimento 2 um video do dicionario do INES por sinal (Lira e
Souza, 1997). Esses videos nao sao usados aqui:

- sao de outra fonte, gravados em condicoes diferentes da V-LIBRASIL;
- Prestes cortou frames de inicio e fim a mao (p. 29-30), e esses cortes nao sao
  reproduziveis;
- a C3 descartava metade dos 44 videos, sem que se saiba quais.

Por isso a reproducao de 11 sinais e aproximada, e a de 7 sinais e a exata.

## Como rodar

Na raiz do repositorio, com o ambiente ativado.

Nova rodada de 11 sinais (configuracao padrao):

```powershell
python -m experiments.exp01_reproduzir_baseline
python -m experiments.exp04_split_agrupado
python -m experiments.exp02_rotulos_aleatorios
python -m experiments.exp03_ponto_unico
python -m experiments.exp01_curva_controle_knn
```

7 sinais:

```powershell
$C = "configs/7sinais.yaml"
python -m experiments.exp05_pipeline_corrigido_loso --config $C --models lstm_corrected
python -m experiments.exp06_ablacao_features --config $C
python -m experiments.exp01_reproduzir_baseline --config $C
python -m experiments.exp01_reproduzir_baseline --config $C --c3
python -m experiments.exp04_split_agrupado --config $C
python -m experiments.exp02_rotulos_aleatorios --config $C
python -m experiments.exp03_ponto_unico --config $C
python -m experiments.exp01_curva_controle_knn --config $C
python -m experiments.diag_exp05 --config $C
```

- Em cada configuracao, o exp04 vem depois do exp01 (todos os videos) porque usa o JSON
  dele para calcular a queda.
- O `diag_exp05` e opcional: da a referencia 1-NN sem treino.
- Os JSON em `results/` e `results/7sinais/` sao o registro. No fim, todos devem ter
  `"smoke": false`.

## Alteracoes posteriores

### 20/10/2026: split no nivel do ponto identico ao `train_test_split`

Registrado antes de rodar, com 7 sinais, o exp01 (todos os videos), o exp02, o exp03 e
a curva kNN.

`src/splits.py::leaky_pointlevel_split` arredondava o tamanho do teste para o inteiro
mais proximo; o `train_test_split` usado por Prestes (Apendice B, p. 61) arredonda para
cima. Com 7 sinais e todos os videos (293.733 pontos), isso colocava no treino 1 ponto
que o original poria no teste. A funcao passa a chamar o proprio `train_test_split`.

- Com 11 sinais e na variante B (`--c3`), a particao e identica antes e depois da
  mudanca. Nenhum resultado ja rodado e afetado.
- Com 7 sinais e todos os videos, as janelas passam a 264.159 de treino e 29.174 de
  teste (a tabela "Ponto de operacao" registrava 264.160 e 29.173).
- O teste `test_split_do_pipeline_original_igual_ao_de_prestes` garante a igualdade.

## Resultado

A preencher depois das rodadas.
