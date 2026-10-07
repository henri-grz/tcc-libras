# Revisao do protocolo de treino do exp05 e do exp06

**Registrado em 05/10/2026, depois da primeira rodada do exp05 e dos diagnosticos
`diag_exp05`, e ANTES de rodar o exp05 e o exp06 com o protocolo revisado.** Qualquer
alteracao posterior deve ser datada e justificada na ultima secao.

Esta e uma revisao feita depois de observar um resultado de teste. O documento existe
para deixar isso explicito, registrar a evidencia que motivou a mudanca e fixar o novo
protocolo antes de ve-lo produzir numeros.

## Primeira rodada (protocolo original)

- Cada fold LOSO testa um sinalizador (11 videos). Dos 22 videos dos outros dois,
  5 vao para validacao e 17 para treino.
- Early stopping na loss de validacao, paciencia de 15 epocas, restauracao dos melhores
  pesos.
- Resultado `lstm_corrected`: acuracia 10,10% +- 3,03, contra 9,09% de chance.
- A melhor epoca foi a **primeira em 9 de 9 execucoes** (3 seeds x 3 folds). O modelo
  avaliado foi, na pratica, o modelo sem treino. A matriz de confusao mostra todas as
  predicoes concentradas em 2 classes.

## Evidencia (`diag_exp05`, 05/10/2026)

| Diagnostico | Resultado |
|---|---|
| Sobreajuste: treina e avalia nos 33 videos, 200 epocas | 100% de acerto, atingido na epoca 39 |
| Historico, protocolo original | melhor epoca = 1 em 9/9; teste 10,10% (reproduz o exp05) |
| Historico, 200 epocas sem early stopping, 17 videos | treino 100% em 9/9; teste 29,29% |
| 1-NN em LOSO, sequencia achatada | 10/33 = 30,30%, p = 0,0005 |
| 1-NN em LOSO, media temporal | 11/33 = 33,33%, p = 0,0001 |

Leitura:

1. O pipeline nao tem defeito: a rede decora os 33 videos.
2. A representacao carrega informacao que generaliza para um sinalizador nao visto:
   o 1-NN, que nao treina nada, fica bem acima do acaso.
3. A loss de validacao sobe desde a epoca 1 (de 2,4 para 6 a 12) enquanto a acuracia
   no sinalizador nao visto sobe. A entropia cruzada pune com forca os erros cometidos
   com confianca, e uma rede que decora o treino erra com confianca. Com 5 videos de
   validacao, a loss foi um sinal enganoso.
4. Cada classe que tem um video na validacao fica com um unico exemplo no treino, de
   outro sinalizador. A validacao era, ela propria, um teste entre sinalizadores com
   5 amostras, e ainda tirava quase 1/4 dos dados de treino.
5. Com 17 videos e batch 16, cada epoca tem 2 passos de gradiente. Paciencia de 15
   epocas equivale a cerca de 30 passos, pouco para a rede sair do ponto inicial.

O diagnostico da falha (melhor epoca = 1 em 9/9, sobreajuste a 100%) nao depende do
conjunto de teste.

## Protocolo revisado

1. **Epocas fixas: `training.epochs` = 200.** Esse valor esta no YAML desde a versao
   inicial do repositorio (commit `a2061cb`), antes de qualquer resultado. Nao foi
   escolhido a partir do teste nem das curvas do diagnostico.
2. **Sem conjunto de validacao.** O treino de cada fold usa os 22 videos dos dois
   sinalizadores que nao estao no teste.
3. **Pesos da ultima epoca**, sem restauracao.

Nada mais muda: arquitetura, taxa de aprendizado, batch, seeds, folds, features,
T = 32 e normalizacao continuam os mesmos. O exp06 adota o mesmo protocolo, porque usa
o mesmo treino.

Implementacao: `training.protocolo_loso: revisado` no YAML, `monitor="none"` em
`src/train.py` e `protocol_train_val` em `src/splits.py`.

## Alternativas descartadas

- **Early stopping pela acuracia de validacao.** Com 5 videos, a acuracia anda em
  degraus de 20 pp. E ruido demais para decidir parada.
- **Escolher o numero de epocas pelas curvas do diagnostico.** O diagnostico mediu
  acuracia de teste em cada variante. Escolher epocas a partir dele seria selecionar
  no teste.
- **Validacao cruzada aninhada dentro dos sinalizadores de treino.** Com 2
  sinalizadores, o laco interno treinaria com 1 exemplo por classe, um regime diferente
  do laco externo.

## Limitacoes assumidas

- Sem validacao, nao ha ajuste de hiperparametros: todos sao fixados a priori. A falha
  S3 (ausencia de validacao) nao e corrigida no protocolo revisado. A justificativa e
  que nada e selecionado, e o teste continua sendo usado apenas para o reporte final.
- O numero do diagnostico (29,29%) nao e previsao do resultado revisado. O diagnostico
  treinou com 17 videos e restaurou os pesos de menor loss de treino. O revisado treina
  com 22 e usa os pesos finais.

## O que sera reportado

- Os **dois** resultados do exp05: o do protocolo original (10,10%, com o colapso do
  treino explicado) e o do protocolo revisado, com referencia a este documento.
- O 1-NN em LOSO como referencia sem treino.

## O que fica vedado

Seja qual for o resultado revisado, **nao** se ajustam epocas, taxa de aprendizado,
arquitetura ou regra de parada com base no teste. Qualquer nova mudanca exige registro
datado na secao abaixo, antes de rodar, e todos os resultados anteriores continuam
reportados.

## Como rodar

```bash
python -m experiments.exp05_pipeline_corrigido_loso                        # revisado
python -m experiments.exp05_pipeline_corrigido_loso --protocolo original   # primeira rodada
python -m experiments.exp06_ablacao_features                               # revisado
```

Cada protocolo grava em arquivo proprio: `results/exp05_pipeline_corrigido_loso_revisado.json`
e `results/exp05_pipeline_corrigido_loso_original.json`, e o mesmo para o exp06.

## Alteracoes posteriores

Nenhuma alteracao de protocolo ate o momento.

## Registro de execucoes (06/10/2026)

Esta secao nao muda o protocolo. Registra rodadas e arquivos que as secoes acima nao
mencionam, sem editar o texto registrado em 05/10/2026.

- **Rodada original do exp06.** Antes desta revisao, o exp06 tambem foi rodado com o
  protocolo original (05/10/2026, 14:28), resultado em
  `results/exp06_ablacao_features_original.json`. Todas as configuracoes ficaram no
  nivel do acaso, pelo mesmo colapso do early stopping: maos 8,08% +- 5,46,
  maos + pose 10,10% +- 3,03, so pose 9,09% +- 0,00. As 9 execucoes de maos + pose
  sao identicas as da primeira rodada do exp05, como esperado. Essa rodada nao foi
  citada na versao inicial deste documento e passa a ser reportada junto com a
  revisada, como o exp05.
- **Primeira rodada do exp05.** O arquivo `results/exp05_pipeline_corrigido_loso.json`
  (antes da separacao por protocolo) foi removido. A nova rodada com
  `--protocolo original` (`results/exp05_pipeline_corrigido_loso_original.json`)
  reproduziu as 9 execucoes exatamente, entao o conteudo e o mesmo. O arquivo antigo
  continua no historico do git, no commit `bd80f1b`.

## Identidade dos sinalizadores (07/10/2026)

Esta secao nao muda o protocolo nem os resultados. Corrige a descricao.

A coluna `articulador` (o numero no nome do arquivo, que vem da coluna "Articulador
1/2/3" do site da V-LIBRASIL) nao identifica a pessoa. Por inspecao visual dos 33
videos, feita pelo autor em 07/10/2026, ha 5 pessoas, mapeadas de forma anonima em
`data/pessoas.csv` (P1: 7 videos, P2: 9, P3: 11, P4: 2, P5: 4). Nenhum sinal tem a
mesma pessoa em mais de um video.

Cada pessoa aparece em uma unica coluna (coluna 1: P1 e P5; coluna 2: P2 e P4;
coluna 3: P3). Por isso, os 3 folds usados no exp05 e no exp06 ja eram independentes de
sinalizador: em nenhum fold uma pessoa do teste aparece no treino. Nada foi rodado de
novo. O que muda e o nome: o protocolo e uma validacao agrupada por sinalizador com 3
folds, dois deles deixando de fora duas pessoas, e nao um "leave-one-signer-out"
estrito com 3 sinalizadores. A garantia passa a ser verificada pelo teste
`test_folds_por_coluna_nao_compartilham_pessoas`.
