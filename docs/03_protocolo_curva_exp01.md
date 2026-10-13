# Protocolo da curva de subamostragem do exp01

**Registrado em 17/09/2026, ANTES de observar qualquer resultado alem do ponto de
8.000 janelas.** Este documento existe para impedir escolha post-hoc do ponto de
operacao. Qualquer alteracao posterior deve ser datada e justificada abaixo.

## Motivacao

A reproducao inicial do pipeline original (`exp01` com `max_sequences = 8.000`)
obteve 13,375% de acuracia de teste contra 70,76% relatados por Prestes (2024), com
90,7% de acuracia de TREINO. O diagnostico e subajuste por subamostragem: 8.000 janelas
sao 1,8% das 440.216 que o pipeline original geraria a partir dos 489.129 pontos do
dataset.

O vazamento em si foi confirmado independentemente do modelo, por kNN sobre um unico
ponto 3D: 65,2% com o split embaralhado no nivel do ponto contra 12,3% com split
agrupado por articulador.

## Grade

`max_sequences` in {8.000, 32.000, 128.000, 440.216 (sem subamostragem)}.

Demais parametros inalterados e identicos aos do bloco `baseline_original` do YAML.
Seed 42. `src/baseline_original.py` e `models.py::LSTMOriginal` nao sao tocados.

## Regra de parada (plato)

Percorrer a grade em ordem crescente. **Interromper quando o ganho de acuracia de
teste entre dois pontos consecutivos for menor que 3 pontos percentuais.** O ponto de
operacao adotado para `exp02` e `exp04` e o **primeiro** ponto da grade a partir do
qual o ganho cai abaixo desse limiar.

Se a grade se esgotar sem plato, adota-se o ponto de fidelidade total (sem
subamostragem), e o texto registra que o limiar nao foi atingido.

Justificativa do limiar: 3 pp e menor que a diferenca entre os dois resultados que o
proprio trabalho original reporta (82,83% e 70,76%, 12,07 pp de diferenca), logo e
fino o bastante para nao mascarar efeito relevante, e grosso o bastante para nao
perseguir ruido de seed unica.

## Curva de controle

Em paralelo, kNN (k=1 e k=5) sobre um unico ponto 3D, variando o tamanho do conjunto
de treino na mesma escala. Serve de teto de memorizacao: indica quanto da acuracia
e recuperavel por memorizacao pura da nuvem de pontos, sem nenhum modelo temporal.
A curva da LSTM e lida contra essa referencia.

## O que a curva mede

Nao mede reconhecimento de sinal. Mede com que densidade o vazamento e amostrado.
Esse e o resultado a ser reportado no capitulo 4.

## Resultado (registrado em 05/10/2026)

| `max_sequences` | Acuracia de teste | Ganho sobre o ponto anterior |
|---|---|---|
| 8.000 | 13,38% | |
| 32.000 | 35,06% | +21,68 pp |
| 128.000 | 49,47% | +14,41 pp |
| sem subamostragem (440.016 janelas de treino) | 54,99% | +5,52 pp |

O ganho nunca ficou abaixo de 3 pp: **o limiar de plato nao foi atingido**. Pela regra
acima, adota-se o ponto de fidelidade total, sem subamostragem
(`--max-sequences 0`), para o `exp02` e o `exp04`.

Referencia de controle no mesmo ponto: kNN (k=1) sobre um unico ponto 3D chega a
64,79% com o split embaralhado e a 11,83% com split agrupado por articulador.

Os arquivos dessas rodadas estao em `results/obsoleto/`. Elas usaram os pesos da melhor
epoca e a ordem dos pontos por video; o exp01 sem subamostragem e refeito com os do
original (`05_protocolo_experimento1.md`). O ponto de operacao adotado aqui nao muda.

## Nota sobre o split por articulador (07/10/2026)

A coluna 3 usada como teste no split agrupado da curva kNN corresponde a uma unica
pessoa (P3), que nao aparece nas colunas 1 e 2 (`data/pessoas.csv`). O split agrupado
por articulador e, portanto, independente de pessoa, e os numeros acima continuam
validos.
