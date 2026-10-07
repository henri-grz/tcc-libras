# Glossario tecnico: os conceitos que sustentam o TCC

Documento de referencia conceitual. Explica os quatro conceitos de que o trabalho
depende, na ordem em que um leitor precisa deles. Todo numero citado aqui foi medido
no dataset deste projeto, nao copiado de literatura. A origem de cada um esta no
Apendice B.

Publico: o autor, para nao reconstruir o raciocinio a cada sessao, e materia-prima
para o capitulo 2 (fundamentacao) e o capitulo 4 (analise).

---

## 1. Generalizacao: por que existe conjunto de teste

Um modelo so vale se acerta em coisas que **nunca viu**.

Por isso separa-se treino e teste. O conjunto de teste e uma simulacao do futuro:
dados novos, que o modelo nao teve chance de decorar.

A pergunta que decide se a nota vale alguma coisa e: **o que exatamente conta como
"novo"?** Quem responde isso e o *split*, a forma de cortar os dados.

Se o corte for inadequado, o teste deixa de ser novo e a nota perde o significado. Nao importa o
quao bom seja o modelo: o problema esta no instrumento de medida, nao no modelo.

Consequencia para este TCC: a reavaliacao do trabalho original nao passa por criticar a
arquitetura, e sim por verificar se o protocolo de avaliacao mede o que se propoe a
medir.

---

## 2. Vazamento de dados (data leakage)

**Vazamento e quando informacao do teste escorre para o treino.** A prova deixa de
medir capacidade e passa a medir memoria.

### Onde vazou no trabalho de Prestes (2024)

O trabalho nunca decidiu o que era "um exemplo". Na pratica tratou **cada ponto 3D
solto** como exemplo independente.

Um frame de video gera 33, 54 ou 75 landmarks: os 33 da pose, mais 21 por mao
detectada. No cache deste projeto, 81,5% dos frames tem 75 e a media e 70,4. O codigo
juntou os 489.129 pontos dos 33 videos numa tabela unica, **embaralhou**, e separou 10%
para teste. Para um frame qualquer do video `Casa_1`:

```
frame t do video Casa_1  ->  75 pontos
       embaralha e corta 10%
       ~67 pontos vao para TREINO  (rotulados "Casa")
       ~8  pontos vao para TESTE   (rotulados "Casa")
```

Na hora do teste o modelo recebe um ponto e precisa dizer o sinal. Os outros ~67
pontos **do mesmo instante do mesmo video** estao no treino, com a resposta colada.

### Quanto vazou, medido

| medida | valor |
|---|---|
| Pontos de teste cujo video tambem esta no treino | **100%** |
| Vizinho mais proximo no treino vem do **mesmo video** | 63,3% |
| Vizinho mais proximo tem a mesma classe | 65,7% |
| Distancia mediana ate esse vizinho | 0,00701 |
| Escala tipica de uma coordenada | 0,75331 |
| **Proximidade do vizinho em relacao a escala do dado** | **~108x mais perto** |

A ultima linha e a que importa: para cada ponto de teste existe, no treino, um ponto
praticamente em cima dele, cem vezes mais perto do que a dispersao natural dos dados.
Nao e um caso parecido. E quase o mesmo ponto.

**Analogia:** uma prova em que 10% das linhas do gabarito foram arrancadas e viraram
as questoes, enquanto os 90% restantes do mesmo gabarito ficaram com o aluno para
estudar. Tirar 70 nao diz nada sobre ele saber a materia.

### O efeito, isolado de qualquer modelo

kNN sobre um unico ponto 3D (ver Apendice A), variando so o split:

| classificador | split | acuracia |
|---|---|---|
| kNN k=1, 1 ponto 3D | embaralhado no ponto (Prestes) | **64,79%** |
| kNN k=1, 1 ponto 3D | sinalizador segurado fora (coluna 3, pessoa P3) | **11,82%** |
| acaso | --- | 9,09% |

O protocolo original entrega ~65% de graca, antes de qualquer modelo entrar em campo.
O trabalho relata 70,76% com uma LSTM de 4 camadas e 252.915 parametros.

---

## 3. As duas falhas sao diferentes e precisam ser separadas

Confundir as duas e o erro mais facil de cometer na escrita. Cada uma e provada por
um experimento diferente.

### C1. A representacao descarta a informacao temporal

Uma "sequencia de 200" do baseline, medida:

| conteudo de uma janela de 200 passos | |
|---|---|
| Videos distintos | **32,7 de 33**, em media |
| Classes distintas | **11 de 11**, sempre |

Uma janela real, sorteada do dataset deste projeto:

```
passo   0: xyz=[ 0.087 -0.005  0.427]  video=Abacaxi_3   classe=Abacaxi
passo   1: xyz=[-0.624 -0.049  0.434]  video=Esperar_1   classe=Esperar
passo   2: xyz=[-0.022  0.112  0.422]  video=Ocupado_2   classe=Ocupado
passo   3: xyz=[ 0.180 -0.278  0.357]  video=Ano_1       classe=Ano
passo   4: xyz=[ 1.086 -0.612  0.418]  video=Abraco_2    classe=Abraco
passo   5: xyz=[-0.699 -0.355  0.379]  video=Escola_1    classe=Escola
   ...
passo 199: xyz=[ 0.371 -0.752  0.396]  video=Ano_1       classe=Ano

   rotulo da janela inteira = "Ano", so porque o passo 199 calhou de ser Ano
```

Isso nao e uma sequencia temporal. E um saco de coordenadas de 33 videos e 11 sinais
em ordem aleatoria, rotulado pelo ultimo ponto.

Uma LSTM existe para aprender **ordem**: que o gesto vai da posicao A para a B para a
C. Aqui nao ha ordem para aprender. Pior: o tensor nao diz **qual** landmark e cada
ponto. Nao da para saber se `[0.087, -0.005, 0.427]` e a ponta do indicador ou o
ombro esquerdo. Configuracao de mao e trajetoria, que sao os parametros linguisticos
que definem um sinal em LIBRAS, **nao sao representaveis nesse formato**.

### C2. O vazamento

O da secao 2. Independente de C1: mesmo com representacao perfeita, embaralhar no
nivel do ponto contaminaria a avaliacao.

### A distincao em uma frase cada

| | |
|---|---|
| **C1** | A pergunta e impossivel de responder honestamente: a informacao necessaria nao esta na entrada |
| **C2** | ...mas o gabarito esta em cima da mesa |

Juntas produzem acuracia alta sem que o modelo tenha aprendido a reconhecer o sinal.

### Como cada uma e verificada

| experimento | o que muda | falha que prova |
|---|---|---|
| **exp03** | modelo ve 1 ponto so, sem nenhum tempo | **C1**, com ressalva. No ponto de operacao completo, o MLP sobre 1 ponto deu 35,25%, abaixo da LSTM (54,99%): sozinho, o exp03 nao prova C1. A prova vem de duas outras evidencias: o kNN sobre 1 ponto chega a 64,79% e supera a LSTM no mesmo split, e as janelas nao tem ordem temporal (ver C1 acima). O ganho da LSTM sobre o MLP nao pode vir de modelagem temporal; vem de ajustar melhor o mapeamento ponto -> video |
| **exp04** | mantem representacao, modelo e hiperparametros, **troca so o split** | **C2**. Caiu de 54,99% para 14,83% +- 5,98 (-40,16 pp). O que cai e atribuivel ao vazamento e a mais nada |

O exp05 (LOSO) muda as duas coisas ao mesmo tempo, por isso sozinho nao atribui culpa.
Ele da o numero honesto; exp03 e exp04 explicam de onde veio a diferenca.

---

## 4. Unidade de analise: o conceito para levar para a defesa

Se for para guardar uma coisa so deste projeto:

> **Antes de treinar qualquer coisa, decida o que e "um exemplo". E corte os dados
> nessa unidade.**

Prestes tratou um **ponto 3D** como exemplo. A unidade certa e o **video**: e ele que
tem um rotulo verdadeiro, um comeco e um fim.

Para reconhecimento de sinais existe uma unidade ainda mais exigente, que e o
protocolo do exp05: o **sinalizador**. Treinar com algumas pessoas e testar em outras,
que nunca apareceram no treino.

Na V-LIBRASIL, cada sinal tem 3 gravacoes, nas colunas "Articulador 1/2/3" do site.
A coluna nao e a pessoa: os 33 videos tem 5 pessoas, P1 a P5, mapeadas em
`data/pessoas.csv`. Mas cada pessoa aparece em uma unica coluna (coluna 1: P1 e P5;
coluna 2: P2 e P4; coluna 3: P3). Por isso, deixar uma coluna de fora deixa de fora
pessoas que nunca aparecem no treino, e os 3 folds por coluna sao independentes de
sinalizador.
O objetivo nao e reconhecer *o Fulano fazendo "Casa"*, e reconhecer *"Casa"* feito por
qualquer pessoa.

A queda entre protocolos e o resultado do TCC:

| protocolo | acuracia (kNN, 1 ponto) |
|---|---|
| Ponto embaralhado (Prestes) | 64,79% |
| Sinalizador segurado fora (coluna 3, P3) | 11,82% |
| Acaso | 9,09% |

Com a representacao de Prestes (um ponto solto), o numero honesto e praticamente o
acaso. E **mais dados nao movem essa linha**: de 8.000 a 329.448 pontos ela fica plana
em ~12%. Decoreba escala. Entendimento, nessa representacao, nao existe para escalar.

Com a representacao corrigida (um video = uma amostra) e o mesmo principio de
sinalizador segurado fora, o exp05 chega a 24,24% +- 9,09 (protocolo revisado, ver
`04_revisao_protocolo_exp05.md`): acima do acaso, mas longe dos 70,76% publicados e no
mesmo nivel de um 1-NN sem treino (30,30% a 33,33%).

| pontos rotulados no treino | split de Prestes | split agrupado |
|---|---|---|
| 8.000 | 38,87% | 11,98% |
| 32.000 | 49,06% | 12,22% |
| 128.000 | 58,18% | 11,96% |
| 440.216 | 64,79% | 11,82% |

As duas curvas juntas sao a figura central do capitulo 4: uma sobe monotonicamente,
a outra e plana. Essa e a assinatura de memorizacao, nao de aprendizado.

De quebra, isso contradiz a explicacao que o proprio trabalho da para a queda de
82,83% para 70,76% ("comum ao adicionar dados", falha S13): no regime dele, adicionar
dados **sobe** a metrica.

---

## Apendice A. kNN, a regua

**k-vizinhos mais proximos.** O classificador mais simples que existe, e e por isso
que serve aqui.

Ele nao aprende. Guarda todos os exemplos de treino numa tabela. Quando chega um
exemplo novo, procura o exemplo de treino mais parecido e copia a resposta. Com k=1
copia do mais proximo; com k=5 pergunta aos 5 mais proximos e vai na maioria.

E o aluno que nao estudou mas levou o livro inteiro para a prova: para cada questao,
procura o exercicio mais parecido ja resolvido e copia.

**Por que ele entra neste trabalho:** nao como concorrente do modelo, mas como regua.
kNN e memorizacao pura, com zero compreensao e zero nocao de tempo. Logo mede o **teto
da decoreba**: quanto da para acertar so decorando. Qualquer modelo que pontue abaixo
disso, naquele protocolo, nao demonstrou aprendizado nenhum.

Observacao util para o texto: no mesmo split e com os mesmos pontos de treino, o MLP
do exp03 chega a 35,25% onde o kNN chega a 64,79%. O teto do vazamento depende da
classe de modelo. Decorar nuvem de pontos e facil para vizinho mais proximo e dificil para
modelo parametrico.

---

## Apendice B. Origem de cada numero

Nenhum numero deste documento veio de literatura. Todos foram medidos no cache de
landmarks deste projeto (33 videos, 11 sinais, 3 gravacoes por sinal feitas por 5 pessoas, 489.129 pontos).

| numero | origem |
|---|---|
| Curvas kNN nos dois splits | `experiments/exp01_curva_controle_knn.py` -> `results/exp01_curva_controle_knn.json` |
| 35,25% do MLP de ponto unico (todos os pontos) | `experiments/exp03_ponto_unico.py` -> `results/exp03_ponto_unico.json` |
| 33,14% do MLP de ponto unico (200.000 pontos, rodada anterior) | `results/exp03_ponto_unico_200k.json` |
| 54,99% da LSTM sem subamostragem | `experiments/exp01_reproduzir_baseline.py` -> `results/exp01_reproduzir_baseline_completo.json` |
| 14,83% +- 5,98 do split agrupado | `experiments/exp04_split_agrupado.py` -> `results/exp04_split_agrupado.json` |
| 24,24% +- 9,09 do pipeline corrigido | `experiments/exp05_pipeline_corrigido_loso.py` -> `results/exp05_pipeline_corrigido_loso_revisado.json` |
| 30,30% e 33,33% do 1-NN em LOSO | `experiments/diag_exp05.py` -> `results/diag_exp05.json` |
| Composicao das janelas, vizinho mais proximo, distancias | medicao pontual sobre `src/baseline_original.build_pointlevel` com o split de `src/splits.leaky_pointlevel_split` |
| 70,76% e 82,83% | PRESTES (2024), relatados pelo trabalho original |

Os numeros de acuracia da LSTM do baseline dependem do ponto de operacao de
`max_sequences`. A curva nao atingiu o plato e o ponto adotado e o completo, sem
subamostragem (`03_protocolo_curva_exp01.md`, registrado em 05/10/2026). A linha do
exp03 na secao 3 foi revisada em 06/10/2026 com os numeros desse ponto.
