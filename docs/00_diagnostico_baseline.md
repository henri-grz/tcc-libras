# Diagnostico do pipeline original: Prestes (2024)

Levantamento completo das falhas do trabalho original, com evidencia de codigo.
Este documento e a materia-prima do capitulo de analise do TCC.

Referencias de pagina correspondem ao PDF `prestes_wp_tcc_bauru.pdf`.

---

## Parte A. Falhas criticas

### C1. A unidade de amostra e o landmark individual, nao o frame

Local: `carregar_e_preparar_dados()`, Apendice B, p. 60-61.

```python
for landmark in item['hand_landmarks']:
    for li in landmark:
        x_l = li['x']; y_l = li['y']; z_l = li['z']
        X.append([x_l, y_l, z_l])
        y.append(classes[rotulo])
```

Cada ponto 3D vira uma linha de `X` com rotulo proprio. Em `principal()`:

```python
input_size = X_train_seq.shape[2]   # resulta em 3
```

Consequencia: a LSTM recebe uma sequencia de 200 coordenadas soltas. Nao ha informacao
de qual landmark, qual mao ou qual frame cada coordenada representa. Configuracao de mao
e trajetoria, que sao os parametros linguisticos do sinal, nao estao representados no
tensor de entrada.

Representacao adotada neste trabalho: 1 timestep = 1 frame = vetor concatenado de todos os landmarks
em ordem fixa. Ver `src/landmarks_spec.py`.

### C2. Vazamento de dados em tres niveis empilhados

Local: `carregar_e_preparar_dados()`, p. 61, e `principal()`, p. 62.

```python
X = scaler.fit_transform(X)                                   # (1)
return train_test_split(X, y, test_size=0.1, random_state=42) # (2)
...
X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)  # (3)
```

1. `StandardScaler` ajustado no dataset completo. Media e desvio do conjunto de teste
   entram no pre-processamento.
2. `train_test_split` com `shuffle=True` por padrao, aplicado a **pontos**, nao a videos
   nem a frames. Cada frame contribui com 33, 54 ou 75 pontos (os 33 da pose, mais 21
   por mao detectada; no cache deste projeto, 81,5% dos frames tem 75 e a media e 70,4),
   logo praticamente todo frame do teste tem irmaos no treino. Sem `stratify`, tambem.
3. As sequencias sao criadas DEPOIS do split, deslizando janela sobre o array ja
   embaralhado. Em `create_sequences`:

```python
label = labels[i + sequence_length - 1]  # rotulo do ultimo ponto da janela
```

Cada "sequencia temporal" e um conjunto de coordenadas de classes diferentes, e o rotulo
e o da ultima linha. Nao existe ordem temporal no tensor.

### C3. Descarte silencioso de videos na geracao dos arquivos JSON

Local: `processa()`, Apendice A, p. 50.

```python
todos_dados.extend(processar_video(class_path))
cont += 1
# Salvando a cada tres videos em um JSON
if cont == 2:                       # <- a condicao e dois, nao tres
    with open(f"...{classes[num]}.json", "w") as arquivo:
        json.dump(todos_dados, arquivo, indent=2)
```

**Correcao registrada em 07/10/2026.** A versao anterior desta secao dizia que os JSON
finais misturavam frames de sinais diferentes, com indicio de rotulos trocados. Isso nao
se sustenta. O `processa()` do Apendice A foi executado literalmente, trocando so a
leitura do video, e a logica foi conferida por simulacao: a mistura so existe em
arquivos intermediarios, sobrescritos logo em seguida. O problema real e o descarte
de videos.

Como funciona, com todos os videos em uma pasta plana listada por `os.listdir`:

- O comentario diz tres, a condicao e dois. A cada 2 videos o JSON e gravado com o
  nome `classes[num]`, e a lista acumulada e o contador recomecam. Um arquivo com o
  mesmo nome e sobrescrito.
- O par que atravessa a fronteira entre dois sinais e gravado com o nome do segundo
  sinal e, logo depois, sobrescrito pelo par seguinte desse mesmo sinal. Um video que
  sobra sozinho no fim nunca e gravado.
- Resultado: cada sinal termina com exatamente 2 videos. No Experimento 1 (7 sinais x 3
  videos da V-LIBRASIL) sobram 14 de 21: um terco e descartado. No Experimento 2, cada
  sinal recebeu um quarto video, do acervo do INES (Lira e Souza, 1997), e sobram 22 de
  44: metade e descartada. Quais videos sobram depende do que vinha depois do nome do
  sinal no nome do arquivo, e o trabalho nao informa.
- Os rotulos so ficam corretos por causa da ordem de listagem. `classes` e a lista
  ordenada do sinal de cada arquivo (com duplicatas), e `num` e a posicao do arquivo em
  `os.listdir`. As duas coincidem quando a listagem e alfabetica, como no NTFS do
  Windows usado pelo autor. Em um sistema de arquivos que nao lista em ordem alfabetica,
  cerca de 90% dos videos salvos ficariam com rotulo errado (simulacao com ordem
  aleatoria).
- `todos_dados.extend(...)` concatena frames de videos distintos sem separador. Nao ha
  como recuperar onde um video termina e o proximo comeca.
- A lista `classes` e construida de forma inconsistente entre os apendices:
  `sorted(clas_Arq)` no Apendice A (com duplicatas) e `sorted(set(clas_Arq))` no
  Apendice B (sem duplicatas). As duas listas tem tamanhos diferentes.

Na leitura, a mesma dependencia da ordem de listagem aparece:

```python
rotulo = 0
for arq in os.listdir(pasta_json):
    ...
    rotulo += 1
```

Assume que a ordem de `os.listdir` coincide com a ordem alfabetica das classes, o que
nao e garantido pelo sistema de arquivos.

Consequencia para a reproducao: o exp01 usa os 33 videos da V-LIBRASIL e nao replica o
descarte. O Experimento 2 original foi treinado e testado com 22 videos (2 por sinal),
possivelmente incluindo videos do INES, que nao estao disponiveis aqui.

---

## Parte B. Falhas secundarias

| # | Falha | Evidencia |
|---|---|---|
| S1 | Dropout inativo em 3 das 4 camadas LSTM | `nn.LSTM(..., dropout=0.3, num_layers=1)` e ignorado pelo PyTorch com aviso. So `lstm2` (`num_layers=2`) aplica. O Quadro 3 afirma dropout 0.3 em todas |
| S2 | Arquitetura implementada diverge do Quadro 3 | `lstm2` e `bidirectional=True, num_layers=2`, o que o Quadro 3 nao menciona |
| S3 | Ausencia de conjunto de validacao | `train_test_split` devolve apenas treino e teste. O teste e usado como reporte final |
| S4 | Early stopping na loss de treino | `early_stopping(running_loss / len(train_loader))`. O texto (p. 32) afirma monitorar acuracia por 5 epocas |
| S5 | Data augmentation definida e nao usada | `aumentar_landmarks`, `gerar_ruido_normal` e `gerar_translacao` definidas (p. 58-59) e nao chamadas em `principal()` |
| S6 | A metrica por classe chamada de acuracia e o recall | A formula em `class_accuracy` e TP/(TP+FN). Verificavel: nos Quadros 4 e 5 as colunas "Acuracia" e "Recall" sao identicas em todas as linhas |
| S7 | Script publicado carrega um modelo em vez de treinar | `carregarModelo = True` fixo (p. 62). O `.pth` carregado nao acompanha o trabalho |
| S8 | Numero de epocas divergente | Texto afirma 150 epocas no experimento 1; o codigo tem `epochs = 200` |
| S9 | Face prevista nos objetivos e nao usada | `visibility` comentado (p. 49), nenhum landmark facial salvo no JSON. O objetivo geral (p. 13) menciona posicionamento de cabeca |
| S10 | Execucao unica, sem estimativa de variancia | Seed unico (42), execucao unica, sem validacao cruzada, sem intervalo de confianca, sem protocolo independente de sinalizador |
| S11 | Matriz de confusao nao apresentada | O codigo plota (p. 58), o texto nao apresenta nem analisa |
| S12 | Modulo 3 sem implementacao | Figuras 9 e 10 preveem teste em tempo real, mas o trabalho nao traz codigo nem experimento desse modulo. O objetivo "prototipo validado" fica sem avaliacao |
| S13 | Queda no experimento 2 sem analise de causa | A queda de 82,83% para 70,76% e atribuida a algo "comum ao adicionar dados" (p. 39), sem analise adicional |
| S14 | Pre-processamento manual | Remocao da posicao neutra feita manualmente em editor de video, sem criterio documentado (p. 29-30), o que impede reproduzir o corte |
| S15 | Escala pequena | 7 e 11 classes contra 1364 termos da V-Librasil |
| S16 | Dependencias e versoes nao informadas | Sem repositorio, sem `requirements.txt`, sem versao de MediaPipe ou PyTorch, sem o dataset JSON |
| S17 | `mp.solutions.holistic` legado | API depreciada em favor de MediaPipe Tasks. Versao usada nao reportada |
| S18 | Ausencia de referencia trivial | Sem comparacao contra classificador simples, sem ablacao. Comparacao com trabalhos correlatos usa datasets e numeros de classes diferentes |

---

## Parte C. Como cada falha e testada neste TCC

| Falha | Experimento que a evidencia |
|---|---|
| C1 | Curva de controle kNN: com um unico ponto 3D, o kNN supera a LSTM no mesmo split (64,79% contra 54,99%), e as janelas nao tem ordem temporal. O MLP do `exp03_ponto_unico.py` fica abaixo da LSTM (35,25%) e, sozinho, nao prova C1 (ver docs/02, secao 3) |
| C2 | `exp02_rotulos_aleatorios.py` embaralha os rotulos no nivel do video e roda o pipeline original. Acuracia alta com rotulos aleatorios e prova de vazamento |
| C2 | `exp04_split_agrupado.py` mantem modelo e representacao originais e apenas troca o split para agrupado por video. A queda mede o vazamento |
| C1 + C2 | `exp05_pipeline_corrigido_loso.py` da o desempenho sem vazamento |
| C3 | Corrigido na origem por `src/extract_landmarks.py`, que derruba a etapa de JSON agregado e usa um arquivo por video com metadados explicitos. O descarte de videos nao e replicado no exp01, que usa os 33 videos |
| S1 a S8 | Documentados no capitulo de analise, corrigidos em `src/models.py` e `src/train.py` |
| S10 | 3 seeds e validacao agrupada por sinalizador em `exp05`, sem nenhuma pessoa do teste no treino |
| S18 | `exp03` funciona tambem como referencia trivial |

---

## Nota de 07/10/2026: duas diferencas da reproducao em relacao ao Apendice B

Duas diferencas entre a reproducao do pipeline original (`src/baseline_original.py`)
e o codigo do Apendice B, identificadas em 07/10/2026:

- **Pesos finais.** As primeiras rodadas de 11 sinais restauraram os pesos da melhor
  epoca da loss de treino. O codigo de Prestes (p. 55) so interrompe o laco e fica com
  os pesos da ultima epoca, 5 epocas depois da melhor.
- **Ordem dos pontos.** As primeiras rodadas de 11 sinais empilharam, para cada video,
  todos os frames da mao esquerda, depois da direita e depois da pose. O original
  (p. 60-61) percorre frame a frame: em cada frame, as maos e depois a pose.

Efeito nas primeiras rodadas de 11 sinais:

- a ordem dos pontos, em media, nao afeta exp01, exp02, exp03 e a curva kNN, porque os
  pontos sao os mesmos e sao embaralhados antes do janelamento. Pode afetar o exp04, que
  nao embaralha, porque a ordem muda o conteudo das janelas;
- os pesos finais mudam o modelo avaliado no exp01, no exp02 e no exp04 (melhor epoca
  em vez da ultima, a ate 5 epocas de distancia). O efeito nao foi medido.

As duas configuracoes passaram a usar a ordem e os pesos do original
(`baseline_original.pesos_finais: ultima_epoca` e `ordem_pontos: por_frame`), e as
rodadas de 11 sinais do pipeline original sao refeitas. As primeiras rodadas estao em
`results/obsoleto/` e nao sao usadas no TCC. Ver `docs/05_protocolo_experimento1.md`.
