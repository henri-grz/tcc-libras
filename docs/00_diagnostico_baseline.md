# Diagnostico do baseline: Prestes (2024)

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

Representacao correta: 1 timestep = 1 frame = vetor concatenado de todos os landmarks
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
   nem a frames. Cada frame contribui de 42 a 63 pontos, logo praticamente todo frame do
   teste tem irmaos no treino. Sem `stratify`, tambem.
3. As sequencias sao criadas DEPOIS do split, deslizando janela sobre o array ja
   embaralhado. Em `create_sequences`:

```python
label = labels[i + sequence_length - 1]  # rotulo do ultimo ponto da janela
```

Cada "sequencia temporal" e um conjunto de coordenadas de classes diferentes, e o rotulo
e o da ultima linha. Nao existe ordem temporal no tensor.

### C3. Bug de rotulagem na geracao dos arquivos JSON

Local: `processa()`, Apendice A, p. 50.

```python
todos_dados.extend(processar_video(class_path))
cont += 1
# Salvando a cada tres videos em um JSON
if cont == 2:                       # <- a condicao e dois, nao tres
    with open(f"...{classes[num]}.json", "w") as arquivo:
        json.dump(todos_dados, arquivo, indent=2)
```

Quatro problemas:

- O comentario diz tres, a condicao e dois. Com 3 videos por sinal no experimento 1
  (7 sinais, 21 videos), os arquivos sao cortados a cada 2 videos, entao cada JSON
  contem frames de sinais diferentes e o ultimo video nunca e salvo.
- `todos_dados.extend(...)` concatena frames de videos distintos sem separador. Nao ha
  como recuperar onde um video termina e o proximo comeca.
- O nome do arquivo vem de `classes[num]`, com `num` sendo o indice de
  `enumerate(os.listdir(caminho))` sobre a pasta plana de videos. Nao ha correspondencia
  garantida entre esse indice e a classe.
- A lista `classes` e construida de forma inconsistente entre os apendices:
  `sorted(clas_Arq)` no Apendice A (com duplicatas) e `sorted(set(clas_Arq))` no
  Apendice B (sem duplicatas). As duas listas tem tamanhos diferentes.

Na leitura, agrava:

```python
rotulo = 0
for arq in os.listdir(pasta_json):
    ...
    rotulo += 1
```

Assume que a ordem de `os.listdir` coincide com a ordem alfabetica das classes, o que
nao e garantido pelo sistema de arquivos.

---

## Parte B. Falhas secundarias

| # | Falha | Evidencia |
|---|---|---|
| S1 | Dropout inativo em 3 das 4 camadas LSTM | `nn.LSTM(..., dropout=0.3, num_layers=1)` e ignorado pelo PyTorch com aviso. So `lstm2` (`num_layers=2`) aplica. O Quadro 3 afirma dropout 0.3 em todas |
| S2 | Arquitetura implementada diverge do Quadro 3 | `lstm2` e `bidirectional=True, num_layers=2`, o que o Quadro 3 nao menciona |
| S3 | Ausencia de conjunto de validacao | `train_test_split` devolve apenas treino e teste. O teste e usado como reporte final |
| S4 | Early stopping na loss de treino | `early_stopping(running_loss / len(train_loader))`. O texto (p. 32) afirma monitorar acuracia por 5 epocas |
| S5 | Data augmentation e codigo morto | `aumentar_landmarks`, `gerar_ruido_normal` e `gerar_translacao` definidas (p. 58-59) e nunca chamadas em `principal()` |
| S6 | Acuracia por classe e, na verdade, recall | A formula em `class_accuracy` e TP/(TP+FN). Verificavel: nos Quadros 4 e 5 as colunas "Acuracia" e "Recall" sao identicas em todas as linhas |
| S7 | Script publicado nao treina | `carregarModelo = True` fixo (p. 62). Carrega um `.pth` que nao e fornecido |
| S8 | Numero de epocas divergente | Texto afirma 150 epocas no experimento 1; o codigo tem `epochs = 200` |
| S9 | Face declarada nos objetivos, nunca usada | `visibility` comentado (p. 49), nenhum landmark facial salvo no JSON. O objetivo geral (p. 13) menciona posicionamento de cabeca |
| S10 | Sem rigor estatistico | Seed unico (42), execucao unica, sem validacao cruzada, sem intervalo de confianca, sem protocolo independente de sinalizador |
| S11 | Matriz de confusao nao apresentada | O codigo plota (p. 58), o texto nao apresenta nem analisa |
| S12 | Modulo 3 sem implementacao | Figuras 9 e 10 preveem teste em tempo real. Sem codigo, sem experimento. O objetivo "prototipo validado" nao foi cumprido |
| S13 | Queda no experimento 2 mal explicada | 82,83% para 70,76% justificado como "comum ao adicionar dados" (p. 39) |
| S14 | Pre-processamento nao reprodutivel | Remocao da posicao neutra feita manualmente em editor de video, sem critério documentado (p. 29-30) |
| S15 | Escala trivial | 7 e 11 classes contra 1364 termos da V-Librasil |
| S16 | Sem requisitos nem versoes | Sem repositorio, sem `requirements.txt`, sem versao de MediaPipe ou PyTorch, sem o dataset JSON |
| S17 | `mp.solutions.holistic` legado | API depreciada em favor de MediaPipe Tasks. Versao usada nao reportada |
| S18 | Ausencia de baseline trivial | Sem comparacao contra classificador simples, sem ablacao. Comparacao com trabalhos correlatos usa datasets e numeros de classes diferentes |

---

## Parte C. Como cada falha e testada neste TCC

| Falha | Experimento que a evidencia |
|---|---|
| C1 | `exp03_ponto_unico.py` mostra que um classificador sobre um unico ponto 3D chega perto do baseline, logo o modelo temporal nao contribui |
| C2 | `exp02_rotulos_aleatorios.py` embaralha os rotulos no nivel do video e roda o pipeline original. Acuracia alta com rotulos aleatorios e prova de vazamento |
| C2 | `exp04_split_agrupado.py` mantem modelo e representacao originais e apenas troca o split para agrupado por video. A queda mede o vazamento |
| C1 + C2 | `exp05_pipeline_corrigido_loso.py` da o numero honesto |
| C3 | Corrigido na origem por `src/extract_landmarks.py`, que derruba a etapa de JSON agregado e usa um arquivo por video com metadados explicitos |
| S1 a S8 | Documentados no capitulo de analise, corrigidos em `src/models.py` e `src/train.py` |
| S10 | 3 seeds e leave-one-signer-out em `exp05` |
| S18 | `exp03` funciona tambem como baseline trivial |
