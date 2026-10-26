# Protocolo do exp07: aumento de dados no pipeline corrigido

**Registrado em 26/10/2026, ANTES de rodar o exp07.** Os smoke de validacao de codigo
(2 seeds e 5 epocas, gravados com `"smoke": true`) nao contam como rodada. Este
documento fixa as transformacoes, as seeds e o criterio de leitura antes de ver qualquer
numero. Qualquer alteracao posterior deve ser datada e justificada na secao "Alteracoes
posteriores".

## Pergunta

Aumentar os dados de treino com transformacoes dos landmarks melhora o reconhecimento
em sinalizadores que o modelo nunca viu?

A resposta indica se o aumento compensa em parte o numero pequeno de pessoas gravadas,
ou se isso exige gravar mais pessoas.

## Motivacao e transparencia

O exp07 foi desenhado depois de ver os resultados do exp05 (docs/04 e docs/05).

- **No treino:** o pipeline corrigido decora o treino. A acuracia de treino da ultima
  epoca foi 100% nos 9 treinos de 7 sinais, e o `diag_exp05` de 11 sinais tambem chegou
  a 100%.
- **No teste:** 23,81% +- 10,10 com 7 sinais (nenhuma seed se distingue do acaso) e
  24,24% +- 9,09 com 11 sinais.

A motivacao e o sobreajuste medido no treino. As transformacoes e as intensidades nao
foram escolhidas olhando o teste: sao valores usuais para sequencias de esqueleto,
fixados abaixo antes de qualquer rodada.

## Duas ressalvas

1. O aumento nao cria pessoas novas. Ele gera variacoes dos mesmos videos de treino.
   Ajuda a rede a nao decorar coordenadas exatas, mas so imita em parte a variacao entre
   sinalizadores. A expectativa e modesta.
2. O aumento e aplicado so ao treino, dentro de cada fold, depois da separacao. Se fosse
   feito antes, versoes de um video cairiam no treino e no teste, e o vazamento da falha
   C2 voltaria.

## Desenho

### E1. Comparacao pareada

Duas condicoes, `sem_aumento` e `com_aumento`, iguais em tudo menos nas transformacoes:

- modelo `lstm_corrected`;
- features: maos + pose, T = 32;
- os mesmos 3 folds por coluna;
- protocolo revisado do docs/04: 200 epocas fixas, sem validacao, pesos da ultima
  epoca.

Com a mesma seed, as duas condicoes tem os mesmos pesos iniciais, a mesma ordem de
lotes e as mesmas mascaras de dropout. Os sorteios do aumento usam um
`torch.Generator` proprio, que nao consome o RNG global.

### E2. Aumento a cada lote

Cada vez que um video de treino entra em um lote, recebe uma versao transformada nova.
O numero de amostras por epoca e de passos de gradiente e igual nas duas condicoes.
Assim, um eventual ganho nao se confunde com "mais passos de treino", que e o que
aconteceria com copias fixas do conjunto.

### E3. Politica de aumento

Transformacoes aplicadas nesta ordem (`src/augment.py`):

1. **Corte temporal.** Remove de 0 a 10% do inicio e de 0 a 10% do fim da sequencia,
   com sorteios independentes, e reamostra para T = 32 por interpolacao linear. Simula
   sinais mais rapidos e recortes diferentes.
2. **Rotacao no plano x-y.** Em torno da origem (ponto medio dos ombros), com angulo
   sorteado em [-15, 15] graus.
   - A rotacao e feita em coordenadas isotropicas. Os frames sao redimensionados para
     1024 x 768 na extracao, e o MediaPipe devolve x como fracao da largura e y como
     fracao da altura.
   - Antes de girar, y e convertido para a unidade de x (multiplicado por 768/1024) e
     depois convertido de volta. O aspecto vem de `extraction` no YAML.
   - Sem essa conversao, uma "rotacao" de 15 graus transformaria um angulo reto em um
     de cerca de 98 graus, deformando o sinal.
3. **Escala em x e em y,** com fatores independentes sorteados em [0,9; 1,1]. Simula
   proporcoes de corpo diferentes.
4. **Ruido gaussiano** com desvio 0,01, em unidades da distancia entre ombros, em cada
   coordenada de cada frame.

Regras:

- O angulo e os fatores de escala sao os mesmos em todos os frames da amostra.
- Pontos ausentes (mao nao detectada, gravada como zeros) continuam zero.
- As flags de presenca so mudam pela reamostragem do corte temporal.
- A coordenada z so recebe ruido.
- **Espelhamento fica de fora** (trocar as maos e inverter x). Em Libras a mao dominante
  importa, e a lateralidade das 5 pessoas nao e conhecida.

Os valores estao no bloco `aumento` de `configs/default.yaml` e
`configs/7sinais.yaml`, identicos nas duas configuracoes.

### E4. Padronizacao

O z-score usa as estatisticas do treino sem aumento, as mesmas do exp05. O aumento
acontece no espaco dos landmarks normalizados pelos ombros, e so depois cada lote e
padronizado com essas estatisticas. O teste recebe exatamente a mesma entrada nas duas
condicoes.

### E5. Seeds

Dez seeds, fixadas a priori: 42, 1337, 2024, 0, 1, 2, 3, 4, 5, 6.

- **Verificacao de integridade.** As tres primeiras sao as do exp05. Nelas, a condicao
  `sem_aumento` deve repetir a matriz de confusao do exp05 em cada fold. O resultado e
  gravado no JSON (`integridade`).
- **Por que 10, e nao 3.** A comparacao e entre duas condicoes proximas, e com 3 seeds o
  teste abaixo nunca chegaria a p < 0,05.

### E6. Configuracoes

O exp07 roda nas duas: 7 sinais (Experimento 1) e 11 sinais.

## Criterio de leitura

- **Metrica por seed:** acertos somados nos 3 folds (traco das matrizes de confusao)
  sobre o total de videos de teste: 21 com 7 sinais e 33 com 11.
- **Teste:** Wilcoxon pareado bilateral nas 10 diferencas (com menos sem), com
  `zero_method="wilcox"` e alfa = 0,05.
  - Diferencas iguais a zero saem do teste.
  - O p e exato: distribuicao exata sem empates e permutacao exaustiva com empates
    (`method="auto"` do scipy). O metodo usado e a versao do scipy vao para o JSON.
- **Leitura:**

| Situacao | Leitura |
|---|---|
| Menos de 6 diferencas nao nulas (com 5 de 5 do mesmo lado, p = 0,0625) | `inconclusivo` |
| p < 0,05 e diferenca media maior que zero | `melhora` |
| p < 0,05 e diferenca media menor que zero | `piora` |
| Demais casos | `sem diferenca detectada` |

- **Tambem reportados, fora do criterio:**
  - acuracia balanceada;
  - seeds com ganho, empate e perda;
  - acertos de cada seed com o binomial unilateral contra 1/K, o criterio do docs/05:
    7 de 21 com 7 sinais e 7 de 33 com 11 sinais sao o minimo para p < 0,05;
  - `train_acc_final`: acuracia de treino na ultima epoca, medida nos lotes que a rede
    viu, com dropout. Em `com_aumento`, esses lotes sao os aumentados;
  - `train_acc_limpo`: acuracia nos videos de treino originais, medida depois do treino,
    sem aumento e sem dropout. E a mesma medida nas duas condicoes;
  - a referencia 1-NN sem treino do `diag_exp05`.
- **Limite da inferencia:** as seeds variam so a aleatoriedade do treino. As pessoas de
  teste sao sempre as mesmas 5, entao o resultado diz se o aumento ajuda com essas
  pessoas, nao em geral.
- **Leitura para a expansao:**
  - `melhora`: o aumento de dados e um caminho parcial para compensar poucas gravacoes
    ao acrescentar sinais;
  - `sem diferenca detectada` ou `piora`: com esta politica, o aumento nao substitui
    novos sinalizadores. Outras politicas ficam como trabalho futuro, nao como nova
    rodada.
  - Uma `melhora` sobre `sem_aumento` nao basta para dizer que o sistema reconhece
    sinais: ela e lida junto do binomial contra o acaso e da referencia 1-NN.

## O que fica vedado

Depois de ver os resultados, nao se alteram:

- os parametros do aumento, as seeds ou o criterio de leitura;
- quais transformacoes entram.

Uma ablacao de cada transformacao nao faz parte do exp07. Com 21 ou 33 videos de teste,
escolher transformacoes pelo resultado seria selecao no teste. Qualquer mudanca exige
secao datada em "Alteracoes posteriores", escrita antes de rodar, e todos os resultados
anteriores continuam reportados.

## Limites assumidos

- **Pessoas.** O aumento nao cria pessoas, e a conclusao vale para as 5 pessoas da base.
- **Parametros.** Foram fixados sem ajuste, porque nao ha conjunto de validacao. Outra
  politica poderia ir melhor, mas isso fica como trabalho futuro.
- **Fora do exp07:** espelhamento e ablacao por transformacao.
- **Poder estatistico.** Com 21 ou 33 videos de teste, e baixo. "Sem diferenca
  detectada" nao prova ausencia de efeito.

## Como rodar

Na raiz do repositorio, com o ambiente ativado:

```powershell
python -m experiments.exp07_aumento_dados --config configs/7sinais.yaml
python -m experiments.exp07_aumento_dados
```

- Os resultados ficam em `results/7sinais/exp07_aumento_dados.json` e em
  `results/exp07_aumento_dados.json`, cada um com a matriz de confusao em PNG.
- Sao 60 treinos por configuracao: 10 seeds x 3 folds x 2 condicoes.
- O `--smoke` grava no mesmo arquivo: nao rodar depois da rodada real.

## Alteracoes posteriores

Nenhuma alteracao ate o momento.

## Resultado

A preencher depois das rodadas.
