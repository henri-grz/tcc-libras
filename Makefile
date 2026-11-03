# Atalhos do projeto. Ver README.md para o contexto e docs/ para a metodologia.
PY ?= python3

.PHONY: help setup synthetic extract test smoke exp01 exp02 exp03 exp04 exp05 exp06 exp07 controles all demo clean-results

help:
	@echo "setup      instala as dependencias"
	@echo "synthetic  gera cache de landmarks sintetico (roda sem os videos)"
	@echo "extract    extrai landmarks dos videos em data/videos (roda uma vez)"
	@echo "test       roda os testes das partes criticas"
	@echo "smoke      roda os 6 experimentos em modo rapido, para validar o codigo"
	@echo "controles  exp01 a exp04, reproducao e experimentos-controle"
	@echo "all        todos os experimentos, execucao completa"
	@echo "demo       interface de demonstracao do modelo treinado (video e webcam)"

setup:
	$(PY) -m pip install -r requirements.txt

synthetic:
	$(PY) -m src.make_synthetic

extract:
	$(PY) -m src.extract_landmarks

test:
	$(PY) tests/test_pipeline.py

smoke: test
	$(PY) -m experiments.exp01_reproduzir_baseline --smoke
	$(PY) -m experiments.exp02_rotulos_aleatorios --smoke
	$(PY) -m experiments.exp03_ponto_unico --smoke
	$(PY) -m experiments.exp04_split_agrupado --smoke
	$(PY) -m experiments.exp05_pipeline_corrigido_loso --smoke
	$(PY) -m experiments.exp06_ablacao_features --smoke

exp01: ; $(PY) -m experiments.exp01_reproduzir_baseline
exp02: ; $(PY) -m experiments.exp02_rotulos_aleatorios
exp03: ; $(PY) -m experiments.exp03_ponto_unico
exp04: ; $(PY) -m experiments.exp04_split_agrupado
exp05: ; $(PY) -m experiments.exp05_pipeline_corrigido_loso
exp06: ; $(PY) -m experiments.exp06_ablacao_features
exp07: ; $(PY) -m experiments.exp07_aumento_dados

controles: exp01 exp02 exp03 exp04
all: controles exp05 exp06 exp07

demo: ; $(PY) -m app.demo

clean-results:
	rm -f results/*.json results/*.png
