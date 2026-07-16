# ⚽ World Cup 2026 Final Predictor

Modelo de Machine Learning para prever o resultado da final da Copa do Mundo de 2026
(**Espanha x Argentina**, 19/07/2026, East Rutherford), usando o histórico de
~49.500 jogos internacionais desde 1872 do dataset
[martj42/international_results](https://github.com/martj42/international_results).

Usa a mesma metodologia do modelo das semifinais
([2026_World_Cup_Semifinals](https://github.com/valeeeeeeeeeeeeee/2026_World_Cup_Semifinals)),
retreinada com os resultados das duas semifinais incluídos (cutoff em
2026-07-19, véspera da final — sem vazamento de dados).

## Metodologia

Três componentes, combinados num ensemble final:

1. **Ratings de força: Elo + Glicko-2** (`src/elo.py`, `src/glicko2.py`) —
   todo o histórico é reprocessado em ordem cronológica por dois sistemas
   complementares: Elo seguindo as regras do
   [World Football Elo](https://www.eloratings.net/about) (K-factor por
   importância da competição, escala por saldo de gols, bônus de mando) e
   **Glicko-2** (Glickman), que além do rating estima o **RD (desvio do
   rating — incerteza)** e a **volatilidade** de cada seleção. O Glicko-2
   não usa margem de gols e reage de forma diferente a sequências recentes,
   dando ao classificador um segundo sinal de força parcialmente
   independente do Elo.
2. **Classificador V/E/D com stacking de 3 bases** (`src/features.py`,
   `src/combine.py`, `src/train_model.py`) — três modelos base:

   - regressão logística multinomial e **CatBoost**, ambos ponderados por
     recência (half-life de 6 anos), sobre features leak-free (diferença e
     nível de Elo, diferença de Glicko-2 e de RD, forma de 10 jogos, dias
     de descanso, importância da competição, mando neutro);
   - probabilidades **V/E/D do modelo Poisson** (matriz de placares a
     partir dos gols esperados) — um sinal só fracamente correlacionado com
     os dois classificadores guiados por ratings, que é a diversidade de
     que uma combinação precisa.

   Um **meta-modelo logístico** combina as três saídas, recebendo os
   **log-odds** das 9 probabilidades base + contexto do jogo (mando neutro,
   importância da competição). Para treinar sem vazamento, as bases são
   reajustadas em janelas cronológicas expansivas (blocos de 2 anos desde
   2000; Poisson numa janela móvel de ~11 anos) e prevêem apenas o bloco
   seguinte — 25.456 exemplos de meta-treino out-of-sample (2000–2026).

   O `train_model.py` avalia quatro combinadores no holdout temporal
   (treino < 2019, validação ≥ 2019) e publica o de menor log-loss
   competitivo:

   | Candidato | Log-loss competitivo |
   |---|---|
   | Logístico sozinho | 0.8438 |
   | CatBoost sozinho | 0.8439 |
   | Poisson V/E/D sozinho | 0.8728 |
   | Blend fixo 0.65/0.35 (baseline das semifinais) | 0.8428 |
   | Média das 3 bases | 0.8440 |
   | **Stacking v2 (deployed)** | **0.8391** |
   | Média stacking + blend | 0.8402 |

   Baseline ingênuo: 1.05. Com a terceira base Poisson, os log-odds e o
   contexto, o stacking passou a **superar o blend fixo** (0.8391 vs
   0.8428) — a versão anterior, com meta-modelo sobre probabilidades cruas
   de duas bases correlacionadas, perdia para o blend. Ressalva: escolher o
   melhor de 4 combinadores no holdout é uma seleção leve; a margem exata
   deve ser lida com esse desconto.
3. **Modelo Poisson de gols** (`src/poisson_model.py`) — forças de
   ataque/defesa estilo Dixon-Coles via regressão Poisson L2-regularizada
   (jogos desde 2015).

Como uma final não pode terminar empatada, `src/predict_final.py` resolve o
empate de duas formas e tira a média:

- **Monte Carlo (50.000 simulações)**: gols do tempo normal ~ Poisson;
  empate vai para prorrogação (taxas reescaladas para 30 min) e, se
  persistir, pênaltis modelados como moeda levemente inclinada pelo Elo.
- **Redistribuição do empate do classificador**: a massa de probabilidade
  de empate em 90 min é redistribuída com a mesma inclinação de pênaltis.

## Resultado — Espanha x Argentina

| Sinal | Espanha | Argentina |
|---|---|---|
| Elo (véspera da final) | 2308 | 2287 |
| Glicko-2 (RD) | 1889 (67) | 1923 (71) |
| Gols esperados (Poisson) | 1.12 | 1.17 |
| 90 minutos (stacking v2) | 34.8% | 37.5% (empate 27.7%) |
| Título — Monte Carlo | 48.8% | 51.2% |
| Título — classificador | 48.9% | 51.1% |
| **Título — ensemble final** | **48.9%** | **51.1%** |

**Campeã prevista: Argentina, por margem estreita.** Elo e Glicko-2
discordam sobre quem é mais forte (Elo dá +21 para a Espanha; Glicko-2, que
ignora margem de gols e reage mais rápido à sequência recente, dá +34 para
a Argentina); o Poisson vê leve vantagem argentina no ataque. O meta-modelo,
vendo os três sinais, pende para a Argentina.
P(prorrogação) = 28.6%, P(pênaltis) = 15.4%.
Placares mais prováveis em 90 min: 1x1 (13.3%), 0x1 (11.9%), 1x0 (11.4%).

Detalhamento completo em `results/final_prediction.json`.

## Comparação de métodos

Log-loss no holdout competitivo (treino < 2019, validação ≥ 2019) e a
previsão que cada método dá para a final (título já resolve o empate via
inclinação de pênaltis por Elo, 51%/49% pró-Espanha):

| Método | Log-loss ↓ | Espanha (90') | Empate | Argentina (90') | Título ESP | Título ARG |
|---|---|---|---|---|---|---|
| Logístico | 0.8438 | 34.4% | 31.8% | 33.8% | 50.6% | 49.4% |
| CatBoost | 0.8439 | 28.8% | 29.9% | 41.2% | 44.1% | 55.9% |
| Poisson V/E/D | 0.8728 | 34.5% | 28.5% | 37.1% | 49.0% | 51.0% |
| Blend fixo 0.65/0.35 | 0.8428 | 32.4% | 31.1% | 36.4% | 48.3% | 51.7% |
| Média das 3 bases | 0.8440 | 32.6% | 30.1% | 37.4% | 47.9% | 52.1% |
| **Stacking v2 (deployed)** | **0.8391** | 34.8% | 27.7% | 37.5% | 48.9% | 51.1% |
| Média stacking + blend | 0.8402 | 33.6% | 29.4% | 37.0% | 48.6% | 51.4% |
| Monte Carlo (simulação Poisson) | — | — | — | — | 48.8% | 51.2% |
| **Ensemble final (MC + stacking)** | — | — | — | — | **48.9%** | **51.1%** |

## Como rodar

```bash
pip install -r requirements.txt
cd src
python train_model.py     # treina e salva o classificador ensemble
python predict_final.py   # roda o pipeline completo e imprime/salva a previsão
```

O `results.csv` é baixado automaticamente na primeira execução (já inclui
os resultados das semifinais: Espanha 2x0 França, Argentina 2x1 Inglaterra).

## Estrutura

```
src/data_io.py         download/carregamento do dataset
src/elo.py             engine de rating Elo
src/glicko2.py         engine de rating Glicko-2 (rating + RD + volatilidade)
src/features.py        gerador cronológico de features (leak-free)
src/combine.py         combinadores (blends, stacking) e Poisson V/E/D
src/train_model.py     treina bases + meta, compara combinadores, publica o melhor
src/poisson_model.py   modelo Poisson de gols estilo Dixon-Coles
src/predict_final.py   pipeline completo -> previsão da final
results/               modelo treinado + métricas + previsão (gerados)
```

## Limitações

- Futebol internacional é intrinsecamente ruidoso; ~61% de acurácia num
  problema de 3 classes é uma vantagem sólida sobre o baseline, não uma
  bola de cristal.
- Fatores de elenco (lesões, suspensões, viagem) não são modelados — apenas
  Elo, forma recente e taxas de gols em nível de seleção.
- A inclinação do shootout por Elo é uma hipótese de modelagem, não ajustada
  em dados históricos de disputas de pênaltis.

## Créditos

Dados: [martj42/international_results](https://github.com/martj42/international_results) (CC0).
Metodologia: [2026_World_Cup_Semifinals](https://github.com/valeeeeeeeeeeeeee/2026_World_Cup_Semifinals).
