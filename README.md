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

1. **Elo ratings** (`src/elo.py`) — todo o histórico é reprocessado em ordem
   cronológica seguindo as regras do [World Football Elo](https://www.eloratings.net/about):
   K-factor por importância da competição, escala por saldo de gols e bônus
   de mando para jogos não neutros.
2. **Classificador V/E/D** (`src/features.py`, `src/train_model.py`) — ensemble
   ponderado por recência (half-life de 6 anos) de regressão logística
   multinomial + gradient boosting, blend fixo `0.65*logistic + 0.35*GBM`,
   sobre features leak-free (diferença e nível de Elo, forma de 10 jogos,
   dias de descanso, importância da competição, mando neutro).

   Holdout temporal (treino < 2019, validação ≥ 2019), log-loss no domínio
   competitivo: logístico 0.847, GBM 0.849, **ensemble 0.846 (deployed)**;
   baseline ingênuo 1.05.
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
| Gols esperados (Poisson) | 1.12 | 1.17 |
| 90 minutos (classificador) | 39.0% | 30.1% (empate 30.8%) |
| Título — Monte Carlo | 48.8% | 51.2% |
| Título — classificador | 54.8% | 45.2% |
| **Título — ensemble final** | **51.8%** | **48.2%** |

**Campeão previsto: Espanha — por margem mínima.**
P(prorrogação) = 28.6%, P(pênaltis) = 15.4%.
Placares mais prováveis em 90 min: 1x1 (13.3%), 0x1 (11.9%), 1x0 (11.4%).

Detalhamento completo em `results/final_prediction.json`.

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
src/features.py        gerador cronológico de features (leak-free)
src/train_model.py     treina e seleciona o classificador V/E/D
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
