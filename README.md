# ⚽ World Cup 2026 Final Predictor

Modelo de Machine Learning para prever o resultado da final da Copa do Mundo de 2026
(**Espanha x Argentina**, 19/07/2026, East Rutherford), usando o histórico de
~49.500 jogos internacionais desde 1872 do dataset
[martj42/international_results](https://github.com/martj42/international_results).

## Como funciona

1. **Rating Elo cronológico** — cada seleção tem um rating atualizado jogo a jogo,
   com K-factor maior para jogos de Copa do Mundo, ajuste por margem de gols e
   bônus de mando de campo.
2. **Features de forma** — pontos por jogo e gols marcados/sofridos nos últimos
   10 jogos de cada seleção.
3. **Classificador** — `GradientBoostingClassifier` (scikit-learn) prevendo
   vitória / empate / derrota, com validação temporal (treino até 2021, teste
   2022–2026): **60,1% de acurácia** vs. 47,7% do baseline de classe majoritária.
4. **Placar provável** — modelo de Poisson com gols esperados distribuídos pela
   expectativa Elo.

## Resultado

| Desfecho em 90 min | Probabilidade |
|---|---|
| Vitória da Espanha | 42,3% |
| Empate | 25,0% |
| Vitória da Argentina | 32,7% |

**Probabilidade de título: Espanha 55,5% x 44,5% Argentina.**
Placar mais provável: 1x1.

## Como rodar

```bash
pip install -r requirements.txt
python model.py
```

O script baixa o `results.csv` automaticamente na primeira execução.

## Créditos

Dados: [martj42/international_results](https://github.com/martj42/international_results) (CC0).
