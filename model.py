# -*- coding: utf-8 -*-
"""
Modelo de ML para prever a final da Copa do Mundo 2026: Espanha x Argentina.

Pipeline:
  1. Carrega ~49 mil jogos internacionais (1872-2026) de martj42/international_results.
  2. Calcula rating Elo cronologicamente (peso por torneio, vantagem de mando,
     ajuste por margem de gols).
  3. Gera features simétricas por jogo: diferença de Elo, forma recente
     (pontos, gols marcados/sofridos nos últimos 10 jogos), mando de campo.
  4. Treina um GradientBoosting (vitória mandante / empate / vitória visitante)
     com validação temporal.
  5. Prevê Espanha x Argentina (campo neutro) e estima o placar mais provável
     via Poisson bivariado com gols esperados derivados do Elo.
"""
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

RNG = 42
np.random.seed(RNG)

# ---------------------------------------------------------------- 1. Dados
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
import os, urllib.request
if not os.path.exists("results.csv"):
    print("Baixando results.csv ...")
    urllib.request.urlretrieve(DATA_URL, "results.csv")

df = pd.read_csv("results.csv", parse_dates=["date"])
played = df.dropna(subset=["home_score", "away_score"]).copy()
played = played.sort_values("date").reset_index(drop=True)
print(f"Jogos disputados: {len(played)} (de {played.date.min().date()} a {played.date.max().date()})")

# Peso do torneio no Elo (K-factor)
def k_factor(t):
    t = t.lower()
    if "fifa world cup qualification" in t: return 40
    if "fifa world cup" in t: return 60
    if any(x in t for x in ["euro", "copa américa", "copa america", "african cup",
                            "afc asian cup", "gold cup", "confederations"]): return 50
    if "friendly" in t: return 20
    return 30

# ------------------------------------------------- 2. Elo + forma recente
elo = defaultdict(lambda: 1500.0)
form = defaultdict(lambda: deque(maxlen=10))   # (pontos, gols_pro, gols_contra)

rows = []
for r in played.itertuples():
    h, a = r.home_team, r.away_team
    eh, ea = elo[h], elo[a]
    home_adv = 0.0 if r.neutral else 80.0

    # forma antes do jogo
    fh, fa = list(form[h]), list(form[a])
    def agg(f):
        if not f: return (1.0, 1.2, 1.2, 0)
        pts = np.mean([x[0] for x in f]); gf = np.mean([x[1] for x in f])
        ga = np.mean([x[2] for x in f]); return (pts, gf, ga, len(f))
    ph, gfh, gah, nh = agg(fh)
    pa, gfa, gaa, na = agg(fa)

    rows.append({
        "date": r.date,
        "elo_diff": (eh + home_adv) - ea,
        "elo_home": eh, "elo_away": ea,
        "neutral": int(r.neutral),
        "form_pts_diff": ph - pa,
        "form_gf_diff": gfh - gfa,
        "form_ga_diff": gah - gaa,
        "is_wc": int("fifa world cup" in r.tournament.lower()
                     and "qualification" not in r.tournament.lower()),
        "hs": r.home_score, "as_": r.away_score,
    })

    # atualização Elo
    exp_h = 1.0 / (1.0 + 10 ** (-((eh + home_adv) - ea) / 400.0))
    res_h = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    margin = abs(r.home_score - r.away_score)
    g = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8.0)
    k = k_factor(r.tournament) * g
    elo[h] = eh + k * (res_h - exp_h)
    elo[a] = ea - k * (res_h - exp_h)

    # atualização forma
    pts_h = 3 if res_h == 1 else (1 if res_h == 0.5 else 0)
    form[h].append((pts_h, r.home_score, r.away_score))
    form[a].append((3 - pts_h if pts_h != 1 else 1, r.away_score, r.home_score))

feat = pd.DataFrame(rows)
feat["y"] = np.select([feat.hs > feat.as_, feat.hs == feat.as_], [2, 1], 0)  # 2=casa, 1=empate, 0=fora

# --------------------------------------- 3. Treino com validação temporal
FEATURES = ["elo_diff", "neutral", "form_pts_diff", "form_gf_diff", "form_ga_diff", "is_wc"]
modern = feat[feat.date >= "1990-01-01"].reset_index(drop=True)
cut = modern.date < "2022-01-01"
Xtr, ytr = modern.loc[cut, FEATURES], modern.loc[cut, "y"]
Xte, yte = modern.loc[~cut, FEATURES], modern.loc[~cut, "y"]

clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, random_state=RNG)
clf.fit(Xtr, ytr)
proba_te = clf.predict_proba(Xte)
print(f"\nValidação temporal (treino <2022, teste 2022-2026, n={len(yte)}):")
print(f"  Acurácia: {accuracy_score(yte, proba_te.argmax(1)):.3f}  |  Log-loss: {log_loss(yte, proba_te):.3f}")
base = max(np.bincount(yte) / len(yte))
print(f"  Baseline (classe majoritária): {base:.3f}")

# Re-treina com todos os dados para a previsão final
clf.fit(modern[FEATURES], modern["y"])

# ------------------------------------------------- 4. Previsão da final
ESP, ARG = "Spain", "Argentina"
def team_form(t):
    f = list(form[t]); return (np.mean([x[0] for x in f]), np.mean([x[1] for x in f]),
                               np.mean([x[2] for x in f]))
p_esp, gf_esp, ga_esp = team_form(ESP)
p_arg, gf_arg, ga_arg = team_form(ARG)

print(f"\nElo atual  — Espanha: {elo[ESP]:.0f}  |  Argentina: {elo[ARG]:.0f}")
print(f"Forma (últimos 10) — Espanha: {p_esp:.1f} pts/j, {gf_esp:.1f} GP, {ga_esp:.1f} GC")
print(f"                   Argentina: {p_arg:.1f} pts/j, {gf_arg:.1f} GP, {ga_arg:.1f} GC")

x_final = pd.DataFrame([{
    "elo_diff": elo[ESP] - elo[ARG],       # campo neutro: sem bônus de mando
    "neutral": 1,
    "form_pts_diff": p_esp - p_arg,
    "form_gf_diff": gf_esp - gf_arg,
    "form_ga_diff": ga_esp - ga_arg,
    "is_wc": 1,
}])[FEATURES]
p = clf.predict_proba(x_final)[0]  # [fora=ARG, empate, casa=ESP]

print("\n=== FINAL DA COPA 2026 — Espanha x Argentina (19/07, East Rutherford) ===")
print(f"  Vitória Espanha (90 min):   {p[2]*100:5.1f}%")
print(f"  Empate (vai à prorrogação): {p[1]*100:5.1f}%")
print(f"  Vitória Argentina (90 min): {p[0]*100:5.1f}%")

# Título (empate resolvido ~50/50 com leve peso pró time de maior Elo via Elo-expectativa)
exp_esp = 1.0 / (1.0 + 10 ** (-(elo[ESP] - elo[ARG]) / 400.0))
print(f"\n  Probabilidade de TÍTULO — Espanha: {(p[2] + p[1]*exp_esp)*100:.1f}%  |  "
      f"Argentina: {(p[0] + p[1]*(1-exp_esp))*100:.1f}%")

# ----------------------------------------- 5. Placar mais provável (Poisson)
# Gols esperados: média de gols em finais de Copa recente ~2.4; divide pela força Elo
total_goals = 2.4
lam_esp = total_goals * exp_esp
lam_arg = total_goals * (1 - exp_esp)
from math import exp as mexp, factorial
def pois(l, k): return mexp(-l) * l**k / factorial(k)
score_p = {(i, j): pois(lam_esp, i) * pois(lam_arg, j) for i in range(6) for j in range(6)}
top = sorted(score_p.items(), key=lambda kv: -kv[1])[:5]
print("\n  Placares mais prováveis (Espanha x Argentina):")
for (i, j), pr in top:
    print(f"    {i} x {j}  —  {pr*100:.1f}%")
