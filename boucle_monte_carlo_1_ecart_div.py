import numpy as np
import matplotlib.pyplot as plt

# --- PARAMÈTRES GLOBAUX ---
niveau_initial = 1000
decrement_annuel = 50.0
annees = 10
jours_par_an = 252
total_jours = annees * jours_par_an
dt = 1.0 / jours_par_an
d_points = decrement_annuel / jours_par_an
nb_trajectoires = 10000
pdi_niveau = niveau_initial * 0.50

# --- PARAMÈTRES DES RÉGIMES DE MARCHÉ ---
fin_periode_1 = 3
fin_periode_2 = 5

# Période 1
r_perf_1 = 0.04
vol_1 = 0.15
croissance_div_1 = 0.04

# Période 2
r_perf_2 = -0.15
vol_2 = 0.35

# Période 3
r_perf_3 = 0.05
vol_3 = 0.18
croissance_div_3 = 0.05

# Choc sur les dividendes pendant la Période 2
var_div_scen = 1

def simuler_scenario(var_div, spot_initial, current_yield, Z_chocs):
    """
    Fonction modifiée pour accepter `spot_initial` et `current_yield` en arguments.
    """
    r_perf_dyn = np.zeros(total_jours)
    vol_dyn = np.zeros(total_jours)
    cash_div_dyn = np.zeros(total_jours)
    
    div_cash_depart = spot_initial * current_yield 
    
    # Construction des régimes dynamiques
    for t in range(total_jours):
        annee = t // jours_par_an
        if annee < fin_periode_1:
            # Période 1
            r_perf_dyn[t] = r_perf_1
            vol_dyn[t] = vol_1
            cash_div_dyn[t] = div_cash_depart * (1 + croissance_div_1)**annee
        elif annee < fin_periode_2:
            # Période 2
            r_perf_dyn[t] = r_perf_2
            vol_dyn[t] = vol_2
            # Coupe du cash généré à la fin de la période 1
            pic_fin_periode_1 = div_cash_depart * (1 + croissance_div_1)**fin_periode_1
            cash_div_dyn[t] = pic_fin_periode_1 * (1 - var_div)
        else:
            # Période 3
            r_perf_dyn[t] = r_perf_3
            vol_dyn[t] = vol_3
            base_periode_3 = div_cash_depart * (1 + croissance_div_1)**fin_periode_1 * (1 - var_div)
            cash_div_dyn[t] = base_periode_3 * (1 + croissance_div_3)**(annee - fin_periode_2)

    trajectoires_dec = np.zeros((nb_trajectoires, total_jours + 1))
    trajectoires_pr = np.zeros((nb_trajectoires, total_jours + 1))
    trajectoires_dec[:, 0] = spot_initial
    trajectoires_pr[:, 0] = spot_initial

    # Indice Decrement (on calcule jour par jour pour économiser la mémoire RAM)
    for t in range(1, total_jours + 1):
        z = Z_chocs[:, t-1]
        choc_jour = vol_dyn[t-1] * np.sqrt(dt) * z
        
        prev_levels = trajectoires_dec[:, t-1]
        yield_dynamique = np.zeros_like(prev_levels)
        mask = prev_levels > 0
        yield_dynamique[mask] = cash_div_dyn[t-1] / prev_levels[mask]
        
        # Pour éviter un overflow de l'exponentielle quand prev_levels est proche de 0
        yield_dynamique = np.clip(yield_dynamique, 0.0, 1000.0)
        
        mu_TR = r_perf_dyn[t-1] + yield_dynamique
        
        evol_tr = np.exp((mu_TR - 0.5 * vol_dyn[t-1]**2) * dt + choc_jour)
        n_niv = prev_levels * evol_tr - d_points
        trajectoires_dec[:, t] = np.maximum(0.0, n_niv)
        
        # --- Indice Price Return ---
        evol_pr = np.exp((r_perf_dyn[t-1] - 0.5 * vol_dyn[t-1]**2) * dt + choc_jour)
        trajectoires_pr[:, t] = trajectoires_pr[:, t-1] * evol_pr
        
    return trajectoires_dec, trajectoires_pr

# ==========================================
# BOUCLE SUR LE SPOT INITIAL
# ==========================================
yield_fixe = 0.04 # 4% de yield initial fixe
spots_test = np.linspace(400, 2000, 33) # De 400 à 2000 par pas de 50
probs_pdi_scen = []
probs_pdi_pr = []
ecarts_finaux_crash = []

np.random.seed(42)
print("Génération des chocs aléatoires...")
Z_commun = np.random.normal(0, 1, size=(nb_trajectoires, total_jours))

print(f"Lancement des simulations (Dividende initial = {yield_fixe*100:.1f}%) pour chaque Spot Initial...")
for spot in spots_test:
    # --- Scénario ---
    traj_dec, traj_pr = simuler_scenario(var_div_scen, spot, yield_fixe, Z_commun)
    valeurs_finales_dec = traj_dec[:, -1]
    valeurs_finales_pr = traj_pr[:, -1]
    
    pdi_niveau_dyn = spot * 0.50
    en_dessous_pdi_dec = valeurs_finales_dec < pdi_niveau_dyn
    en_dessous_pdi_pr = valeurs_finales_pr < pdi_niveau_dyn
    
    prob_dec = np.mean(en_dessous_pdi_dec) * 100
    prob_pr = np.mean(en_dessous_pdi_pr) * 100
    
    probs_pdi_scen.append(prob_dec)
    probs_pdi_pr.append(prob_pr)
    
    breakeven_yield_spot = (decrement_annuel / spot) * 100
    ecart_spot = (yield_fixe * 100) - breakeven_yield_spot
    
    # Calcul du niveau moyen du PR et du Decrement pour les trajectoires où le Decrement finit sous le PDI
    if np.any(en_dessous_pdi_dec):
        moy_pr_crash_dec_pts = np.mean(valeurs_finales_pr[en_dessous_pdi_dec])
        moy_pr_crash_dec_pct = (moy_pr_crash_dec_pts / spot) * 100
        
        moy_dec_crash_dec_pts = np.mean(valeurs_finales_dec[en_dessous_pdi_dec])
        moy_dec_crash_dec_pct = (moy_dec_crash_dec_pts / spot) * 100
        
        ecart_final_pct = moy_pr_crash_dec_pct - moy_dec_crash_dec_pct
    else:
        moy_pr_crash_dec_pct = np.nan
        moy_dec_crash_dec_pct = np.nan
        ecart_final_pct = np.nan
        
    ecarts_finaux_crash.append(ecart_final_pct)
        
    print(f"Spot : {spot:4.0f} | Ecart : {ecart_spot:5.1f}% | Prob Dec : {prob_dec:5.1f}% | Prob PR : {prob_pr:5.1f}% | Moy PR (crash) : {moy_pr_crash_dec_pct:4.1f}% | Moy Dec (crash) : {moy_dec_crash_dec_pct:4.1f}%")

# ==========================================
# VISUALISATION : COURBE DE SENSIBILITÉ
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(spots_test, probs_pdi_scen, marker='o', label=f'Indice Decrement (-50 pts)', color='orange', linewidth=2)
ax.plot(spots_test, probs_pdi_pr, linestyle='-', label=f'Indice Price Return (Référence)', color='blue', linewidth=2, alpha=0.7)

# Point d'équilibre où le yield_fixe correspond exactement au prélèvement
spot_breakeven = decrement_annuel / yield_fixe
ax.axvline(x=spot_breakeven, color='black', linestyle='--', linewidth=2, label=f"Breakeven (Spot = {spot_breakeven:.0f} pts)")

# Axe secondaire pour l'écart de performance
ax2 = ax.twinx()
ax2.plot(spots_test, ecarts_finaux_crash, linestyle='--', marker='x', color='red', linewidth=2, label='Sur-perte du Decrement (Moy PR - Moy Dec)')
ax2.set_ylabel("Sur-perte du Decrement (% du Spot initial)", color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Personnalisation de l'axe X pour afficher l'Écart et le Spot
ticks = np.arange(400, 2200, 200) # 400, 600, 800, 1000, 1200...
ax.set_xticks(ticks)
labels = []
for t in ticks:
    breakeven_y = (decrement_annuel / t) * 100
    ecart = (yield_fixe * 100) - breakeven_y
    labels.append(f"{ecart:+.1f}%\n({t:.0f} pts)")
ax.set_xticklabels(labels)

# Esthétique du graphique
ax.set_title(f"Sensibilité au Spot Initial (Dividende initial fixé à {yield_fixe*100:.1f}%) : Probabilité de toucher PDI (50%)", fontweight='bold')
ax.set_xlabel("Écart de Dividende Initial (Niveau du Spot Initial)")
ax.set_ylabel("Probabilité de toucher la barrière PDI (%)")
ax.grid(True, linestyle=':', alpha=0.7)

# Rassembler les légendes des deux axes
lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10, loc='best')

plt.tight_layout()
plt.show()
