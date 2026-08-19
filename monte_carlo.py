import numpy as np
import matplotlib.pyplot as plt

#Paramètres globaux
niveau_initial = 850
decrement_annuel = 50.0
annees = 10
jours_par_an = 252
total_jours = annees * jours_par_an
dt = 1.0 / jours_par_an
d_points = decrement_annuel / jours_par_an
nb_trajectoires = 5
yield_initial = 0.04  

#Paramètres des régimes de marché
fin_periode_1 = 3
fin_periode_2 = 5

#Période 1
r_perf_1 = 0.04
vol_1 = 0.15
croissance_div_1 = 0.04

#Période 2
r_perf_2 = -0.15
vol_2 = 0.35

#Période 3
r_perf_3 = 0.05
vol_3 = 0.18
croissance_div_3 = 0.05

# Chocs sur les dividendes pendant la Période 
var_div_scen1 = 0.40  
var_div_scen2 = 1.00 

def simuler_scenario(var_div, Z_chocs):
    r_perf_dyn = np.zeros(total_jours)
    vol_dyn = np.zeros(total_jours)
    cash_div_dyn = np.zeros(total_jours)
    
    div_cash_depart = niveau_initial * yield_initial 
    
    # Construction des régimes dynamiques
    for t in range(total_jours):
        annee = t // jours_par_an
        if annee < fin_periode_1:
            #Période 1
            r_perf_dyn[t] = r_perf_1
            vol_dyn[t] = vol_1
            cash_div_dyn[t] = div_cash_depart * (1 + croissance_div_1)**annee
        elif annee < fin_periode_2:
            #Période 2
            r_perf_dyn[t] = r_perf_2
            vol_dyn[t] = vol_2
            # Coupe du cash généré à la fin de la période 1
            pic_fin_periode_1 = div_cash_depart * (1 + croissance_div_1)**fin_periode_1
            cash_div_dyn[t] = pic_fin_periode_1 * (1 - var_div)
        else:
            #Période 3
            r_perf_dyn[t] = r_perf_3
            vol_dyn[t] = vol_3
            base_periode_3 = div_cash_depart * (1 + croissance_div_1)**fin_periode_1 * (1 - var_div)
            cash_div_dyn[t] = base_periode_3 * (1 + croissance_div_3)**(annee - fin_periode_2)

    trajectoires_dec = np.zeros((nb_trajectoires, total_jours + 1))
    trajectoires_pr = np.zeros((nb_trajectoires, total_jours + 1))
    trajectoires_dec[:, 0] = niveau_initial
    trajectoires_pr[:, 0] = niveau_initial

    # Indice Price Return 
    chocs = vol_dyn * np.sqrt(dt) * Z_chocs
    evolutions_PR = np.exp((r_perf_dyn - 0.5 * vol_dyn**2) * dt + chocs)
    trajectoires_pr[:, 1:] = niveau_initial * np.cumprod(evolutions_PR, axis=1)

    # Indice Decrement
    for t in range(1, total_jours + 1):
        prev_levels = trajectoires_dec[:, t-1]
        yield_dynamique = np.zeros_like(prev_levels)
        mask = prev_levels > 0
        yield_dynamique[mask] = cash_div_dyn[t-1] / prev_levels[mask]
        
        mu_TR = r_perf_dyn[t-1] + yield_dynamique
        
        evolutions_TR = np.exp((mu_TR - 0.5 * vol_dyn[t-1]**2) * dt + chocs[:, t-1])
        n_niv = prev_levels * evolutions_TR - d_points
        trajectoires_dec[:, t] = np.maximum(0.0, n_niv)
        
    return trajectoires_pr, trajectoires_dec

# Simulation
np.random.seed(42)
Z_commun = np.random.normal(0, 1, size=(nb_trajectoires, total_jours))

# Scénario 1
traj_pr_1, traj_dec_1 = simuler_scenario(var_div=var_div_scen1, Z_chocs=Z_commun)

# Scénario 2
traj_pr_2, traj_dec_2 = simuler_scenario(var_div=var_div_scen2, Z_chocs=Z_commun)


fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
axe_temps = np.linspace(0, annees, total_jours + 1)
pdi_niveau = niveau_initial * 0.50

titres = [
    f"Scénario 1 : Coupe des Divs (-{var_div_scen1*100:.0f}%)",
    f"Scénario 2 : Coupe des Divs (-{var_div_scen2*100:.0f}%)"
]
trajectoires = [(traj_pr_1, traj_dec_1), (traj_pr_2, traj_dec_2)]

for i, ax in enumerate(axs):
    traj_pr, traj_dec = trajectoires[i]
    
    # Régimes de marché
    ax.axvspan(0, fin_periode_1, color='grey', alpha=0.1)
    ax.axvspan(fin_periode_1, fin_periode_2, color='red', alpha=0.1)
    ax.axvspan(fin_periode_2, annees, color='green', alpha=0.1)
    
    trans = ax.get_xaxis_transform()
    
    # Textes des périodes
    text_p1 = f"Période 1\nDrift: {r_perf_1*100:+.0f}%\nVol: {vol_1*100:.0f}%\nDiv: {croissance_div_1*100:+.0f}%"
    ax.text((0 + fin_periode_1)/2, 0.95, text_p1, transform=trans, ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.3'))
    
    var_div_str = f"{-var_div_scen1*100:.0f}%" if i == 0 else f"{-var_div_scen2*100:.0f}%"
    text_p2 = f"Période 2\nDrift: {r_perf_2*100:+.0f}%\nVol: {vol_2*100:.0f}%\nChoc Div: {var_div_str}"
    ax.text((fin_periode_1 + fin_periode_2)/2, 0.95, text_p2, transform=trans, ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.3'))

    text_p3 = f"Période 3\nDrift: {r_perf_3*100:+.0f}%\nVol: {vol_3*100:.0f}%\nDiv: {croissance_div_3*100:+.0f}%"
    ax.text((fin_periode_2 + annees)/2, 0.95, text_p3, transform=trans, ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.3'))

    ax.plot(axe_temps, traj_dec[0], color='blue', linewidth=1.5, alpha=0.8, label='Decrement (-50 pts)')
    ax.plot(axe_temps, traj_dec[1:].T, color='blue', linewidth=1.5, alpha=0.8)

    ax.plot(axe_temps, traj_pr[0], color='green', linewidth=1.5, alpha=0.8, label='Price Return')
    ax.plot(axe_temps, traj_pr[1:].T, color='green', linewidth=1.5, alpha=0.8)

    ax.axhline(pdi_niveau, color='red', linestyle='--', linewidth=2.5, label=f'PDI ({pdi_niveau} pts)')
    
    ax.set_title(titres[i], fontweight='bold')
    ax.set_xlabel("Années")
    if i == 0: ax.set_ylabel("Niveau de l'indice")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, 10)
    ax.set_ylim(bottom=0)
    
    ax.legend(loc='lower left', fontsize=9)

plt.tight_layout()
plt.show()