import numpy as np
import plotly.graph_objects as go
import gc
from plotly.subplots import make_subplots
from monte_carlo_2 import MarketScenario, DecrementIndex, AutocallProduct, SimulationEngine

def main():
    # --- PARAMÈTRES GLOBAUX ---
    decrement_annuel = 50.0
    annees = 10
    jours_par_an = 252
    yield_fixe = 0.04 # 4% de yield initial fixe
    nb_trajectoires = 10000
    
    # --- CONFIGURATION DU STRESS TEST ---
    # Coupe des dividendes de -100% en période 2 et 3
    mes_regimes = [
        {"duree_annees": 3, "r_perf": 0.04, "vol": 0.15, "yield_initial": yield_fixe, "croiss_div": 0.04},
        {"duree_annees": 2, "r_perf": -0.15, "vol": 0.35, "yield_initial": 0.0, "croiss_div": 0.0},
        {"duree_annees": 5, "r_perf": 0.05, "vol": 0.18, "yield_initial": 0.05, "croiss_div": 0.05}
    ]

    spots_test = np.linspace(400, 2000, 33) # De 400 à 2000 par pas de 50
    
    # Variables de stockage pour les graphiques
    probs_pdi_dec = []
    probs_pdi_pr = []
    probs_rappel = []
    ecarts_finaux_crash = []
    
    print(f"Lancement de l'étude de sensibilité (Dividende initial = {yield_fixe*100:.1f}%) sur {len(spots_test)} Spots...")
    print("--------------------------------------------------")

    for spot in spots_test:
        print(f"\n=======================================================")
        print(f"SIMULATION POUR SPOT INITIAL = {spot:.0f} pts")
        print(f"=======================================================\n")
        
        # Initialisation des objets
        scenario_krach = MarketScenario(config_regimes=mes_regimes, annees=annees, jours_par_an=jours_par_an)
        mon_indice_dec = DecrementIndex(niveau_initial=spot, decrement_annuel=decrement_annuel)
        
        pdi_niveau_dyn = spot * 0.50
        mon_autocall = AutocallProduct(barriere_rappel=spot, niveau_pdi=pdi_niveau_dyn, non_call_period_mois=11, frequence_obs_mois=4)
        
        # On garde un seed fixe pour avoir des courbes de sensibilité "lisses" entre chaque test de spot
        moteur = SimulationEngine(nb_trajectoires=nb_trajectoires, seed=42)
        
        # Lancement de la simulation
        traj_pr, traj_dec, est_rappele = moteur.run(mon_indice_dec, scenario_krach, mon_autocall)
        
        # Affichage des statistiques complet (tel que demandé)
        nom_scenario = f"Spot {spot:.0f} pts"
        moteur.afficher_statistiques(nom_scenario, traj_pr, traj_dec, est_rappele, mon_autocall, scenario_krach)
        
        # Calculs supplémentaires pour le graphique de sensibilité
        valeurs_finales_dec = traj_dec[:, -1]
        valeurs_finales_pr = traj_pr[:, -1]
        
        en_dessous_pdi_dec = (valeurs_finales_dec < pdi_niveau_dyn) & (~est_rappele)
        en_dessous_pdi_pr = (valeurs_finales_pr < pdi_niveau_dyn) & (~est_rappele)
        
        prob_dec = np.mean(en_dessous_pdi_dec) * 100
        prob_pr = np.mean(en_dessous_pdi_pr) * 100
        prob_rappel = np.mean(est_rappele) * 100
        
        probs_pdi_dec.append(prob_dec)
        probs_pdi_pr.append(prob_pr)
        probs_rappel.append(prob_rappel)
        
        # Sur-perte en cas de crash
        if np.any(en_dessous_pdi_dec):
            moy_pr_crash_pct = (np.mean(valeurs_finales_pr[en_dessous_pdi_dec]) / spot) * 100
            moy_dec_crash_pct = (np.mean(valeurs_finales_dec[en_dessous_pdi_dec]) / spot) * 100
            ecart_final_pct = moy_pr_crash_pct - moy_dec_crash_pct
        else:
            ecart_final_pct = np.nan
            
        ecarts_finaux_crash.append(ecart_final_pct)
        
        # Libération explicite de la mémoire pour éviter un MemoryError
        del traj_pr, traj_dec, est_rappele, moteur, scenario_krach, mon_indice_dec, mon_autocall
        gc.collect()


    # ==========================================
    # VISUALISATION PLOTLY
    # ==========================================
    # Point d'équilibre où le yield_fixe correspond exactement au prélèvement
    spot_breakeven = decrement_annuel / yield_fixe

    # Création des labels de l'axe X personnalisés (Écart + Spot)
    x_tick_vals = np.arange(400, 2200, 200)
    x_tick_text = []
    for t in x_tick_vals:
        breakeven_y = (decrement_annuel / t) * 100
        ecart = (yield_fixe * 100) - breakeven_y
        x_tick_text.append(f"{ecart:+.1f}%<br>({t:.0f} pts)")

    # Initialisation de la figure avec un axe Y secondaire
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Probabilité PDI Decrement
    fig.add_trace(
        go.Scatter(x=spots_test, y=probs_pdi_dec, mode='lines+markers',
                   name='Probabilité de toucher le PDI',
                   line=dict(color='orange', width=2),
                   marker=dict(size=6)),
        secondary_y=False,
    )


    # Probabilité de Rappel (Autocall)
    fig.add_trace(
        go.Scatter(x=spots_test, y=probs_rappel, mode='lines+markers',
                   name='Probabilité de Rappel (Autocall)',
                   line=dict(color='green', width=2),
                   marker=dict(symbol='diamond', size=6)),
        secondary_y=False,
    )

    # Sur-perte Decrement (Axe secondaire)
    fig.add_trace(
        go.Scatter(x=spots_test, y=ecarts_finaux_crash, mode='lines+markers',
                   name='Sur-perte du Decrement (Moy PR - Moy Dec)',
                   line=dict(color='red', width=2, dash='dash'),
                   marker=dict(symbol='x', size=8)),
        secondary_y=True,
    )

    # Ligne de Breakeven
    fig.add_vline(x=spot_breakeven, line_dash="dash", line_color="black", line_width=2,
                  annotation_text=f"Breakeven (Spot = {spot_breakeven:.0f} pts)", 
                  annotation_position="top right")

    # Annotation avec les hypothèses de marché
    annotation_text = "<b>Hypothèses de Marché</b><br>"
    for i, regime in enumerate(mes_regimes):
        annotation_text += f"P{i+1} ({regime['duree_annees']} ans) : Drift {regime['r_perf']*100:+.0f}%, Vol {regime['vol']*100:.0f}%, Div In. {regime['yield_initial']*100:.1f}%, Croiss. {regime['croiss_div']*100:+.1f}%<br>"
        
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.0, y=-0.25,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.85)",
        bordercolor="lightgray",
        borderwidth=1,
        font=dict(size=10, color="gray")
    )

    # Mise en forme globale
    fig.update_layout(
        title=dict(text=f"<b>Sensibilité au Spot Initial (Dividende initial fixé à {yield_fixe*100:.1f}%) : Probabilité PDI</b>", font=dict(size=18)),
        xaxis=dict(
            title="Écart de Dividende Initial (Niveau du Spot Initial)",
            tickvals=x_tick_vals,
            ticktext=x_tick_text,
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            title="Probabilités (Toucher PDI / Rappel) (%)",
            rangemode='tozero',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis2=dict(
            title=dict(text="Sur-perte du Decrement (% du Spot initial)", font=dict(color="red")),
            tickfont=dict(color="red"),
            rangemode='tozero',
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.6),
        plot_bgcolor='white',
        hovermode="x unified",
        margin=dict(b=150) # Laisse de la place pour la légende et l'annotation
    )

    fig.show()

if __name__ == "__main__":
    main()
