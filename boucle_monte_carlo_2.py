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
    moyennes_dec_crash = []
    moyennes_pr_crash = []
    
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
        
        if np.any(en_dessous_pdi_dec):
            moy_pr_crash_pct = (np.mean(valeurs_finales_pr[en_dessous_pdi_dec]) / spot) * 100
            moy_dec_crash_pct = (np.mean(valeurs_finales_dec[en_dessous_pdi_dec]) / spot) * 100
        else:
            moy_pr_crash_pct = np.nan
            moy_dec_crash_pct = np.nan
            
        moyennes_pr_crash.append(moy_pr_crash_pct)
        moyennes_dec_crash.append(moy_dec_crash_pct)
        
        # Libération explicite de la mémoire pour éviter un MemoryError
        del traj_pr, traj_dec, est_rappele, moteur, scenario_krach, mon_indice_dec, mon_autocall
        gc.collect()


    # ==========================================
    # VISUALISATION PLOTLY
    # ==========================================
    dummy_moteur = SimulationEngine()
    fig_prob, fig_niveaux = dummy_moteur.plot_sensibilite(
        spots_test, probs_pdi_dec, probs_rappel, moyennes_dec_crash, moyennes_pr_crash, 
        decrement_annuel, yield_fixe, mes_regimes
    )
    fig_prob.show()
    fig_niveaux.show()

if __name__ == "__main__":
    main()
