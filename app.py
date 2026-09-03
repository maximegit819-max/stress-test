import streamlit as st
import numpy as np
import io
import contextlib
import plotly.graph_objects as go
import pandas as pd
import monte_carlo_2
import importlib
import gc

importlib.reload(monte_carlo_2)
from monte_carlo_2 import MarketScenario, DecrementIndex, AutocallProduct, SimulationEngine

st.set_page_config(page_title="Monte Carlo - Autocall", layout="wide")

st.title("Simulateur Monte Carlo")

# --- SIDEBAR ---
mode = st.sidebar.radio("Mode d'analyse", ["Scénario Fixe", "Analyse de Sensibilité (Spots)", "Matrice d'Équivalence (PR)", "Surface 3D (Decrement)"])
st.sidebar.divider()

st.sidebar.header("Paramètres")

with st.sidebar.expander("1. Durée", expanded=True):
    annees = st.number_input("Années totales", value=10, min_value=1)

with st.sidebar.expander("2. Configuration des Périodes", expanded=True):
    nb_periodes = st.number_input("Nombre de Périodes", value=3, min_value=1, step=1)
    
    mes_regimes_input = []
    somme_annees = 0
    for i in range(int(nb_periodes)):
        st.markdown(f"**Période {i+1}**")
        def_d = 3 if i==0 else (2 if i==1 else 5)
        def_rp = 4.0 if i==0 else (-15.0 if i==1 else 5.0)
        def_vol = 15.0 if i==0 else (35.0 if i==1 else 18.0)
        def_yi = 3.0 if i==0 else (1.0 if i==1 else 4.0)
        def_yi = 3.0 if i==0 else (1.0 if i==1 else 4.0)
        
        d = st.number_input(f"Durée (ans) P{i+1}", value=def_d, key=f"d_pct_{i}")
        rp = st.number_input(f"Drift Total Return P{i+1} (%)", value=def_rp, format="%.1f", key=f"rp_pct_{i}")
        vol = st.number_input(f"Volatilité P{i+1} (%)", value=def_vol, format="%.1f", key=f"vol_pct_{i}")
        if mode != "Surface 3D (Decrement)":
            yi = st.number_input(f"Yield P{i+1} (%)", value=def_yi, format="%.2f", key=f"yi_pct_{i}")
        else:
            yi = def_yi
        st.divider()
        
        somme_annees += d
        mes_regimes_input.append({
            "duree_annees": d,
            "r_perf": rp / 100.0,
            "vol": vol / 100.0,
            "yield_initial": yi / 100.0
        })

with st.sidebar.expander("3. Indice Decrement", expanded=(mode != "Analyse de Sensibilité (Spots)")):
    if mode in ["Scénario Fixe", "Matrice d'Équivalence (PR)", "Surface 3D (Decrement)"]:
        niveau_initial = st.number_input("Niveau Initial", value=1000.0, step=100.0)
    else:
        st.info("Le Niveau Initial est testé sur une plage.")
        spot_min = st.number_input("Spot Min", value=400.0, step=50.0)
        spot_max = st.number_input("Spot Max", value=2000.0, step=50.0)
        nb_spots = st.number_input("Nombre d'itérations", value=33, step=1)
        
    decrement_annuel = st.number_input("Décrément (pts)", value=50.0, step=5.0)

with st.sidebar.expander("4. Produit Autocall", expanded=False):
    if mode != "Surface 3D (Decrement)":
        st.info("Les barrières (Rappel et PDI) s'adaptent au Spot Initial testé.")
        barriere_rappel_pct = st.number_input("Barrière Rappel (%)", value=100.0, step=10.0) / 100.0
        niveau_pdi_pct = st.number_input("Niveau PDI (%)", value=50.0, step=10.0) / 100.0
        degressivite = st.number_input("Dégressivité de Rappel (%/obs)", value=0.0, step=1.0, help="Baisse en pourcentage du niveau initial à chaque constatation après la période de lock-up.")
    else:
        barriere_rappel_pct = 1.0
        niveau_pdi_pct = 0.5
        degressivite = 0.0
        
    coupon_periode = st.number_input("Coupon par observation (%)", value=2.0, step=0.1)
    non_call_period_mois = st.number_input("Non-Call (mois)", value=11, step=1)
    frequence_obs_mois = st.number_input("Fréq. Obs (mois)", value=4, step=1)

with st.sidebar.expander("5. Moteur de Simulation", expanded=False):
    nb_trajectoires = st.number_input("Nb Trajectoires", value=2000 if mode == "Scénario Fixe" else 1000, step=500)
    seed = st.number_input("Seed aléatoire", value=42, step=1)

if mode == "Scénario Fixe":
    btn_text = "Lancer le Scénario Fixe"
elif mode == "Analyse de Sensibilité (Spots)":
    btn_text = "Lancer l'Analyse de Sensibilité"
elif mode == "Matrice d'Équivalence (PR)":
    btn_text = "Générer la Matrice"
    st.sidebar.divider()
    tolerance = st.sidebar.slider("Tolérance d'équivalence (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
else:
    btn_text = "Générer la Surface 3D"
    st.sidebar.divider()
    list_coupons = np.arange(0.25, 5.25, 0.25)
    coupon_3d = st.sidebar.selectbox("Coupon pour vue 3D", [f"{c:.2f}%" for c in list_coupons], index=len(list_coupons)//2)

lancer = st.sidebar.button(btn_text, type="primary", use_container_width=True)

# --- MAIN AREA ---
if lancer:
    if somme_annees != annees:
        st.error(f"La somme des durées des régimes ({somme_annees}) doit être exactement égale au total d'années ({annees}).")
    else:
        moteur = SimulationEngine(nb_trajectoires=int(nb_trajectoires), seed=int(seed))
        scenario_krach = MarketScenario(config_regimes=mes_regimes_input, annees=int(annees))

        if mode == "Scénario Fixe":
            with st.spinner(f"Génération des {int(nb_trajectoires)} trajectoires de Monte Carlo en cours..."):
                barriere_rappel = niveau_initial * barriere_rappel_pct
                niveau_pdi = niveau_initial * niveau_pdi_pct
                
                mon_indice_dec = DecrementIndex(niveau_initial=niveau_initial, decrement_annuel=decrement_annuel)
                mon_autocall = AutocallProduct(barriere_rappel=barriere_rappel, niveau_pdi=niveau_pdi, non_call_period_mois=int(non_call_period_mois), frequence_obs_mois=int(frequence_obs_mois), degressivite=float(degressivite), coupon_periode=float(coupon_periode))
                moteur = monte_carlo_2.SimulationEngine(nb_trajectoires=int(nb_trajectoires), seed=42)
                traj_pr, traj_dec, est_rappele_dec, obs_de_rappel_dec, payoffs_dec, est_rappele_pr, obs_de_rappel_pr, payoffs_pr = moteur.run(mon_indice_dec, scenario_krach, mon_autocall)
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    nom_scenario = f"Scénario Fixe (Spot {niveau_initial:.0f})"
                    reps_scen, bin_stats = moteur.afficher_statistiques(nom_scenario, traj_pr, traj_dec, est_rappele_dec, payoffs_dec, est_rappele_pr, payoffs_pr, mon_autocall, scenario_krach)
                stats_text = f.getvalue()
                
                st.success(f"Simulation terminée avec succès !")
                
                col_stats, col_graphs = st.columns([1, 2])
                
                with col_stats:
                    st.subheader("Statistiques")
                    st.markdown(stats_text)
                    
                with col_graphs:
                    st.subheader("Visualisations")
                    fig1, fig2 = moteur.plot_results(nom_scenario, traj_pr, traj_dec, reps_scen, mon_autocall, scenario_krach, mon_indice_dec)
                    fig_dist_dec, fig_dist_pr, fig_dist_rappel = moteur.plot_distributions(traj_pr, traj_dec, mon_autocall, scenario_krach)
                    if len(bin_stats) > 0:
                        fig_binned = moteur.plot_binned_averages(bin_stats)
                    else:
                        fig_binned = None
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    if fig_binned:
                        st.subheader("Analyse par Tranches (Sous PDI)")
                        st.plotly_chart(fig_binned, use_container_width=True)
                        
                    st.subheader("Distributions")
                    st.plotly_chart(fig_dist_dec, use_container_width=True)
                    st.plotly_chart(fig_dist_pr, use_container_width=True)
                    st.plotly_chart(fig_dist_rappel, use_container_width=True)
                    
        elif mode == "Analyse de Sensibilité (Spots)": # Analyse de Sensibilité
            spots_test = np.linspace(spot_min, spot_max, int(nb_spots))
            
            probs_pdi_dec = []
            probs_pdi_pr = []
            probs_rappel = []
            moyennes_pr_crash = []
            moyennes_dec_crash = []
            moyennes_payoffs_dec = []
            moyennes_payoffs_pr = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, spot in enumerate(spots_test):
                status_text.text(f"Simulation pour Spot = {spot:.0f} pts ({i+1}/{len(spots_test)})...")
                
                mon_indice_dec = DecrementIndex(niveau_initial=spot, decrement_annuel=decrement_annuel)
                pdi_niveau_dyn = spot * niveau_pdi_pct
                barriere_rappel = spot * barriere_rappel_pct
                
                mon_autocall = AutocallProduct(barriere_rappel=barriere_rappel, niveau_pdi=pdi_niveau_dyn, non_call_period_mois=int(non_call_period_mois), frequence_obs_mois=int(frequence_obs_mois), degressivite=float(degressivite), coupon_periode=float(coupon_periode))
                
                # Réinitialiser la seed à chaque boucle pour que les courbes de sensibilité soient très lisses
                moteur.seed = int(seed)
                np.random.seed(moteur.seed)
                
                traj_pr, traj_dec, est_rappele_dec, obs_de_rappel_dec, payoffs_dec, est_rappele_pr, obs_de_rappel_pr, payoffs_pr = moteur.run(mon_indice_dec, scenario_krach, mon_autocall)
                
                valeurs_finales_dec = traj_dec[:, -1]
                valeurs_finales_pr = traj_pr[:, -1]
                
                en_dessous_pdi_dec = (valeurs_finales_dec < pdi_niveau_dyn) & (~est_rappele_dec)
                en_dessous_pdi_pr = (valeurs_finales_pr < pdi_niveau_dyn) & (~est_rappele_pr)
                
                probs_pdi_dec.append(np.mean(en_dessous_pdi_dec) * 100)
                probs_pdi_pr.append(np.mean(en_dessous_pdi_pr) * 100)
                probs_rappel.append(np.mean(est_rappele_dec) * 100)
                
                if np.any(en_dessous_pdi_dec):
                    moy_pr_crash_pct = (np.mean(valeurs_finales_pr[en_dessous_pdi_dec]) / spot) * 100
                    moy_dec_crash_pct = (np.mean(valeurs_finales_dec[en_dessous_pdi_dec]) / spot) * 100
                else:
                    moy_pr_crash_pct = np.nan
                    moy_dec_crash_pct = np.nan
                    
                moyennes_pr_crash.append(moy_pr_crash_pct)
                moyennes_dec_crash.append(moy_dec_crash_pct)
                moyennes_payoffs_dec.append(np.mean(payoffs_dec) * 100)
                moyennes_payoffs_pr.append(np.mean(payoffs_pr) * 100)
                
                del traj_pr, traj_dec, est_rappele_dec, est_rappele_pr, mon_indice_dec, mon_autocall
                gc.collect()
                
                progress_bar.progress((i + 1) / len(spots_test))
                
            status_text.text("Génération du graphique interactif...")
            
            yield_fixe = mes_regimes_input[0]["yield_initial"]
            
            fig_prob, fig_niveaux, fig_ecart, fig_prob_d1, fig_ecart_d1, fig_payoff = moteur.plot_sensibilite(
                spots_test, probs_pdi_dec, probs_rappel, moyennes_dec_crash, moyennes_pr_crash, moyennes_payoffs_dec, moyennes_payoffs_pr,
                decrement_annuel, yield_fixe, mes_regimes_input
            )
            
            st.success("Analyse de Sensibilité terminée !")
            progress_bar.empty()
            status_text.empty()
            
            st.plotly_chart(fig_prob, use_container_width=True)
            st.plotly_chart(fig_prob_d1, use_container_width=True)
            st.plotly_chart(fig_payoff, use_container_width=True)
            st.plotly_chart(fig_niveaux, use_container_width=True)
            st.plotly_chart(fig_ecart, use_container_width=True)
            st.plotly_chart(fig_ecart_d1, use_container_width=True)

        elif mode == "Matrice d'Équivalence (PR)":
            st.header("Matrice d'Équivalence PR")
            
            # Paramètres de la grille
            list_pdis = np.arange(40.0, 105.0, 5.0)
            list_barrieres = np.arange(100.0, 155.0, 5.0)
            
            with st.spinner("Calcul du Payoff Cible (Decrement) et génération de la matrice..."):
                spot = float(niveau_initial)
                pdi_niveau = spot * niveau_pdi_pct
                barriere_rappel = spot * barriere_rappel_pct
                
                # 1. Calcul du Payoff Cible sur Decrement
                mon_indice_dec = DecrementIndex(niveau_initial=spot, decrement_annuel=decrement_annuel)
                mon_autocall = AutocallProduct(
                    barriere_rappel=barriere_rappel, niveau_pdi=pdi_niveau, 
                    non_call_period_mois=int(non_call_period_mois), frequence_obs_mois=int(frequence_obs_mois), 
                    degressivite=float(degressivite), coupon_periode=float(coupon_periode)
                )
                scenario_krach = MarketScenario(
                    annees=int(annees), jours_par_an=252,
                    config_regimes=mes_regimes_input
                )
                moteur = monte_carlo_2.SimulationEngine(nb_trajectoires=10000, seed=42)
                _, _, _, _, payoffs_dec, _, _, _ = moteur.run(mon_indice_dec, scenario_krach, mon_autocall)
                
                target_payoff = np.mean(payoffs_dec) * 100
                
                st.success(f"**Payoff Cible (Decrement)** : {target_payoff:.2f}%  *(Tolérance : ±{tolerance}%)*")
                
                # 2. Génération de la grille PR
                df = moteur.generer_matrice_structurelle(
                    mon_indice_dec, scenario_krach, mon_autocall, 
                    list_coupons, list_pdis, list_barrieres
                )
                    
                # 3. Filtrage Visuel
                df_filtered = df.copy()
                for col in df_filtered.columns:
                    df_filtered[col] = df_filtered[col].apply(
                        lambda x: f"{x:.2f}%" if abs(x - target_payoff) <= tolerance else ""
                    )
                
                # Affichage de la matrice
                st.markdown("Seules les combinaisons de structure PR atteignant le Payoff Cible sont affichées ci-dessous :")
                st.dataframe(df_filtered, use_container_width=True, height=600)
                
        elif mode == "Surface 3D (Decrement)":
            st.header("Surface 3D (Decrement)")
            
            # Paramètres de la grille (nouveaux paramètres demandés)
            list_coupons = [float(coupon_periode)]

            list_pdis = np.arange(35.0, 85.0, 5.0)
            list_barrieres = np.arange(80.0, 125.0, 5.0)
            
            with st.spinner("Génération de la surface 3D sur l'indice Decrement..."):
                spot = float(niveau_initial)
                pdi_niveau = spot * niveau_pdi_pct
                barriere_rappel = spot * barriere_rappel_pct
                
                mon_indice_dec = DecrementIndex(niveau_initial=spot, decrement_annuel=decrement_annuel)
                mon_autocall = AutocallProduct(
                    barriere_rappel=barriere_rappel, niveau_pdi=pdi_niveau, 
                    non_call_period_mois=int(non_call_period_mois), frequence_obs_mois=int(frequence_obs_mois), 
                    degressivite=float(degressivite), coupon_periode=float(coupon_periode)
                )
                scenario_krach = MarketScenario(
                    annees=int(annees), jours_par_an=252,
                    config_regimes=mes_regimes_input
                )
                moteur = monte_carlo_2.SimulationEngine(nb_trajectoires=10000, seed=42)
                
                # Génération de la grille avec use_decrement=True
                df = moteur.generer_matrice_structurelle(
                    mon_indice_dec, scenario_krach, mon_autocall, 
                    list_coupons, list_pdis, list_barrieres, use_decrement=True
                )
                
                # Extraction des données pour Plotly
                coupon_col = f"{coupon_periode:.2f}%"
                df_plot = df[[coupon_col]].reset_index()
                df_plot['PDI'] = df_plot['PDI'].str.replace('%', '').astype(float)
                df_plot['Barrière'] = df_plot['Barrière'].str.replace('%', '').astype(float)
                pivot_df = df_plot.pivot(index='PDI', columns='Barrière', values=coupon_col)
                
                x_vals = pivot_df.columns.values.astype(float)
                y_vals = pivot_df.index.values.astype(float)
                z_vals = pivot_df.values.astype(float)
                
                x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
                
                fig3d = go.Figure()
                fig3d.add_trace(go.Surface(
                    z=z_vals, x=x_mesh, y=y_mesh, 
                    colorscale='Viridis', name="Payoff Decrement", showscale=False
                ))
                
                fig3d.update_layout(
                    title=f"Topographie des payoffs Decrement (Coupon fixé à {coupon_periode:.2f}%)",
                    scene=dict(
                        xaxis_title='Barrière Initiale (%)',
                        yaxis_title='Niveau PDI (%)',
                        zaxis_title='Payoff (%)'
                    ),
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig3d, use_container_width=True)

else:
    st.info("Sélectionnez le mode d'analyse dans la barre latérale, ajustez les paramètres, puis cliquez sur le bouton pour lancer.")
