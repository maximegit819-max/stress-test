import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class MarketScenario:
    """Gère l'environnement de marché de manière dynamique (N périodes)"""
    def __init__(self, config_regimes, annees=10, jours_par_an=252):
        self.annees = annees
        self.jours_par_an = jours_par_an
        self.total_jours = annees * jours_par_an
        self.dt = 1.0 / jours_par_an
        
        self.config_regimes = config_regimes
        
        jour_actuel = 0
        annee_actuelle = 0
        #configure les régimes
        for i, regime in enumerate(self.config_regimes):
            regime['jour_debut'] = jour_actuel
            regime['annee_debut'] = annee_actuelle
            annee_actuelle += regime["duree_annees"]
            jour_actuel += int(regime["duree_annees"] * self.jours_par_an)
            regime['jour_fin'] = min(jour_actuel, self.total_jours)
            regime['annee_fin'] = min(annee_actuelle, self.annees)

        self.r_perf_dyn = np.zeros(self.total_jours)
        self.vol_dyn = np.zeros(self.total_jours)
        
        self.build_regimes()
        
    def build_regimes(self):
        for regime in self.config_regimes:
            j_deb = regime['jour_debut']
            j_fin = regime['jour_fin']
            if j_deb < self.total_jours:
                self.r_perf_dyn[j_deb:j_fin] = regime["r_perf"]
                self.vol_dyn[j_deb:j_fin] = regime["vol"]
                
    def get_cash_dividend(self, t, niveau_initial, spots_debut_periode):
        annee_prec = (t - 1) // self.jours_par_an
        
        current_regime = self.config_regimes[-1]
        regime_idx = len(self.config_regimes) - 1
        for i, regime in enumerate(self.config_regimes):
            if annee_prec < regime['annee_fin']:
                current_regime = regime
                regime_idx = i
                break
                
        yield_initial = current_regime['yield_initial']
        croiss_div = current_regime['croiss_div']
        
        if regime_idx == 0:
            spot_base = niveau_initial
        else:
            spot_base = spots_debut_periode.get(regime_idx, niveau_initial)
            
        div_base = spot_base * yield_initial
        annees_dans_regime = annee_prec - current_regime['annee_debut']
        
        return div_base * (1 + croiss_div)**annees_dans_regime


class DecrementIndex:
    """Modélise spécifiquement la mécanique mathématique d'un indice Decrement"""
    def __init__(self, niveau_initial=1000.0, decrement_annuel=50.0):
        self.niveau_initial = niveau_initial
        self.decrement_annuel = decrement_annuel
        
    def simulate(self, scenario: MarketScenario, nb_trajectoires, Z_chocs):
        d_points = self.decrement_annuel / scenario.jours_par_an
        dt = scenario.dt
        total_jours = scenario.total_jours
        
        trajectoires_dec = np.zeros((nb_trajectoires, total_jours + 1))
        trajectoires_pr = np.zeros((nb_trajectoires, total_jours + 1))
        
        trajectoires_dec[:, 0] = self.niveau_initial
        trajectoires_pr[:, 0] = self.niveau_initial
        
        spots_debut_periode = {}
        jour_regime_map = {regime['jour_debut']: i for i, regime in enumerate(scenario.config_regimes) if i > 0}
        
        for t in range(1, total_jours + 1):
            z = Z_chocs[:, t-1]
            vol_j = scenario.vol_dyn[t-1]
            r_perf_j = scenario.r_perf_dyn[t-1]
            
            choc_jour = vol_j * np.sqrt(dt) * z
            
            # --- Indice Price Return ---
            evol_pr = np.exp((r_perf_j - 0.5 * vol_j**2) * dt + choc_jour)
            n_niv_pr = trajectoires_pr[:, t-1] * evol_pr
            
            if (t - 1) in jour_regime_map:
                idx = jour_regime_map[t - 1]
                spots_debut_periode[idx] = np.mean(trajectoires_pr[:, t-1])
            
            # --- Calcul du dividende cash ---
            cash_div_jour = scenario.get_cash_dividend(t, self.niveau_initial, spots_debut_periode)
            
            # --- Indice Decrement ---
            prev_levels = trajectoires_dec[:, t-1]
            yield_dynamique = np.zeros_like(prev_levels)
            mask = prev_levels > 0
            
            yield_dynamique[mask] = cash_div_jour / prev_levels[mask]
            yield_dynamique = np.clip(yield_dynamique, 0.0, 1000.0)
            
            mu_TR = r_perf_j + yield_dynamique
            evol_tr = np.exp((mu_TR - 0.5 * vol_j**2) * dt + choc_jour)
            
            n_niv_dec = prev_levels * evol_tr - d_points
            n_niv_dec = np.maximum(0.0, n_niv_dec)
            
            trajectoires_dec[:, t] = n_niv_dec
            trajectoires_pr[:, t] = n_niv_pr
            
        return trajectoires_pr, trajectoires_dec


class AutocallProduct:
    """Modélise le produit structuré et ses règles de marché"""
    def __init__(self, barriere_rappel=1000.0, niveau_pdi=500.0, non_call_period_mois=11, frequence_obs_mois=4):
        self.barriere_rappel = barriere_rappel
        self.niveau_pdi = niveau_pdi
        self.non_call_period_mois = non_call_period_mois
        self.frequence_obs_mois = frequence_obs_mois
        
    def evaluate(self, trajectoires_dec, scenario: MarketScenario, nb_trajectoires):
        est_rappele = np.zeros(nb_trajectoires, dtype=bool)
        jours_par_mois = scenario.jours_par_an // 12
        jours_entre_observations = jours_par_mois * self.frequence_obs_mois
        
        for t in range(1, scenario.total_jours + 1):
            if (t % jours_entre_observations == 0) and (t >= jours_par_mois * self.non_call_period_mois):
                nouveaux_rappels = (~est_rappele) & (trajectoires_dec[:, t] >= self.barriere_rappel)
                est_rappele = est_rappele | nouveaux_rappels
                
        return est_rappele

class SimulationEngine:
    """Le chef d'orchestre qui lance les calculs globaux"""
    def __init__(self, nb_trajectoires=10000, seed=42):
        self.nb_trajectoires = nb_trajectoires
        self.seed = seed
        
    def run(self, index: DecrementIndex, scenario: MarketScenario, product: AutocallProduct):
        np.random.seed(self.seed)
        print(f"Simulation de {self.nb_trajectoires} trajectoires en cours...")
        Z_chocs = np.random.normal(0, 1, size=(self.nb_trajectoires, scenario.total_jours))
        
        traj_pr, traj_dec = index.simulate(scenario, self.nb_trajectoires, Z_chocs)
        est_rappele = product.evaluate(traj_dec, scenario, self.nb_trajectoires)
        
        return traj_pr, traj_dec, est_rappele

    def afficher_statistiques(self, nom_scenario, traj_pr, traj_dec, est_rappele, product: AutocallProduct, scenario: MarketScenario):
        print(f"#### Statistiques pour le {nom_scenario}\n")
        
        pct_rappel = np.mean(est_rappele) * 100
        print(f"- **Pourcentage de trajectoires rappelées (Autocall)** : {pct_rappel:.2f}%")
        
        reps = []
        valeurs_finales_dec = traj_dec[:, -1]
        valeurs_finales_pr = traj_pr[:, -1]
        
        en_dessous_pdi = (valeurs_finales_dec < product.niveau_pdi) & (~est_rappele)
        pct_en_dessous = np.mean(en_dessous_pdi) * 100
        
        print(f"\n#### À MATURITÉ (Année {scenario.annees}) :")
        print(f"- **Pourcentage de fois où l'indice Decrement finit en dessous du PDI** : {pct_en_dessous:.2f}%")
        
        if np.any(en_dessous_pdi):
            moy_dec_en_dessous = np.mean(valeurs_finales_dec[en_dessous_pdi])
            moy_pr_en_dessous = np.mean(valeurs_finales_pr[en_dessous_pdi])
            print(f"- **Position moyenne de l'indice Decrement** (pour ces cas) : {moy_dec_en_dessous:.2f} pts")
            print(f"- **Position moyenne de l'indice PR** (pour ces cas) : {moy_pr_en_dessous:.2f} pts\n")
            
            dec_sous_pdi = valeurs_finales_dec[en_dessous_pdi]
            pr_sous_pdi = valeurs_finales_pr[en_dessous_pdi]
            
            q25 = np.percentile(dec_sous_pdi, 25)
            q50 = np.percentile(dec_sous_pdi, 50)
            q75 = np.percentile(dec_sous_pdi, 75)
            
            print("**Analyse par quartiles (sur les trajectoires sous le PDI)**")
            
            def get_rep(mask):
                if not np.any(mask): return None, None
                med_dec = np.median(dec_sous_pdi[mask])
                indices_in_sous = np.where(mask)[0]
                idx = indices_in_sous[np.argmin(np.abs(dec_sous_pdi[indices_in_sous] - med_dec))]
                original_idx = np.where(en_dessous_pdi)[0][idx]
                return traj_pr[original_idx], traj_dec[original_idx]

            mask_q1 = dec_sous_pdi <= q25
            if np.any(mask_q1):
                print(f"- **Q1 (Niveau <= {q25:.2f} pts)** : Moy Decrement = {np.mean(dec_sous_pdi[mask_q1]):.2f} pts | Moy PR = {np.mean(pr_sous_pdi[mask_q1]):.2f} pts")
            reps.append(get_rep(mask_q1))
                
            mask_q2 = (dec_sous_pdi > q25) & (dec_sous_pdi <= q50)
            if np.any(mask_q2):
                print(f"- **Q2 ({q25:.2f} < Niveau <= {q50:.2f} pts)** : Moy Decrement = {np.mean(dec_sous_pdi[mask_q2]):.2f} pts | Moy PR = {np.mean(pr_sous_pdi[mask_q2]):.2f} pts")
            reps.append(get_rep(mask_q2))
                
            mask_q3 = (dec_sous_pdi > q50) & (dec_sous_pdi <= q75)
            if np.any(mask_q3):
                print(f"- **Q3 ({q50:.2f} < Niveau <= {q75:.2f} pts)** : Moy Decrement = {np.mean(dec_sous_pdi[mask_q3]):.2f} pts | Moy PR = {np.mean(pr_sous_pdi[mask_q3]):.2f} pts")
            reps.append(get_rep(mask_q3))
                
            mask_q4 = dec_sous_pdi > q75
            if np.any(mask_q4):
                print(f"- **Q4 (Niveau > {q75:.2f} pts)** : Moy Decrement = {np.mean(dec_sous_pdi[mask_q4]):.2f} pts | Moy PR = {np.mean(pr_sous_pdi[mask_q4]):.2f} pts")
            reps.append(get_rep(mask_q4))
        else:
            print("- Aucun scénario ne finit en dessous du PDI à maturité.")
            
        return reps

    def _add_background_regimes_plotly(self, fig, scenario):
        """Ajoute les bandes de couleurs et les infos des périodes pour Plotly"""
        colors = ['grey', 'red', 'green', 'orange', 'purple', 'blue']
        for i, regime in enumerate(scenario.config_regimes):
            color = colors[i % len(colors)]
            fig.add_vrect(x0=regime['annee_debut'], x1=regime['annee_fin'], fillcolor=color, opacity=0.1, line_width=0, layer="below")
            fig.add_annotation(x=(regime['annee_debut'] + regime['annee_fin'])/2, y=1.0, yref="paper", 
                               text=f"Période {i+1}<br>Drift: {regime['r_perf']*100:+.0f}%<br>Vol: {regime['vol']*100:.0f}%<br>Yield Div: {regime['yield_initial']*100:.1f}%",
                               showarrow=False, bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgray")

    def _add_product_levels_plotly(self, fig, product):
        """Ajoute les barrières du produit Autocall pour Plotly"""
        fig.add_hline(y=product.niveau_pdi, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"PDI ({product.niveau_pdi} pts)", annotation_position="bottom right")
        fig.add_hline(y=product.barriere_rappel, line_dash="dashdot", line_color="purple", line_width=2,
                      annotation_text=f"Rappel ({product.barriere_rappel} pts)", annotation_position="top right")

    def plot_results(self, nom_scenario, traj_pr, traj_dec, reps_scen, product, scenario, index):
        """Génère et affiche les graphiques interactifs Plotly"""
        import numpy as np
        import plotly.graph_objects as go
        axe_temps = np.linspace(0, scenario.annees, scenario.total_jours + 1)
        titre_base = nom_scenario
        
        # ==========================================
        # GRAPHIQUE 1 : Analyse par Quartiles
        # ==========================================
        fig1 = go.Figure()
        
        colors = ['red', 'orange', 'olive', 'green']
        labels = ['Q1 (Pire)', 'Q2', 'Q3', 'Q4 (Moins pire)']
        
        for q_idx, rep in enumerate(reps_scen):
            if rep[0] is not None:
                t_pr, t_dec = rep
                fig1.add_trace(go.Scatter(x=axe_temps, y=t_dec, mode='lines', 
                                          line=dict(color=colors[q_idx], width=2), 
                                          name=f'Dec {labels[q_idx]}'))
                fig1.add_trace(go.Scatter(x=axe_temps, y=t_pr, mode='lines', 
                                          line=dict(color=colors[q_idx], width=1, dash='dash'), 
                                          opacity=0.7, name=f'Trajectoire équivalente Prize Return ({labels[q_idx]})'))
                
        self._add_background_regimes_plotly(fig1, scenario)
        self._add_product_levels_plotly(fig1, product)
        
        fig1.update_layout(
                           xaxis_title="Années", yaxis_title="Niveau de l'indice",
                           xaxis=dict(range=[0, scenario.annees]), yaxis=dict(rangemode='tozero'),
                           legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
                           margin=dict(l=40, r=40, t=60, b=80), hovermode="x unified")

        # ==========================================
        # GRAPHIQUE 2 : Le nuage de points
        # ==========================================
        fig2 = go.Figure()
        
        valeurs_finales_toutes_dec = traj_dec[:, -1]
        mediane_generale_finale = np.median(valeurs_finales_toutes_dec)
        idx_proche_mediane = np.argmin(np.abs(valeurs_finales_toutes_dec - mediane_generale_finale))
        traj_proche_mediane = traj_dec[idx_proche_mediane]
        

                                      
        fig2.add_trace(go.Scatter(x=axe_temps, y=traj_proche_mediane, mode='lines', 
                                  line=dict(color='black', width=3), 
                                  name='Trajectoire (niveau final = médiane globale)'))
                                  
        colors_q = ['red', 'orange', 'olive', 'green']
        
        q25_all = np.percentile(valeurs_finales_toutes_dec, 25)
        q50_all = np.percentile(valeurs_finales_toutes_dec, 50)
        q75_all = np.percentile(valeurs_finales_toutes_dec, 75)
        
        def get_rep_all(mask):
            if not np.any(mask): return None, None
            med_dec = np.median(valeurs_finales_toutes_dec[mask])
            indices_in_mask = np.where(mask)[0]
            idx = indices_in_mask[np.argmin(np.abs(valeurs_finales_toutes_dec[indices_in_mask] - med_dec))]
            return traj_pr[idx], traj_dec[idx]
            
        reps_all = [
            get_rep_all(valeurs_finales_toutes_dec <= q25_all),
            get_rep_all((valeurs_finales_toutes_dec > q25_all) & (valeurs_finales_toutes_dec <= q50_all)),
            get_rep_all((valeurs_finales_toutes_dec > q50_all) & (valeurs_finales_toutes_dec <= q75_all)),
            get_rep_all(valeurs_finales_toutes_dec > q75_all)
        ]

        for i, rep in enumerate(reps_all):
            if rep[0] is not None:
                color = colors_q[i % len(colors_q)]
                fig2.add_trace(go.Scatter(x=axe_temps, y=rep[1], mode='lines', 
                                          line=dict(color=color, width=2.5), 
                                          name=f'Médiane Globale Q{i+1} (Decrement)'))
                fig2.add_trace(go.Scatter(x=axe_temps, y=rep[0], mode='lines', 
                                          line=dict(color=color, width=2, dash='dot'), 
                                          name=f'Trajectoire équivalente Prize Return (Q{i+1})'))
        
        self._add_background_regimes_plotly(fig2, scenario)
        self._add_product_levels_plotly(fig2, product)
        
        fig2.update_layout(
                           xaxis_title="Années", yaxis_title="Niveau de l'indice Decrement",
                           xaxis=dict(range=[0, scenario.annees]), yaxis=dict(range=[0, max(2000, np.max(traj_proche_mediane)*1.2)]),
                           legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
                           margin=dict(l=40, r=40, t=60, b=80))
                           
        return fig1, fig2

    def plot_sensibilite(self, spots_test, probs_pdi_dec, probs_rappel, ecarts_finaux_crash, decrement_annuel, yield_fixe, mes_regimes):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import numpy as np
        
        spot_breakeven = decrement_annuel / yield_fixe

        x_tick_vals = np.arange(min(spots_test), max(spots_test)+200, 200)
        x_tick_text = []
        for t in x_tick_vals:
            breakeven_y = (decrement_annuel / t) * 100
            ecart = (yield_fixe * 100) - breakeven_y
            x_tick_text.append(f"{ecart:+.1f}%<br>({t:.0f} pts)")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=spots_test, y=probs_pdi_dec, mode='lines+markers',
                       name='Probabilité de toucher le PDI',
                       line=dict(color='orange', width=2),
                       marker=dict(size=6)),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=spots_test, y=probs_rappel, mode='lines+markers',
                       name='Probabilité de Rappel (Autocall)',
                       line=dict(color='green', width=2),
                       marker=dict(symbol='diamond', size=6)),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=spots_test, y=ecarts_finaux_crash, mode='lines+markers',
                       name='Sur-perte du Decrement (Moy PR - Moy Dec)',
                       line=dict(color='red', width=2, dash='dash'),
                       marker=dict(symbol='x', size=8)),
            secondary_y=True,
        )

        fig.add_vline(x=spot_breakeven, line_dash="dash", line_color="black", line_width=2,
                      annotation_text=f"Breakeven (Spot = {spot_breakeven:.0f} pts)", 
                      annotation_position="top right")

        annotation_text = "<b>Hypothèses de Marché</b><br>"
        for i, regime in enumerate(mes_regimes):
            annotation_text += f"P{i+1} ({regime['duree_annees']} ans) : Drift {regime['r_perf']*100:+.0f}%, Vol {regime['vol']*100:.0f}%, Div In. {regime['yield_initial']*100:.1f}%, Croiss. {regime['croiss_div']*100:+.1f}%<br>"
            
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.0, y=-0.35,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=10, color="gray")
        )

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
            legend=dict(orientation="h", yanchor="top", y=-0.55, xanchor="center", x=0.5),
            plot_bgcolor='white',
            hovermode="x unified",
            margin=dict(b=250),
            height=700
        )

        return fig

if __name__ == "__main__":
    mes_regimes = [
        {"duree_annees": 3, "r_perf": 0.04, "vol": 0.15, "yield_initial": 0.03, "croiss_div": 0.04},
        {"duree_annees": 2, "r_perf": -0.15, "vol": 0.35, "yield_initial": 0.01, "croiss_div": 0.0},
        {"duree_annees": 5, "r_perf": 0.05, "vol": 0.18, "yield_initial": 0.04, "croiss_div": 0.05}
    ]
    
    scenario_krach = MarketScenario(config_regimes=mes_regimes, annees=10, jours_par_an=252)
    mon_indice_dec = DecrementIndex(niveau_initial=1000.0, decrement_annuel=50.0)
    pdi_pts = mon_indice_dec.niveau_initial * 0.50
    mon_autocall = AutocallProduct(barriere_rappel=1000.0, niveau_pdi=pdi_pts, non_call_period_mois=11, frequence_obs_mois=4)
    
    moteur = SimulationEngine(nb_trajectoires=10000,seed=42)
    traj_pr, traj_dec, est_rappele = moteur.run(mon_indice_dec, scenario_krach, mon_autocall)
    
    nom_scenario = f"Scénario N-Périodes Test"
    reps_scen = moteur.afficher_statistiques(nom_scenario, traj_pr, traj_dec, est_rappele, mon_autocall, scenario_krach)
    
    fig1, fig2 = moteur.plot_results(nom_scenario, traj_pr, traj_dec, reps_scen, mon_autocall, scenario_krach, mon_indice_dec)
    fig1.show()
    fig2.show()
