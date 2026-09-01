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
        self.yield_dyn = np.zeros(self.total_jours)
        
        self.build_regimes()
        
    def build_regimes(self):
        for regime in self.config_regimes:
            j_deb = regime['jour_debut']
            j_fin = regime['jour_fin']
            if j_deb < self.total_jours:
                self.r_perf_dyn[j_deb:j_fin] = regime["r_perf"]
                self.vol_dyn[j_deb:j_fin] = regime["vol"]
                self.yield_dyn[j_deb:j_fin] = regime["yield_initial"]
                

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
        
        for t in range(1, total_jours + 1):
            z = Z_chocs[:, t-1]
            vol_j = scenario.vol_dyn[t-1]
            r_perf_j = scenario.r_perf_dyn[t-1]  # Drift Total Return
            yield_j = scenario.yield_dyn[t-1]    # Constant Yield
            
            choc_jour = vol_j * np.sqrt(dt) * z
            
            # --- Indice Total Return (Base) ---
            evol_tr = np.exp((r_perf_j - 0.5 * vol_j**2) * dt + choc_jour)
            
            # --- Indice Price Return (Soustraction du dividende dans l'exponentielle) ---
            evol_pr = np.exp((r_perf_j - yield_j - 0.5 * vol_j**2) * dt + choc_jour)
            
            # Application
            n_niv_pr = trajectoires_pr[:, t-1] * evol_pr
            n_niv_dec = (trajectoires_dec[:, t-1] * evol_tr) - d_points
            
            # Plancher à 0
            n_niv_dec = np.maximum(0.0, n_niv_dec)
            
            trajectoires_dec[:, t] = n_niv_dec
            trajectoires_pr[:, t] = n_niv_pr
            
        return trajectoires_pr, trajectoires_dec


class AutocallProduct:
    """Modélise le produit structuré et ses règles de marché"""
    def __init__(self, barriere_rappel=1000.0, niveau_pdi=500.0, non_call_period_mois=11, frequence_obs_mois=4, degressivite=0.0):
        self.barriere_rappel = barriere_rappel
        self.niveau_pdi = niveau_pdi
        self.non_call_period_mois = non_call_period_mois
        self.frequence_obs_mois = frequence_obs_mois
        self.degressivite = degressivite
        
    def evaluate(self, trajectoires_dec, scenario: MarketScenario, nb_trajectoires):
        est_rappele = np.zeros(nb_trajectoires, dtype=bool)
        jours_par_mois = scenario.jours_par_an // 12
        jours_entre_observations = jours_par_mois * self.frequence_obs_mois
        
        baisse_en_points = self.barriere_rappel * (self.degressivite / 100.0)
        nb_observations_passees = 0
        
        for t in range(1, scenario.total_jours + 1):
            if (t % jours_entre_observations == 0) and (t >= jours_par_mois * self.non_call_period_mois):
                barriere_actuelle = self.barriere_rappel - (nb_observations_passees * baisse_en_points)
                nouveaux_rappels = (~est_rappele) & (trajectoires_dec[:, t] >= barriere_actuelle)
                est_rappele = est_rappele | nouveaux_rappels
                nb_observations_passees += 1
                
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
        
        bin_stats = []
        if np.any(en_dessous_pdi):
            moy_dec_en_dessous = np.mean(valeurs_finales_dec[en_dessous_pdi])
            moy_pr_en_dessous = np.mean(valeurs_finales_pr[en_dessous_pdi])
            print(f"- **Position moyenne de l'indice Decrement** (pour ces cas) : {moy_dec_en_dessous:.2f} pts")
            print(f"- **Position moyenne de l'indice PR** (pour ces cas) : {moy_pr_en_dessous:.2f} pts\n")
            
            dec_sous_pdi = valeurs_finales_dec[en_dessous_pdi]
            pr_sous_pdi = valeurs_finales_pr[en_dessous_pdi]
            
            bins = [0, 10, 25, 50, 75, 90, 100]
            print("**Moyennes par tranches de centiles (sur les trajectoires sous le PDI)**")
            for i in range(len(bins)-1):
                p_low = bins[i]
                p_high = bins[i+1]
                val_high = np.percentile(dec_sous_pdi, p_high)
                
                if p_low == 0:
                    mask = dec_sous_pdi <= val_high
                else:
                    val_low = np.percentile(dec_sous_pdi, p_low)
                    mask = (dec_sous_pdi > val_low) & (dec_sous_pdi <= val_high)
                
                if np.any(mask):
                    moy_dec = np.mean(dec_sous_pdi[mask])
                    moy_pr = np.mean(pr_sous_pdi[mask])
                    label = f"{p_low}%-{p_high}%"
                    bin_stats.append({"label": label, "moy_dec": moy_dec, "moy_pr": moy_pr})
                    print(f"- Tranche [{label}] : Moyenne Decrement = {moy_dec:.2f} pts | Moyenne PR = {moy_pr:.2f} pts")
            
            pct_levels = [10, 25, 50, 75, 90]
            pct_values = [np.percentile(dec_sous_pdi, p) for p in pct_levels]
            
            print("\n**Trajectoires représentatives (Analyse par centiles)**")
            
            for p, val in zip(pct_levels, pct_values):
                idx = np.argmin(np.abs(dec_sous_pdi - val))
                original_idx = np.where(en_dessous_pdi)[0][idx]
                
                print(f"- **Top {p}%** : Niveau cible Decrement = {val:.2f} pts | PR équivalent = {pr_sous_pdi[idx]:.2f} pts")
                reps.append((traj_pr[original_idx], traj_dec[original_idx], f"Top {p}%"))
        else:
            print("- Aucun scénario ne finit en dessous du PDI à maturité.")
            
        return reps, bin_stats

    def _add_background_regimes_plotly(self, fig, scenario):
        """Ajoute les bandes de couleurs et les infos des périodes pour Plotly"""
        colors = ['grey', 'red', 'green', 'orange', 'purple', 'blue']
        for i, regime in enumerate(scenario.config_regimes):
            color = colors[i % len(colors)]
            fig.add_vrect(x0=regime['annee_debut'], x1=regime['annee_fin'], fillcolor=color, opacity=0.1, line_width=0, layer="below")
            fig.add_annotation(x=(regime['annee_debut'] + regime['annee_fin'])/2, y=1.0, yref="paper", 
                               text=f"Période {i+1}<br>Drift: {regime['r_perf']*100:+.0f}%<br>Vol: {regime['vol']*100:.0f}%<br>Yield Div: {regime['yield_initial']*100:.1f}%",
                               showarrow=False, bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgray")

    def _add_product_levels_plotly(self, fig, product, scenario):
        """Ajoute les barrières du produit Autocall pour Plotly"""
        fig.add_hline(y=product.niveau_pdi, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"PDI ({product.niveau_pdi:.0f} pts)", annotation_position="bottom right")
        
        if product.degressivite == 0.0:
            fig.add_hline(y=product.barriere_rappel, line_dash="dashdot", line_color="purple", line_width=2,
                          annotation_text=f"Rappel ({product.barriere_rappel:.0f} pts)", annotation_position="top right")
        else:
            import plotly.graph_objects as go
            jours_par_mois = scenario.jours_par_an // 12
            jours_entre_observations = jours_par_mois * product.frequence_obs_mois
            baisse_en_points = product.barriere_rappel * (product.degressivite / 100.0)
            
            x_barriere = [0]
            y_barriere = [product.barriere_rappel]
            
            nb_observations_passees = 0
            
            for t in range(1, scenario.total_jours + 1):
                if (t % jours_entre_observations == 0) and (t >= jours_par_mois * product.non_call_period_mois):
                    annee = t / scenario.jours_par_an
                    
                    # Fin de la marche actuelle
                    barriere_actuelle = product.barriere_rappel - (nb_observations_passees * baisse_en_points)
                    x_barriere.append(annee)
                    y_barriere.append(barriere_actuelle)
                    
                    # Début de la marche suivante
                    nb_observations_passees += 1
                    barriere_prochaine = product.barriere_rappel - (nb_observations_passees * baisse_en_points)
                    x_barriere.append(annee)
                    y_barriere.append(barriere_prochaine)
                    
            # Dernière marche jusqu'à la fin
            x_barriere.append(scenario.annees)
            y_barriere.append(product.barriere_rappel - (nb_observations_passees * baisse_en_points))
            
            fig.add_trace(go.Scatter(x=x_barriere, y=y_barriere, mode='lines', 
                                     line=dict(color='purple', width=2, dash='dashdot'), 
                                     name=f'Rappel Dégressif (-{product.degressivite}%/obs)', showlegend=False))

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
        
        colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850']
        
        for idx_rep in range(len(reps_scen)-1, -1, -1):
            rep = reps_scen[idx_rep]
            if rep[0] is not None:
                t_pr, t_dec, label = rep
                c = colors[idx_rep % len(colors)]
                fig1.add_trace(go.Scatter(x=axe_temps, y=t_dec, mode='lines', 
                                          line=dict(color=c, width=2), 
                                          name=f'Dec ({label})',
                                          legendgroup=label))
                fig1.add_trace(go.Scatter(x=axe_temps, y=t_pr, mode='lines', 
                                          line=dict(color=c, width=1, dash='dash'), 
                                          opacity=0.7, name=f'PR équivalent ({label})',
                                          legendgroup=label))
                
        self._add_background_regimes_plotly(fig1, scenario)
        self._add_product_levels_plotly(fig1, product, scenario)
        
        fig1.update_layout(
                           xaxis_title="Années", yaxis_title="Sous PDI",
                           xaxis=dict(range=[0, scenario.annees]), yaxis=dict(rangemode='tozero'),
                           legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
                           margin=dict(l=40, r=40, t=60, b=80), hovermode="x unified")

        # ==========================================
        # GRAPHIQUE 2 : Le nuage de points
        # ==========================================
        fig2 = go.Figure()
        
        valeurs_finales_toutes_dec = traj_dec[:, -1]
        pct_levels = [90, 75, 50, 25, 10]
        colors_pct_map = {90: '#1a9850', 75: '#d9ef8b', 50: '#fee08b', 25: '#fc8d59', 10: '#d73027'}
        
        max_y_display = 2000
        for p in pct_levels:
            val_cible = np.percentile(valeurs_finales_toutes_dec, p)
            idx_closest = np.argmin(np.abs(valeurs_finales_toutes_dec - val_cible))
            t_dec = traj_dec[idx_closest]
            t_pr = traj_pr[idx_closest]
            c = colors_pct_map[p]
            
            if p == 50:
                max_y_display = max(2000, np.max(t_dec)*1.2)
            
            label = f"Top {p}%"
            fig2.add_trace(go.Scatter(x=axe_temps, y=t_dec, mode='lines', 
                                      line=dict(color=c, width=2.5), 
                                      name=f'Dec ({label})',
                                      legendgroup=label))
            fig2.add_trace(go.Scatter(x=axe_temps, y=t_pr, mode='lines', 
                                      line=dict(color=c, width=1.5, dash='dot'), 
                                      name=f'PR équivalent ({label})',
                                      legendgroup=label))
        
        self._add_background_regimes_plotly(fig2, scenario)
        self._add_product_levels_plotly(fig2, product, scenario)
        
        fig2.update_layout(
                           xaxis_title="Années", yaxis_title=None,
                           xaxis=dict(range=[0, scenario.annees]), yaxis=dict(range=[0, max_y_display]),
                           legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
                           margin=dict(l=40, r=40, t=60, b=80), hovermode="x unified")
                           
        return fig1, fig2

    def plot_sensibilite(self, spots_test, probs_pdi_dec, probs_rappel, moyennes_dec_crash, moyennes_pr_crash, decrement_annuel, yield_fixe, mes_regimes):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        spots_array = np.array(spots_test)
        
        # --- Calculs Mathématiques ---
        # 1. Écart de sur-perte
        ecarts = np.array(moyennes_pr_crash) - np.array(moyennes_dec_crash)
        
        from scipy.signal import savgol_filter
        
        # L'écart entre deux spots (ici 50) pour avoir la bonne échelle de dérivée
        delta_spot = spots_array[1] - spots_array[0]

        # 2. Dérivées Secondes (Accélération du risque) via Savitzky-Golay
        # Ce filtre calcule directement la dérivée (deriv=2) en ajustant un polynôme local (polyorder=4)
        # sur une fenêtre glissante (window_length=25).
        d2_prob = savgol_filter(probs_pdi_dec, window_length=25, polyorder=4, deriv=2, delta=delta_spot)
        d2_prob = np.maximum(d2_prob, 0)
        
        # Pour l'écart (Sur-perte)
        d2_ecart = savgol_filter(ecarts, window_length=25, polyorder=4, deriv=2, delta=delta_spot)
        d2_ecart = np.maximum(d2_ecart, 0)
        
        # 3. Dérivées Premières (Vitesse du risque pour le Scoring)
        d1_prob = -savgol_filter(probs_pdi_dec, window_length=25, polyorder=4, deriv=1, delta=delta_spot)
        d1_prob = np.maximum(d1_prob, 0)
        
        d1_ecart = -savgol_filter(ecarts, window_length=25, polyorder=4, deriv=1, delta=delta_spot)
        d1_ecart = np.maximum(d1_ecart, 0)

        x_tick_vals = np.arange(min(spots_test), max(spots_test)+200, 200)
        x_tick_text = []
        for t in x_tick_vals:
            breakeven_y = (decrement_annuel / t) * 100
            ecart = (yield_fixe * 100) - breakeven_y
            x_tick_text.append(f"{ecart:+.1f}%<br>({t:.0f} pts)")

        annotation_text = "<b>Hypothèses de Marché</b><br>"
        for i, regime in enumerate(mes_regimes):
            annotation_text += f"P{i+1} ({regime['duree_annees']} ans) : Drift {regime['r_perf']*100:+.0f}%, Vol {regime['vol']*100:.0f}%, Yield {regime['yield_initial']*100:.1f}%<br>"

        # ==========================================
        # GRAPHIQUE 1 : Probabilités + Accélération (D2)
        # ==========================================
        fig_prob = make_subplots(specs=[[{"secondary_y": True}]])

        fig_prob.add_trace(
            go.Scatter(x=spots_test, y=probs_pdi_dec, mode='lines+markers',
                       name='Probabilité PDI (Dec)',
                       line=dict(color='orange', width=2), marker=dict(size=6)),
            secondary_y=False
        )
        fig_prob.add_trace(
            go.Scatter(x=spots_test, y=probs_rappel, mode='lines+markers',
                       name='Probabilité Autocall',
                       line=dict(color='green', width=2), marker=dict(symbol='diamond', size=6)),
            secondary_y=False
        )
        # Dérivée Seconde
        fig_prob.add_trace(
            go.Scatter(x=spots_test, y=d2_prob, mode='lines',
                       name='Accélération du Risque (Dérivée 2nde)',
                       line=dict(color='orange', width=0), fill='tozeroy', opacity=0.15),
            secondary_y=True
        )

        fig_prob.add_annotation(text=annotation_text, xref="paper", yref="paper", x=0.0, y=-0.35, showarrow=False, align="left", bgcolor="rgba(255, 255, 255, 0.85)", bordercolor="lightgray", borderwidth=1, font=dict(size=10, color="gray"))

        fig_prob.update_layout(
            title=dict(text=f"<b>1. Probabilités & Accélération du Risque PDI</b>", font=dict(size=18)),
            xaxis=dict(title="Écart de Dividende Initial (Niveau du Spot Initial)", tickvals=x_tick_vals, ticktext=x_tick_text, showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="Probabilités (%)", rangemode='tozero', showgrid=True, gridcolor='lightgray'),
            yaxis2=dict(title="Accélération du Risque", showgrid=False, showticklabels=False), # Masquer l'échelle D2 car elle est abstraite
            legend=dict(orientation="h", yanchor="top", y=-0.55, xanchor="center", x=0.5),
            plot_bgcolor='white', hovermode="x unified", margin=dict(b=250), height=700
        )
        
        # ==========================================
        # GRAPHIQUE 1 BIS : Vitesse du Risque PDI (Dérivée 1ère)
        # ==========================================
        fig_prob_d1 = go.Figure()
        fig_prob_d1.add_trace(
            go.Scatter(x=spots_test, y=d1_prob, mode='lines',
                       name='Vitesse du Risque (Score d1)',
                       line=dict(color='orange', width=2), fill='tozeroy', opacity=0.3)
        )
        fig_prob_d1.add_annotation(text=annotation_text, xref="paper", yref="paper", x=0.0, y=-0.35, showarrow=False, align="left", bgcolor="rgba(255, 255, 255, 0.85)", bordercolor="lightgray", borderwidth=1, font=dict(size=10, color="gray"))
        fig_prob_d1.update_layout(
            title=dict(text=f"<b>1 Bis. Vitesse d'augmentation du Risque PDI (Dérivée 1ère)</b>", font=dict(size=18)),
            xaxis=dict(title="Écart de Dividende Initial (Niveau du Spot Initial)", tickvals=x_tick_vals, ticktext=x_tick_text, showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="Vitesse du risque", rangemode='tozero', showgrid=True, gridcolor='lightgray'),
            legend=dict(orientation="h", yanchor="top", y=-0.55, xanchor="center", x=0.5),
            plot_bgcolor='white', hovermode="x unified", margin=dict(b=250), height=500
        )

        # ==========================================
        # GRAPHIQUE 2 : Niveaux Finaux sous PDI
        # ==========================================
        fig_niveaux = go.Figure()

        fig_niveaux.add_trace(
            go.Scatter(x=spots_test, y=moyennes_dec_crash, mode='lines+markers',
                       name='Niveau moyen Decrement (% du Spot)',
                       line=dict(color='orange', width=2, dash='dash'), marker=dict(symbol='circle', size=6))
        )
        fig_niveaux.add_trace(
            go.Scatter(x=spots_test, y=moyennes_pr_crash, mode='lines+markers',
                       name='Niveau moyen Price Return (% du Spot)',
                       line=dict(color='blue', width=2, dash='dot'), marker=dict(symbol='square', size=6))
        )

        fig_niveaux.add_annotation(text=annotation_text, xref="paper", yref="paper", x=0.0, y=-0.35, showarrow=False, align="left", bgcolor="rgba(255, 255, 255, 0.85)", bordercolor="lightgray", borderwidth=1, font=dict(size=10, color="gray"))

        fig_niveaux.update_layout(
            title=dict(text=f"<b>2. Niveaux Finaux Moyens en cas de Krach</b>", font=dict(size=18)),
            xaxis=dict(title="Écart de Dividende Initial (Niveau du Spot Initial)", tickvals=x_tick_vals, ticktext=x_tick_text, showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="Niveau Final (% du Spot Initial)", rangemode='tozero', showgrid=True, gridcolor='lightgray'),
            legend=dict(orientation="h", yanchor="top", y=-0.55, xanchor="center", x=0.5),
            plot_bgcolor='white', hovermode="x unified", margin=dict(b=250), height=700
        )

        # ==========================================
        # GRAPHIQUE 3 : Sur-perte & Dérivée Seconde
        # ==========================================
        fig_ecart = make_subplots(specs=[[{"secondary_y": True}]])

        fig_ecart.add_trace(
            go.Scatter(x=spots_test, y=ecarts, mode='lines+markers',
                       name='Sur-perte (Écart PR - Dec)',
                       line=dict(color='red', width=2), marker=dict(symbol='x', size=6)),
            secondary_y=False
        )
        # Dérivée Seconde de l'écart
        fig_ecart.add_trace(
            go.Scatter(x=spots_test, y=d2_ecart, mode='lines',
                       name='Accélération de la Sur-perte',
                       line=dict(color='red', width=0), fill='tozeroy', opacity=0.15),
            secondary_y=True
        )

        fig_ecart.add_annotation(text=annotation_text, xref="paper", yref="paper", x=0.0, y=-0.35, showarrow=False, align="left", bgcolor="rgba(255, 255, 255, 0.85)", bordercolor="lightgray", borderwidth=1, font=dict(size=10, color="gray"))

        fig_ecart.update_layout(
            title=dict(text=f"<b>3. Sur-perte du Decrement & Accélération</b>", font=dict(size=18)),
            xaxis=dict(title="Écart de Dividende Initial (Niveau du Spot Initial)", tickvals=x_tick_vals, ticktext=x_tick_text, showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="Sur-perte (% du Spot Initial)", rangemode='tozero', showgrid=True, gridcolor='lightgray'),
            yaxis2=dict(title="Accélération", showgrid=False, showticklabels=False), # Échelle masquée
            legend=dict(orientation="h", yanchor="top", y=-0.55, xanchor="center", x=0.5),
            plot_bgcolor='white', hovermode="x unified", margin=dict(b=250), height=700
        )
        
        # ==========================================
        # GRAPHIQUE 3 BIS : Vitesse de la Sur-perte (Dérivée 1ère)
        # ==========================================
        fig_ecart_d1 = go.Figure()
        fig_ecart_d1.add_trace(
            go.Scatter(x=spots_test, y=d1_ecart, mode='lines',
                       name='Vitesse de la Sur-perte (Score d1)',
                       line=dict(color='red', width=2), fill='tozeroy', opacity=0.3)
        )
        fig_ecart_d1.add_annotation(text=annotation_text, xref="paper", yref="paper", x=0.0, y=-0.35, showarrow=False, align="left", bgcolor="rgba(255, 255, 255, 0.85)", bordercolor="lightgray", borderwidth=1, font=dict(size=10, color="gray"))
        fig_ecart_d1.update_layout(
            title=dict(text=f"<b>3 Bis. Vitesse de la Sur-perte (Dérivée 1ère)</b>", font=dict(size=18)),
            xaxis=dict(title="Écart de Dividende Initial (Niveau du Spot Initial)", tickvals=x_tick_vals, ticktext=x_tick_text, showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="Vitesse de la sur-perte", rangemode='tozero', showgrid=True, gridcolor='lightgray'),
            legend=dict(orientation="h", yanchor="top", y=-0.55, xanchor="center", x=0.5),
            plot_bgcolor='white', hovermode="x unified", margin=dict(b=250), height=500
        )
        
        return fig_prob, fig_niveaux, fig_ecart, fig_prob_d1, fig_ecart_d1

    def plot_distributions(self, traj_pr, traj_dec, product, scenario):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        # 1. Distribution des niveaux finaux (sans tenir compte des rappels)
        # On plafonne à 5000 pour regrouper les valeurs extrêmes et on subdivise finement
        valeurs_finales_dec = np.clip(traj_dec[:, -1], a_min=None, a_max=5000)
        valeurs_finales_pr = np.clip(traj_pr[:, -1], a_min=None, a_max=5000)
        
        # Graphique Decrement
        fig_dist_dec = go.Figure()
        fig_dist_dec.add_trace(go.Histogram(x=valeurs_finales_dec, autobinx=False, xbins=dict(start=0, end=5050, size=50), name="Decrement", marker_color='blue', opacity=0.75))
        fig_dist_dec.add_vline(x=product.niveau_pdi, line_dash="dash", line_color="red", annotation_text="PDI")
        tickvals_dist = list(range(0, 5001, 500))
        ticktext_dist = [str(v) if v < 5000 else "5000+" for v in tickvals_dist]

        fig_dist_dec.update_layout(
            title="Distribution des Niveaux Finaux (Decrement - Plafonné à 5000)",
            xaxis_title="Niveau Final (pts)", 
            yaxis_title="Nombre de trajectoires",
            xaxis=dict(range=[0, 5200], tickvals=tickvals_dist, ticktext=ticktext_dist)
        )
        
        # Graphique Price Return
        fig_dist_pr = go.Figure()
        fig_dist_pr.add_trace(go.Histogram(x=valeurs_finales_pr, autobinx=False, xbins=dict(start=0, end=5050, size=50), name="Price Return", marker_color='orange', opacity=0.75))
        fig_dist_pr.add_vline(x=product.niveau_pdi, line_dash="dash", line_color="red", annotation_text="PDI (Indicatif)")
        fig_dist_pr.update_layout(
            title="Distribution des Niveaux Finaux (Price Return - Plafonné à 5000)",
            xaxis_title="Niveau Final (pts)", 
            yaxis_title="Nombre de trajectoires",
            xaxis=dict(range=[0, 5200], tickvals=tickvals_dist, ticktext=ticktext_dist)
        )
        
        # 2. Distribution des dates de rappel
        nb_trajectoires = traj_dec.shape[0]
        date_rappel = np.full(nb_trajectoires, -1)
        
        jours_par_mois = scenario.jours_par_an // 12
        jours_entre_observations = jours_par_mois * product.frequence_obs_mois
        
        est_rappele = np.zeros(nb_trajectoires, dtype=bool)
        dates_possibles = []
        
        for t in range(1, scenario.total_jours + 1):
            if (t % jours_entre_observations == 0) and (t >= jours_par_mois * product.non_call_period_mois):
                dates_possibles.append(t)
                nouveaux_rappels = (~est_rappele) & (traj_dec[:, t] >= product.barriere_rappel)
                date_rappel[nouveaux_rappels] = t
                est_rappele = est_rappele | nouveaux_rappels
                
        dates_rappel_valid = date_rappel[date_rappel > 0]
        map_obs = {date: i+1 for i, date in enumerate(dates_possibles)}
        obs_rappel = np.array([map_obs[d] for d in dates_rappel_valid])
        
        unique_obs, counts = np.unique(obs_rappel, return_counts=True)
        
        fig_dist_rappel = go.Figure()
        fig_dist_rappel.add_trace(go.Bar(x=unique_obs, y=counts, name="Rappels", marker_color='green', hovertemplate="%{y} trajectoires<extra></extra>"))
        
        fig_dist_rappel.update_layout(
            title="Distribution des Périodes de Rappel (Autocall)",
            xaxis_title=f"Numéro d'observation (Fréquence : {product.frequence_obs_mois} mois)", 
            yaxis_title="Nombre de trajectoires",
            xaxis=dict(dtick=1)
        )
        
        return fig_dist_dec, fig_dist_pr, fig_dist_rappel

    def plot_binned_averages(self, bin_stats):
        import plotly.graph_objects as go
        
        labels = [b['label'] for b in bin_stats]
        dec_vals = [b['moy_dec'] for b in bin_stats]
        pr_vals = [b['moy_pr'] for b in bin_stats]        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=dec_vals,
            name='Decrement', marker_color='blue'
        ))
        fig.add_trace(go.Bar(
            x=labels, y=pr_vals,
            name='Price Return', marker_color='orange'
        ))
        
        fig.update_layout(
            title="Niveaux Moyens par Tranches de Centiles (Sous PDI)",
            xaxis_title="Tranches de Centiles",
            yaxis_title="Niveau Final Moyen (pts)",
            barmode='group',
            plot_bgcolor='white',
            hovermode="x unified"
        )
        
        return fig

if __name__ == "__main__":
    mes_regimes = [
        {"duree_annees": 3, "r_perf": 0.04, "vol": 0.15, "yield_initial": 0.03},
        {"duree_annees": 2, "r_perf": -0.15, "vol": 0.35, "yield_initial": 0.01},
        {"duree_annees": 5, "r_perf": 0.05, "vol": 0.18, "yield_initial": 0.04}
    ]
    
    scenario_krach = MarketScenario(config_regimes=mes_regimes, annees=10, jours_par_an=252)
    mon_indice_dec = DecrementIndex(niveau_initial=1000.0, decrement_annuel=50.0)
    pdi_pts = mon_indice_dec.niveau_initial * 0.50
    mon_autocall = AutocallProduct(barriere_rappel=1000.0, niveau_pdi=pdi_pts, non_call_period_mois=11, frequence_obs_mois=4, degressivite=1.0)
    
    moteur = SimulationEngine(nb_trajectoires=10000,seed=42)
    traj_pr, traj_dec, est_rappele = moteur.run(mon_indice_dec, scenario_krach, mon_autocall)
    
    nom_scenario = f"Scénario N-Périodes Test"
    reps_scen, bin_stats = moteur.afficher_statistiques(nom_scenario, traj_pr, traj_dec, est_rappele, mon_autocall, scenario_krach)
    
    fig1, fig2 = moteur.plot_results(nom_scenario, traj_pr, traj_dec, reps_scen, mon_autocall, scenario_krach, mon_indice_dec)
    fig1.show()
    fig2.show()
    if bin_stats:
        fig_binned = moteur.plot_binned_averages(bin_stats)
        fig_binned.show()
