import numpy as np
import plotly.graph_objects as go

def calculer_pdi_equivalent(traj_pr, traj_dec, est_rappele, spot, pdi_dec_pct):
    """
    Calcule le niveau de PDI sur l'indice PR qui donne la même Espérance de Perte (EL)
    que le Decrement avec son pdi_dec_pct.
    """
    # 1. Espérance de perte du Decrement
    pdi_abs_dec = spot * pdi_dec_pct
    
    # Perte = 0 si rappelé ou niveau final > PDI
    # Sinon Perte = 1 - (Niveau Final / Spot)
    pertes_dec = np.where(
        (~est_rappele) & (traj_dec[:, -1] < pdi_abs_dec),
        1.0 - (traj_dec[:, -1] / spot),
        0.0
    )
    # On force à 0 toute perte négative (impossible normalement avec la condition < PDI, mais par sécurité)
    pertes_dec = np.clip(pertes_dec, 0.0, None)
    el_dec = np.mean(pertes_dec)
    
    if el_dec == 0:
        return pdi_dec_pct, 0.0
        
    # 2. Chercher le PDI équivalent sur le PR par dichotomie (Bisection)
    # L'espérance de perte PR augmente de façon monotone avec le niveau de PDI.
    
    def calc_el_pr(p):
        pdi_abs_pr = spot * p
        pertes_pr = np.where(
            (~est_rappele) & (traj_pr[:, -1] < pdi_abs_pr),
            1.0 - (traj_pr[:, -1] / spot),
            0.0
        )
        return np.mean(np.clip(pertes_pr, 0.0, None))
        
    low = pdi_dec_pct
    high = 1.0  # On plafonne la recherche à 100% de barrière
    
    # Vérification des bornes
    el_high = calc_el_pr(high)
    if el_high <= el_dec:
        # Même avec une barrière à 100%, le PR reste moins risqué que le Decrement
        return high, el_dec
        
    tolerance = 1e-3
    max_iter = 50
    
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        el_mid = calc_el_pr(mid)
        
        diff = el_mid - el_dec
        if abs(diff) < tolerance:
            return mid, el_dec
            
        if diff < 0:
            # La perte du PR est trop faible, il faut remonter la barrière PDI
            low = mid
        else:
            # La perte du PR est trop forte, il faut baisser la barrière PDI
            high = mid
            
    return (low + high) / 2.0, el_dec


def plot_courbe_pdi_equivalent(spots_test, pdis_equivalents, pdi_dec_pct):
    """
    Trace le 4ème graphique (PDI équivalent PR en fonction du Spot Initial)
    """
    fig = go.Figure()
    
    # Ligne de base (PDI Decrement)
    fig.add_trace(go.Scatter(
        x=spots_test, y=[pdi_dec_pct * 100] * len(spots_test),
        mode='lines', name=f'PDI Initial (Decrement) : {pdi_dec_pct*100:.0f}%',
        line=dict(color='gray', dash='dash')
    ))
    
    # Courbe du PDI équivalent
    fig.add_trace(go.Scatter(
        x=spots_test, y=np.array(pdis_equivalents) * 100,
        mode='lines+markers', name='PDI Équivalent sur Price Return (%)',
        line=dict(color='purple', width=3), marker=dict(symbol='star', size=8)
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>4. PDI Équivalent sur Indice Standard (Iso-Risque Expected Loss)</b>", font=dict(size=18)),
        xaxis=dict(title="Niveau du Spot Initial (pts)", showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title="Barrière PDI Équivalente (%)", showgrid=True, gridcolor='lightgray'),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        plot_bgcolor='white', hovermode="x unified", height=500
    )
    
    return fig
