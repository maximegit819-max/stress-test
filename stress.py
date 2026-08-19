class IndiceDecrement:

    def __init__(self, niveau_initial, decrement_annuel, div_model):

        self.niveau_initial = niveau_initial
        self.decrement_annuel = decrement_annuel
        self.div_model = div_model
        self.ecart_div = div_model - decrement_annuel/niveau_initial    #c'est l'ecart div initial que l'on regarde si j'ai bien compris

    def calcule_baisse_quotidienne(self, pourcentage_annuel, annees, jours_par_an=252):
        total_jours = annees * jours_par_an
        
        r_perf = (1 + pourcentage_annuel) ** (1/jours_par_an) - 1
        r_div = (1 + self.div_model) ** (1/jours_par_an) - 1
        d_points = self.decrement_annuel / jours_par_an
        
        indice_pure = self.niveau_initial
        indice_decrement = self.niveau_initial
        
        for _ in range(total_jours):
            indice_pure = indice_pure * (1 + r_perf)
            indice_decrement = max(0.0, indice_decrement * (1 + r_perf + r_div) - d_points)
            
        return [
            (indice_pure / self.niveau_initial) * 100,
            (indice_decrement / self.niveau_initial) * 100,
            (indice_pure / self.niveau_initial) * 100 - (indice_decrement / self.niveau_initial) * 100
        ]


    def recherche_baisse(self, annees, cible_decr=50.0, tol=0.01, max_iter=100):


        points = [-0.1, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        
        low = None
        high = None
        
        vals = []
        for p in points:
            val = self.calcule_baisse_quotidienne(p, annees)[1] - cible_decr
            vals.append((p, val))
        
        # On cherche deux points consécutifs avec changement de signe
        for i in range(len(vals) - 1):
            p1, v1 = vals[i]
            p2, v2 = vals[i+1]
            if v1 * v2 <= 0:
                low = p1
                high = p2
                break
                

        if low is None:
            val_min = min(v for p, v in vals)
            val_max = max(v for p, v in vals)
            raise ValueError(
                f"Veuillez mettre d'autres proposition de baisse linéaire annuelle"
            )
            
        # Résolution par dichotomie (bisection)
        f_low = self.calcule_baisse_quotidienne(low, annees)[1] - cible_decr
        
        for _ in range(max_iter):
            mid = (low + high) / 2.0
            f_mid = self.calcule_baisse_quotidienne(mid, annees)[1] - cible_decr

                
            if abs(f_mid) < tol:
                return mid
                
            if f_low * f_mid < 0:
                high = mid
            else:
                low = mid
                f_low = f_mid
                
        return (low + high) / 2.0
