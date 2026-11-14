import numpy as np
from scipy.spatial.distance import pdist


def evaluate_incidence_matrix(matrix_path="ltlf_incidence_matrix.npy"):
    M = np.load(matrix_path).astype(float)
    N = M.shape[0]

    # --- 1. METRICHE PER LE FORMULE (Righe) ---
    # Quante tracce soddisfa ogni formula (idealmente vicino a 1)
    formula_specificity = np.sum(M, axis=1)
    # Quante formule sono "quasi tautologie" (accettano > 50% delle tracce)
    trivial_formulas = np.sum(formula_specificity > (0.5 * N))

    # --- 2. METRICHE PER I MODELLI/TRACCE (Colonne) ---
    # Quante formule soddisfa ogni singola traccia
    trace_promiscuity = np.sum(M, axis=0)

    # --- 3. METRICHE GLOBALI DELLA MATRICE ---
    # a) Densità extra-diagonale
    off_diag_mask = ~np.eye(N, dtype=bool)
    off_diag_density = np.mean(M[off_diag_mask])

    # b) Overlap semantico medio (Similarità di Jaccard tra coppie di formule)
    # pdist calcola la distanza di Jaccard (1 - similarità)
    jaccard_distances = pdist(M, metric='jaccard')
    mean_jaccard_similarity = 1.0 - np.nanmean(jaccard_distances)

    # c) Rango algebrico e Rango Effettivo (SVD Spectral Entropy)
    singular_values = np.linalg.svd(M, compute_uv=False)
    algebraic_rank = np.linalg.matrix_rank(M)

    # Calcolo dell'Effective Rank (Roy & Vetterli)
    sv_pos = singular_values[singular_values > 1e-12]
    p = sv_pos / np.sum(sv_pos)
    spectral_entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(spectral_entropy)

    print("=== REPORT QUALITÀ DATASET LTLf ===")
    print(f"Dimensione Matrice:          {N} x {N}")
    print(f"Traccia (Diagonale):         {int(np.trace(M))} / {N}\n")

    print("--- 1. BONTÀ FORMULE (Righe) ---")
    print(
        f"Tracce accettate per formula (Media ± Std): {np.mean(formula_specificity):.2f} ± {np.std(formula_specificity):.2f}")
    print(f"Mediana tracce accettate:                   {np.median(formula_specificity):.1f} (Ottimo: 1.0)")
    print(f"Formule troppo lasche (>50% tracce):        {trivial_formulas}\n")

    print("--- 2. BONTÀ MODELLI/TRACCE (Colonne) ---")
    print(
        f"Formule soddisfatte per traccia (Media ± Std): {np.mean(trace_promiscuity):.2f} ± {np.std(trace_promiscuity):.2f}")
    print(f"Mediana formule soddisfatte:                   {np.median(trace_promiscuity):.1f} (Ottimo: 1.0)\n")

    print("--- 3. ORTOGONALITÀ GLOBALE MATRICE ---")
    print(f"Densità Extra-Diagonale:     {off_diag_density:.4f} (Ottimo -> 0)")
    print(f"Jaccard Overlap Medio:       {mean_jaccard_similarity:.4f} (Ottimo -> 0)")
    print(f"Rango Algebrico:             {algebraic_rank} / {N}")
    print(f"Rango Effettivo (SVD):       {effective_rank:.2f} / {N} (Più alto = più ortogonale)")

    return {
        "formula_scores": formula_specificity,
        "trace_scores": trace_promiscuity,
        "off_diag_density": off_diag_density,
        "mean_jaccard": mean_jaccard_similarity,
        "effective_rank": effective_rank
    }


if __name__ == "__main__":
    evaluate_incidence_matrix()