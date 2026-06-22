"""
LTLfBERT – Analisi dell'isomorfismo Hopfield–Attention
Visualizza la superficie energetica dello spazio latente e le basin di attrazione.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hopfield_energy(xi_patterns, beta):
    """
    Energia di una Modern Hopfield Network:
        E(x) = -beta^{-1} * logsumexp(beta * xi^T x) + 0.5 * x^T x + const
    xi_patterns: (M, D) – pattern memorizzati
    beta: temperatura inversa (= 1/sqrt(d_k) negli attention layers)
    """
    def energy(x):
        x = x / (np.linalg.norm(x) + 1e-8)
        dots = beta * (xi_patterns @ x)
        return -np.log(np.sum(np.exp(dots - dots.max()))) - dots.max() + 0.5 * np.dot(x, x)
    return energy


def plot_energy_surface_2d(patterns, labels, output_path, beta=1.0, resolution=100):
    """
    Proietta pattern in 2D (PCA) e visualizza la superficie energetica.
    Mostra come le basin di attrazione corrispondono alle classi LTLf.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(patterns)

    # Griglia 2D
    margin = 1.5
    x_min, x_max = proj[:,0].min()-margin, proj[:,0].max()+margin
    y_min, y_max = proj[:,1].min()-margin, proj[:,1].max()+margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution))

    # Calcola energia su griglia (ricostituendo il vettore pieno tramite PCA inverse)
    energy_fn = hopfield_energy(patterns, beta)
    Z = np.zeros(xx.shape)
    for i in range(resolution):
        for j in range(resolution):
            pt_2d = np.array([xx[i,j], yy[i,j]])
            pt_full = pca.inverse_transform(pt_2d.reshape(1,-1))[0]
            Z[i,j] = energy_fn(pt_full)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, Z, levels=30, cmap="RdYlBu_r", alpha=0.7)
    plt.colorbar(contour, ax=ax, label="Energia E(x)")

    colors = ["#e74c3c" if l == 1 else "#3498db" for l in labels]
    ax.scatter(proj[:,0], proj[:,1], c=colors, s=20, zorder=5, alpha=0.8, edgecolors="white", linewidths=0.5)

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#e74c3c", label="SAT"),
        mpatches.Patch(color="#3498db", label="UNSAT"),
    ], fontsize=10)
    ax.set_title(f"Superficie Energetica Hopfield (β={beta:.2f})\nMinimi locali = classi semantiche LTLf", fontsize=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Superficie energetica salvata: {output_path}")


def attention_as_hopfield_update(Q, K, V, beta=None):
    """
    Dimostra formalmente che l'attention è un passo di update Hopfield.
    Q: (seq, d_k)  query
    K: (M, d_k)    keys = pattern memorizzati
    V: (M, d_v)    values
    Restituisce il nuovo stato x_{t+1} = softmax(beta * K Q^T) V
    """
    d_k = Q.shape[-1]
    if beta is None:
        beta = 1.0 / np.sqrt(d_k)
    scores = beta * (Q @ K.T)          # (seq, M)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)  # softmax
    return weights @ V                  # (seq, d_v)
