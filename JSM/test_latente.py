import torch
import torch.nn.functional as F
import spot
import random
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score
from ltlfbert import LTLfBERT, ConformanceClassifier

# Importa l'architettura dal tuo file
from ltlfbert import LTLfBERT

# --- PARAMETRI ---
NUM_RANDOM_TRACES = 200
TRACE_LENGTH = 10
APS = ["p0", "p1", "p2", "p3"] # Le variabili corrette usate nel training
USE_LORA = True


# ==========================================
# 1. FUNZIONI DI SUPPORTO AGGIORNATE (Sincronizzate con generate_dataset.py)
# ==========================================


def generate_random_trace(length):
    """
    Genera una traccia casuale.
    Nel tuo script originale (random_trace), uno stato è un `set` contenente solo le props vere.
    Replichiamo esattamente quella logica.
    """
    trace = []
    for _ in range(length):
        state = {p for p in APS if random.random() > 0.5}
        trace.append(state)
    return trace

def trace_to_string(trace):
    """
    Converte la traccia in stringa testuale ESATTAMENTE come nel tuo `trace_to_str`.
    Output atteso: "1010 0111 0000"
    """
    return " ".join("".join("1" if p in s else "0" for p in APS) for s in trace)

def eval_ltlf_ast(f, trace, t):
    """
    Valutatore LTLf in puro Python.
    Adattato per lavorare con i `set` generati da `generate_random_trace`.
    """
    if t >= len(trace): return False
    kind = f.kind()
    if kind == spot.op_ap: return f.ap_name() in trace[t]
    elif kind == spot.op_Not: return not eval_ltlf_ast(f[0], trace, t)
    elif kind == spot.op_And: return all(eval_ltlf_ast(c, trace, t) for c in f)
    elif kind == spot.op_Or: return any(eval_ltlf_ast(c, trace, t) for c in f)
    elif kind == spot.op_X: return eval_ltlf_ast(f[0], trace, t + 1) if t + 1 < len(trace) else False
    elif kind == spot.op_F: return any(eval_ltlf_ast(f[0], trace, i) for i in range(t, len(trace)))
    elif kind == spot.op_G: return all(eval_ltlf_ast(f[0], trace, i) for i in range(t, len(trace)))
    elif kind == spot.op_U:
        for i in range(t, len(trace)):
            if eval_ltlf_ast(f[1], trace, i):
                if all(eval_ltlf_ast(f[0], trace, k) for k in range(t, i)):
                    return True
        return False
    return False

# ==========================================
# (Nel Blocco Main, prima di calcolare Ground Truth)
# Assicurati di passare APS a Spot se devi rigenerare formule, ma
# se carichi dataset_formulas.txt, controlla che usino p0, p1, p2, p3.
# ==========================================
# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in uso: {device}")

    # 1. Carica le formule generate precedentemente
    with open("dataset_formulas.txt", "r") as f:
        formulas = [line.strip() for line in f.readlines()]
    N_FORMULAS = len(formulas)

    # 2. Inizializza Modello e Tokenizer
    print("Caricamento di CodeBERT e LTLf-BERT...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # Inizializza l'architettura completa come salvata in ltlfbert_best.pt
    backbone = LTLfBERT(use_lora=USE_LORA)
    model = ConformanceClassifier(backbone=backbone)

    state_dict = torch.load("ltlfbert_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Genera le tracce casuali
    print(f"Generazione di {NUM_RANDOM_TRACES} tracce casuali per la validazione...")
    random_traces = [generate_random_trace(TRACE_LENGTH) for _ in range(NUM_RANDOM_TRACES)]
    trace_strings = [trace_to_string(t) for t in random_traces]

    # 4. Calcola la Matrice Booleana Ground Truth (Spot)
    print("Calcolo Matrice Logica Ground Truth (Spot)...")
    M_gt = np.zeros((N_FORMULAS, NUM_RANDOM_TRACES), dtype=int)
    for i, f_str in enumerate(formulas):
        f_ast = spot.formula(f_str)
        for j, trace in enumerate(random_traces):
            if eval_ltlf_ast(f_ast, trace, t=0):
                M_gt[i, j] = 1

    # 5. Estrazione Embedding e Calcolo Matrice Geometrica (LTLf-BERT)
    print("Estrazione Embedding Spazio Latente (PyTorch)...")
    with torch.no_grad():
        # Tokenizza le formule
        f_inputs = tokenizer(formulas, padding=True, truncation=True, return_tensors="pt").to(device)
        # Tokenizza le tracce
        t_inputs = tokenizer(trace_strings, padding=True, truncation=True, return_tensors="pt").to(device)

        # Aggiungi ".backbone." per estrarre gli embedding dall'encoder interno
        z_formulas = model.backbone.encode_formula(f_inputs["input_ids"], f_inputs["attention_mask"])
        z_traces = model.backbone.encode_trace(t_inputs["input_ids"], t_inputs["attention_mask"])

        # Calcola similarità coseno: prodotto scalare tra vettori normalizzati (da ProjectionHead)
        M_pred = torch.mm(z_formulas, z_traces.T).cpu().numpy()

    # 6. Valutazione Allineamento Spazio Latente (ROC-AUC)
    # Appiattiamo le matrici per confrontare ogni coppia (formula, traccia)
    y_true = M_gt.flatten()
    y_scores = M_pred.flatten()

    # Evita errori se la matrice booleana è tutta zeri o tutta uni per pura casualità
    if len(np.unique(y_true)) > 1:
        auc_score = roc_auc_score(y_true, y_scores)
        print("\n=== RISULTATI VALIDAZIONE SPAZIO LATENTE ===")
        print(f"ROC-AUC: {auc_score:.4f} (1.0 = Allineamento Perfetto, 0.5 = Casuale)")
    else:
        print("Errore: I modelli casuali generati non soddisfano nessuna formula o le soddisfano tutte.")
        print("Prova ad aumentare la lunghezza o il numero di tracce casuali.")