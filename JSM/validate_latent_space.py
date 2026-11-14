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
# MAIN EXECUTION
# ==========================================

def compute_validation_matrices(model, tokenizer, formulas, device, num_traces=200, trace_length=10):
    """Genera tracce, calcola il ground truth e le predizioni del modello."""
    # 1. Genera tracce
    random_traces = [generate_random_trace(trace_length) for _ in range(num_traces)]
    trace_strings = [trace_to_string(t) for t in random_traces]

    # 2. Matrice Logica Ground Truth (Spot)
    n_formulas = len(formulas)
    M_gt = np.zeros((n_formulas, num_traces), dtype=int)
    for i, f_str in enumerate(formulas):
        f_ast = spot.formula(f_str)
        for j, trace in enumerate(random_traces):
            if eval_ltlf_ast(f_ast, trace, t=0):
                M_gt[i, j] = 1

    # 3. Estrazione Predizioni (PyTorch)
    with torch.no_grad():
        f_inputs = tokenizer(formulas, padding=True, truncation=True, return_tensors="pt").to(device)
        t_inputs = tokenizer(trace_strings, padding=True, truncation=True, return_tensors="pt").to(device)

        z_formulas = model.backbone.encode_formula(f_inputs["input_ids"], f_inputs["attention_mask"])
        z_traces = model.backbone.encode_trace(t_inputs["input_ids"], t_inputs["attention_mask"])

        # Prodotto cartesiano tramite repeat
        z_f_expanded = z_formulas.repeat_interleave(num_traces, dim=0)
        z_t_expanded = z_traces.repeat(n_formulas, 1)

        diff = torch.abs(z_f_expanded - z_t_expanded)
        features = torch.cat([z_f_expanded, z_t_expanded, diff], dim=-1)

        # Inferenza a blocchi per evitare OOM
        BATCH_SIZE = 10000
        probs_list = []
        for i in range(0, features.size(0), BATCH_SIZE):
            batch_feats = features[i: i + BATCH_SIZE]
            logits = model.classifier(batch_feats)
            batch_probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            probs_list.append(batch_probs)

        M_pred = torch.cat(probs_list).cpu().numpy().reshape((n_formulas, num_traces))

    return M_gt, M_pred
