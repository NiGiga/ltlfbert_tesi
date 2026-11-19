import random
import spot
import numpy as np
import json

# 1. PARAMETRI DEL DATASET
NUM_FORMULAS = 1000
MAX_DEPTH = 4
APS = ["p0", "p1", "p2", "p3"]  # Le variabili proposizionali


# ==========================================
# FASE 1: GENERAZIONE FORMULE
# ==========================================
def generate_random_ltlf(depth):
    """Genera una formula LTLf casuale limitando la profondità dell'AST."""
    if depth <= 0:
        return random.choice(APS)

    # Pesi: 30% Variabili, 35% Unari, 35% Binari
    op_type = random.choices(['ap', 'unary', 'binary'], weights=[0.3, 0.35, 0.35])[0]

    if op_type == 'ap':
        return random.choice(APS)
    elif op_type == 'unary':
        op = random.choice(['!', 'X', 'F', 'G'])
        child = generate_random_ltlf(depth - 1)
        return f"{op}({child})"
    else:
        op = random.choice(['&', '|', 'U'])
        left = generate_random_ltlf(depth - 1)
        right = generate_random_ltlf(depth - 1)
        return f"({left}) {op} ({right})"


def get_satisfiable_dataset(target_size, depth):
    """Crea un set di formule uniche e sintatticamente soddisfacibili."""
    formulas = set()
    print("Generazione formule in corso...")
    while len(formulas) < target_size:
        f_str = generate_random_ltlf(depth)
        # Spot semplifica la formula (rimuove ridondanze come a | a)
        f_spot = spot.simplify(spot.formula(f_str))

        # Saltiamo tautologie (1) o contraddizioni (0)
        if f_spot.is_tt() or f_spot.is_ff():
            continue

        # Verifichiamo se è soddisfacibile su tracce finite (LTLf)
        aut = spot.translate(f_spot.to_str(), "finite")
        if not aut.is_empty():
            formulas.add(f_spot.to_str())

    return list(formulas)


# ==========================================
# FASE 2: ESTRAZIONE TRACCE (MODELLI)
# ==========================================
def extract_finite_trace(formula_str):
    """Usa Spot per trovare un modello accettante e lo converte in array Python."""
    aut = spot.translate(formula_str, "finite")
    run = aut.accepting_run()

    # Spot formatta la traccia come stringa: "a & !b; 1; cycle{0}"
    word_str = str(spot.twa_word(run))

    # Prendiamo solo il prefisso finito per LTLf
    prefix = word_str.split("; cycle")[0]

    trace = []
    for step in prefix.split(";"):
        step = step.strip()
        state = {ap: False for ap in APS}  # Default tutto False

        if step and step != "1":  # Se non è un "don't care" totale
            parts = step.replace('(', '').replace(')', '').split('&')
            for p in parts:
                p = p.strip()
                if p.startswith('!'):
                    if p[1:] in APS: state[p[1:]] = False
                elif p in APS:
                    state[p] = True
        trace.append(state)

    return trace


# ==========================================
# FASE 3: VALUTATORE LTLf IN PYTHON
# ==========================================
def eval_ltlf_ast(f, trace, t):
    """Valuta in modo ricorsivo l'AST di Spot su una traccia finita (Semantica LTLf)."""
    # Se il tempo eccede la traccia finita
    if t >= len(trace):
        return False

    kind = f.kind()

    if kind == spot.op_ap:
        return trace[t].get(f.ap_name(), False)
    elif kind == spot.op_Not:
        return not eval_ltlf_ast(f[0], trace, t)
    elif kind == spot.op_And:
        return all(eval_ltlf_ast(child, trace, t) for child in f)
    elif kind == spot.op_Or:
        return any(eval_ltlf_ast(child, trace, t) for child in f)
    elif kind == spot.op_X:
        # Strong Next in LTLf: falso alla fine della traccia
        return eval_ltlf_ast(f[0], trace, t + 1) if t + 1 < len(trace) else False
    elif kind == spot.op_F:
        return any(eval_ltlf_ast(f[0], trace, i) for i in range(t, len(trace)))
    elif kind == spot.op_G:
        return all(eval_ltlf_ast(f[0], trace, i) for i in range(t, len(trace)))
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
if __name__ == "__main__":
    # 1. Genera le formule
    formulas = get_satisfiable_dataset(NUM_FORMULAS, MAX_DEPTH)
    print(f"Generate {len(formulas)} formule LTLf uniche e soddisfacibili.")

    # 2. Genera la traccia per ciascuna formula
    print("Estrazione delle tracce accettanti in corso...")
    traces = [extract_finite_trace(f) for f in formulas]

    # 3. Costruisci la Matrice di Incidenza (1000 x 1000)
    print("Costruzione della matrice di valutazione incrociata...")
    M = np.zeros((NUM_FORMULAS, NUM_FORMULAS), dtype=int)

    for i in range(NUM_FORMULAS):
        f_ast = spot.formula(formulas[i])  # Parsiamo l'AST una sola volta per riga
        for j in range(NUM_FORMULAS):
            # Valuta la traccia j sulla formula i
            if eval_ltlf_ast(f_ast, traces[j], t=0):
                M[i, j] = 1

    # 4. Sanity Check
    diagonal_sum = np.trace(M)
    print(f"\nVerifica Diagonale: {diagonal_sum} / {NUM_FORMULAS} (Deve essere {NUM_FORMULAS})")

    density = (np.sum(M) - diagonal_sum) / (NUM_FORMULAS * (NUM_FORMULAS - 1))
    print(f"Densità extra-diagonale: {density:.4f} (Più vicino a 0 è, più le formule sono ortogonali)")

    # 5. Salva i dati per il passaggio successivo con LTLf-BERT
    np.save("ltlf_incidence_matrix.npy", M)
    with open("dataset_formulas.txt", "w") as file:
        for f in formulas:
            file.write(f + "\n")

    print("Matrice e formule salvate con successo.")

    # json per modello
    with open("dataset_traces.json", "w") as f:
        json.dump(traces, f)