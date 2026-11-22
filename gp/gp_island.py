#!/usr/bin/env python3
"""
===============================================================================
gp_island.py - Algoritmo Genetico a Isole per Formule LTLf Ortogonali
===============================================================================

Questo script implementa un'architettura ad Isole (Island Model / Coarse-Grained
Parallelism) per evolvere formule LTLf massimamente ortogonali tra loro.

Perché questa architettura?
---------------------------
1. Parallelismo reale su HPC:
   Invece di usare un solo thread su 8 core, creiamo 8 popolazioni indipendenti
   ("isole"), ognuna mappata su un processo CPU separato.
2. Superamento della saturazione dello spazio (Speciation):
   Le 8 isole sono organizzate in 4 nicchie semantiche distinte (Safety,
   Liveness, Step Dynamics, Rare Triggers). Ognuna esplora una regione diversa
   dello spazio temporale, ritardando ed evitando il collasso della diversità.
3. Topologia ad Anello Bipartito:
   - Scambio intra-nicchia frequente (tra isole gemelle per affinare la nicchia).
   - Scambio inter-nicchia periodico (anello globale per favorire formule composite,
     es. combinare pattern Safety 'G' con Liveness 'F' per generare Response patterns).

Uso tipico:
  python gp/gp_island.py --n-formulas 5000 --max-len 100 --output-dir data_gp_island
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

# Assicura che la root del repository e gp/ siano sempre nel PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GP_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [BASE_DIR, GP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from deap import base, creator, gp, tools

# Importiamo le funzioni di utilità già collaudate in gp_ltlf
from gp_ltlf import (
    AP,
    MAX_LEN,
    MIN_LEN,
    build_dataset,
    make_trace_pool,
    sample_traces_fast,
    sem,
    signature,
    similarity,
    to_nested,
    to_spot_str,
)

# -----------------------------------------------------------------------------
# 1. Definizione delle Nicchie Semantiche
# -----------------------------------------------------------------------------
# Ogni coppia di isole ha una "personalità" logica diversa, definita da:
# - sat_lo / sat_hi: range accettabile di soddisfacibilità sulle tracce.
# - favored_ops: operatori che ricevono un bonus di fitness nell'isola.
NICHES = {
    0: {"name": "Safety-A",   "type": "safety",   "sat_lo": 0.25, "sat_hi": 0.85, "favored": ["G", "IMPL", "AND"]},
    1: {"name": "Safety-B",   "type": "safety",   "sat_lo": 0.25, "sat_hi": 0.85, "favored": ["G", "IMPL", "AND"]},
    2: {"name": "Liveness-A", "type": "liveness", "sat_lo": 0.20, "sat_hi": 0.80, "favored": ["F", "U", "OR"]},
    3: {"name": "Liveness-B", "type": "liveness", "sat_lo": 0.20, "sat_hi": 0.80, "favored": ["F", "U", "OR"]},
    4: {"name": "Step-A",     "type": "step",     "sat_lo": 0.20, "sat_hi": 0.80, "favored": ["X", "IMPL", "AND"]},
    5: {"name": "Step-B",     "type": "step",     "sat_lo": 0.20, "sat_hi": 0.80, "favored": ["X", "IMPL", "AND"]},
    6: {"name": "Rare-A",     "type": "rare",     "sat_lo": 0.10, "sat_hi": 0.35, "favored": ["AND", "NOT", "OR"]},
    7: {"name": "Rare-B",     "type": "rare",     "sat_lo": 0.10, "sat_hi": 0.35, "favored": ["AND", "NOT", "OR"]},
}


# -----------------------------------------------------------------------------
# 2. Configurazione DEAP Globale
# -----------------------------------------------------------------------------
def build_shared_pset():
    """Grammatica completa LTLf condivisa da tutte le isole.
    
    Avere una grammatica condivisa garantisce che un individuo nato sull'isola
    Safety possa migrare sull'isola Liveness ed essere accoppiato via crossover
    senza errori di tipo.
    """
    pset = gp.PrimitiveSet("LTLF", 0)
    pset.addPrimitive(lambda a: a, 1, name="NOT")
    pset.addPrimitive(lambda a: a, 1, name="X")
    pset.addPrimitive(lambda a: a, 1, name="F")
    pset.addPrimitive(lambda a: a, 1, name="G")
    pset.addPrimitive(lambda a, b: a, 2, name="AND")
    pset.addPrimitive(lambda a, b: a, 2, name="OR")
    pset.addPrimitive(lambda a, b: a, 2, name="IMPL")
    pset.addPrimitive(lambda a, b: a, 2, name="U")
    for p in AP:
        pset.addTerminal(p, name=p)
    return pset


# Inizializziamo i tipi DEAP una sola volta a livello di modulo
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


# -----------------------------------------------------------------------------
# 3. Valutatore Locale dell'Isola (Fitness Specialistica)
# -----------------------------------------------------------------------------
class IslandEvaluator:
    """Valuta gli individui su una specifica isola, combinando:
    1. Ortogonalità rispetto all'archivio globale (massimizzare distanza).
    2. Range di soddisfacibilità proprio della nicchia.
    3. Premialità per gli operatori caratteristici dell'isola.
    4. Parsimonia sulla dimensione dell'albero per prevenire il bloat.
    """
    def __init__(self, vals, valid, archive, niche_cfg, size_penalty=0.003):
        self.vals = vals
        self.valid = valid
        self.archive = archive  # matrice booleana (K, T) delle formule già accettate
        self.cfg = niche_cfg
        self.size_penalty = size_penalty
        self.cache = {}

    def sig_of(self, ind):
        nested = to_nested(ind)
        s = to_spot_str(nested)
        if s not in self.cache:
            # Protezione memoria: la cache locale dell'isola non supera mai 10.000 formule
            if len(self.cache) >= 10000:
                self.cache.clear()
            self.cache[s] = signature(nested, self.vals, self.valid)
        return s, self.cache[s]

    def __call__(self, ind):
        s, sig = self.sig_of(ind)
        rate = sig.mean()

        # Penalità di soddisfacibilità specifica della nicchia
        pen = 0.0
        if rate < self.cfg["sat_lo"]:
            pen += (self.cfg["sat_lo"] - rate) * 3.5
        if rate > self.cfg["sat_hi"]:
            pen += (rate - self.cfg["sat_hi"]) * 3.5

        # Bonus di nicchia: incoraggia la comparsa degli operatori target dell'isola
        bonus = 0.0
        tree_str = str(ind)
        for op in self.cfg["favored"]:
            if op in tree_str:
                bonus += 0.04

        # Penalità di similarità rispetto all'archivio globale
        sim = similarity(sig, self.archive) if self.archive.size > 0 else 0.0

        # Fitness totale: vogliamo minimizzare similarità, penalità e dimensione
        fitness = -sim - pen + bonus - (self.size_penalty * len(ind))
        return (fitness,)


# -----------------------------------------------------------------------------
# 4. Evoluzione di una Singola Isola (Funzione eseguita in parallelo)
# -----------------------------------------------------------------------------
def run_island_epoch(island_id, serialized_pop, archive, vals, valid, n_gen,
                     pop_size, max_height, seed):
    """Esegue un'epoca di evoluzione su una singola isola (core CPU).
    
    Riceve la popolazione serializzata (per compatibilità multiprocessing pulita),
    svolge n_gen generazioni, seleziona i candidati migliori da proporre
    all'archivio globale, e restituisce la nuova popolazione pronta per la migrazione.
    """
    random.seed(seed)
    np_rng = np.random.default_rng(seed)

    niche_cfg = NICHES[island_id % len(NICHES)]
    pset = build_shared_pset()

    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)

    # Limite di altezza per evitare alberi spropositati
    for op_name in ("mate", "mutate"):
        tb.decorate(op_name, gp.staticLimit(key=lambda ind: ind.height, max_value=max_height))

    evaluator = IslandEvaluator(vals, valid, archive, niche_cfg)
    tb.register("evaluate", evaluator)

    # Ricostruiamo o inizializziamo la popolazione
    if serialized_pop:
        pop = []
        for s in serialized_pop:
            try:
                # Ricrea l'albero DEAP dalla stringa serializzata
                ind = creator.Individual.from_string(s, pset)
            except Exception:
                ind = tb.individual()
            pop.append(ind)
        # Se mancano individui, riempiamo fino a pop_size
        while len(pop) < pop_size:
            pop.append(tb.individual())
    else:
        pop = tb.population(n=pop_size)

    # Valutiamo gli individui iniziali privi di fitness valida
    for ind in pop:
        if not ind.fitness.valid:
            ind.fitness.values = tb.evaluate(ind)

    # Ciclo generazionale locale dell'isola
    cxpb, mutpb = 0.7, 0.3
    for _ in range(n_gen):
        offspring = list(map(tb.clone, tb.select(pop, len(pop))))
        for a, b in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
        for m in offspring:
            if random.random() < mutpb:
                tb.mutate(m)
                del m.fitness.values
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = tb.evaluate(ind)

        # Elitismo locale: i 3 migliori sopravvivono direttamente
        pop = tools.selBest(pop, 3) + offspring[:-3]

    # Estraiamo i candidati dell'isola: individui validi per l'archivio globale
    candidates = []
    seen_local = set()
    for ind in tools.selBest(pop, min(len(pop), 15)):
        s, sig = evaluator.sig_of(ind)
        if s in seen_local:
            continue
        rate = sig.mean()
        # Solo formule che non siano trivialmente sempre vere o sempre false
        if 0.05 <= rate <= 0.95:
            candidates.append((s, len(ind), sig))
            seen_local.add(s)

    # Serializziamo la popolazione finale in stringhe per la restituzione al Master
    final_pop_strings = [str(ind) for ind in pop]

    return island_id, final_pop_strings, candidates


# -----------------------------------------------------------------------------
# 5. Topologia e Migrazione
# -----------------------------------------------------------------------------
def apply_migration(populations, epoch, intra_rate=2, inter_rate=1):
    """Applica la migrazione ad anello bipartito tra le popolazioni delle isole.
    
    1. Scambio Intra-Nicchia (ogni epoca):
       Le isole gemelle (0<->1, 2<->3, 4<->5, 6<->7) si scambiano i top individui.
    2. Scambio Inter-Nicchia (ogni 2 epoche):
       Ponte tra nicchie ad anello: 1 -> 2, 3 -> 4, 5 -> 6, 7 -> 0.
    """
    n_islands = len(populations)

    # 1. Intra-nicchia (isole gemelle a coppie)
    for i in range(0, n_islands, 2):
        j = i + 1
        if j < n_islands and populations[i] and populations[j]:
            # I top individui della prima vanno in coda alla seconda (e viceversa)
            migrants_i = populations[i][:intra_rate]
            migrants_j = populations[j][:intra_rate]

            # Rimpiazzo sui peggiori (in fondo alla lista)
            populations[i] = populations[i][:-intra_rate] + migrants_j
            populations[j] = populations[j][:-intra_rate] + migrants_i

    # 2. Inter-nicchia (anello globale ogni 2 epoche)
    if epoch % 2 == 0:
        bridges = [(1, 2), (3, 4), (5, 6), (7, 0)]
        for src, dst in bridges:
            if src < n_islands and dst < n_islands and populations[src] and populations[dst]:
                migrant = populations[src][:inter_rate]
                populations[dst] = populations[dst][:-inter_rate] + migrant

    return populations


def get_executor(n_workers):
    """Restituisce un ProcessPoolExecutor per vero parallelismo multi-core su HPC.
    
    Se l'ambiente locale ha restrizioni sui semafori POSIX (es. container o sandbox),
    effettua un fallback automatico su ThreadPoolExecutor per consentire il debug locale.
    """
    try:
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
        # Test di inizializzazione
        return ex, "process (multi-core reale)"
    except Exception:
        return concurrent.futures.ThreadPoolExecutor(max_workers=n_workers), "thread (fallback locale)"


# -----------------------------------------------------------------------------
# 6. Main: Coordinatore Master (Algoritmo Principale)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-formulas", type=int, default=5000,
                        help="Numero target di formule ortogonali da generare (default: 5000)")
    parser.add_argument("--n-islands", type=int, default=8,
                        help="Numero di isole parallele / processi CPU (default: 8)")
    parser.add_argument("--pop-per-island", type=int, default=60,
                        help="Dimensione della popolazione per ciascuna isola (default: 60)")
    parser.add_argument("--gen-per-epoch", type=int, default=10,
                        help="Generazioni DEAP per isola prima di migrazione e raccolta (default: 10)")
    parser.add_argument("--max-epochs", type=int, default=2000,
                        help="Numero massimo di epoche globali prima dell'arresto (default: 2000)")
    parser.add_argument("--resume", action="store_true",
                        help="Riprende dall'archivio formulas.json esistente nella cartella output senza ricominciare da zero")
    parser.add_argument("--max-stall-epochs", type=int, default=20,
                        help="Epoche consecutive senza nuove formule prima dello stop per saturazione (default: 20)")
    parser.add_argument("--sim-thr", type=float, default=0.6,
                        help="Soglia massima di similarità per entrare nell'archivio (default: 0.6)")
    parser.add_argument("--pool-size", type=int, default=512,
                        help="Numero di tracce nel pool di valutazione delle firme (default: 512)")
    parser.add_argument("--min-len", type=int, default=MIN_LEN,
                        help="Lunghezza minima delle tracce (default: 3)")
    parser.add_argument("--max-len", type=int, default=MAX_LEN,
                        help="Lunghezza massima delle tracce / orizzonte temporale (default: 20 o 100)")
    parser.add_argument("--max-height", type=int, default=7,
                        help="Altezza massima degli alberi LTLf (default: 7)")
    parser.add_argument("--traces-per-formula", type=int, default=4,
                        help="Tracce campionate per formula nel dataset finale (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed iniziale per la riproducibilità (default: 42)")
    parser.add_argument("--output-dir", type=str, default="data_gp",
                        help="Cartella di destinazione dei dataset e delle formule (default: data_gp)")
    args = parser.parse_args()

    # Inizializzazione pool di tracce di riferimento
    rng = np.random.default_rng(args.seed)
    print("=" * 75)
    print("🚀 AVVIO GP AD ISOLE: Generazione Formule LTLf Ortogonali")
    print(f"  • Isole parallele (CPU Core): {args.n_islands}")
    print(f"  • Popolazione totale:         {args.n_islands * args.pop_per_island} ({args.pop_per_island} per isola)")
    print(f"  • Orizzonte temporale tracce: min_len={args.min_len}, max_len={args.max_len}")
    print(f"  • Soglia similarità phi:      <= {args.sim_thr}")
    print(f"  • Obiettivo formule:          {args.n_formulas}")
    executor, exec_mode = get_executor(args.n_islands)
    print(f"  • Modalità esecutore:         {exec_mode}")
    print("=" * 75)

    vals, valid = make_trace_pool(args.pool_size, rng, min_len=args.min_len, max_len=args.max_len)

    # Strutture dati del Master
    archive_sigs = np.zeros((0, args.pool_size), dtype=bool)
    accepted_formulas = []  # Lista di tuple: (stringa_formula, dimensione_albero)
    seen_formulas = set()
    populations = [[] for _ in range(args.n_islands)]

    t0 = time.perf_counter()
    stall_epochs = 0
    resolved_output = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(BASE_DIR, args.output_dir)
    os.makedirs(resolved_output, exist_ok=True)

    # Ripresa opzionale da checkpoint precedente
    checkpoint_file = os.path.join(resolved_output, "formulas.json")
    if args.resume and os.path.exists(checkpoint_file):
        print(f"📥 Ripresa attiva: caricamento archivio precedente da {checkpoint_file}...", flush=True)
        try:
            with open(checkpoint_file) as fp:
                saved = json.load(fp)
            from compare_datasets import spot_to_nested
            batch_sigs = []
            for f, size in saved:
                if f not in seen_formulas:
                    seen_formulas.add(f)
                    accepted_formulas.append((f, size))
                    nested = spot_to_nested(f)
                    sig = signature(nested, vals, valid)
                    batch_sigs.append(sig)
            if batch_sigs:
                archive_sigs = np.vstack(batch_sigs)
            print(f"✅ Caricate con successo {len(accepted_formulas)} formule nell'archivio globale!", flush=True)
        except Exception as e:
            print(f"⚠️ Errore nel caricamento del checkpoint: {e}. Si riparte da zero.", flush=True)

    with executor:
        for epoch in range(1, args.max_epochs + 1):
            t_epoch_start = time.perf_counter()

            # Lanciamo tutte le isole in parallelo
            futures = []
            for island_id in range(args.n_islands):
                f = executor.submit(
                    run_island_epoch,
                    island_id=island_id,
                    serialized_pop=populations[island_id],
                    archive=archive_sigs,
                    vals=vals,
                    valid=valid,
                    n_gen=args.gen_per_epoch,
                    pop_size=args.pop_per_island,
                    max_height=args.max_height,
                    seed=args.seed + (epoch * 100) + island_id,
                )
                futures.append(f)

            # Raccogliamo i risultati al termine dell'epoca
            epoch_candidates = []
            for f in concurrent.futures.as_completed(futures):
                island_id, new_pop, candidates = f.result()
                populations[island_id] = new_pop
                epoch_candidates.extend(candidates)

            # -----------------------------------------------------------------
            # Fase di Raccolta (Harvesting) sull'Archivio Globale
            # -----------------------------------------------------------------
            newly_added = 0
            # Ordiniamo i candidati per dimensione (preferenza per formule compatte)
            epoch_candidates.sort(key=lambda x: x[1])

            for s, size, sig in epoch_candidates:
                if s in seen_formulas:
                    continue
                # Controllo di ortogonalità contro l'archivio globale
                if similarity(sig, archive_sigs) <= args.sim_thr:
                    archive_sigs = np.vstack([archive_sigs, sig[None, :]])
                    seen_formulas.add(s)
                    accepted_formulas.append((s, size))
                    newly_added += 1
                    if len(accepted_formulas) >= args.n_formulas:
                        break

            # -----------------------------------------------------------------
            # Fase di Migrazione (Topologia Bipartita)
            # -----------------------------------------------------------------
            populations = apply_migration(populations, epoch)

            # Gestione stallo
            if newly_added > 0:
                stall_epochs = 0
            else:
                stall_epochs += 1

            elapsed = time.perf_counter() - t0
            t_epoch = time.perf_counter() - t_epoch_start
            print(f"[Epoca {epoch:3d}/{args.max_epochs}] "
                  f"Formule: {len(accepted_formulas):5d}/{args.n_formulas} "
                  f"(+{newly_added:2d}) | "
                  f"Stall: {stall_epochs:2d}/{args.max_stall_epochs} | "
                  f"Tempo epoca: {t_epoch:.1f}s | Tot: {elapsed:.0f}s",
                  flush=True)

            # Checkpoint intermedio su disco ogni 10 epoche
            if epoch % 10 == 0 or len(accepted_formulas) >= args.n_formulas:
                with open(os.path.join(resolved_output, "formulas.json"), "w") as fp:
                    json.dump(accepted_formulas, fp, indent=2)

            # Condizioni di arresto
            if len(accepted_formulas) >= args.n_formulas:
                print(f"\n🎯 Obiettivo raggiunto: {len(accepted_formulas)} formule ortogonali trovate!")
                break
            if stall_epochs >= args.max_stall_epochs:
                print(f"\n⚠️ Arresto per saturazione spazio: nessuna formula ortogonale trovata per {stall_epochs} epoche consecutive.")
                break

    # -------------------------------------------------------------------------
    # 7. Generazione Dataset Finale (train/val/test/ood)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"Salvataggio finale in: {resolved_output}")
    with open(os.path.join(resolved_output, "formulas.json"), "w") as fp:
        json.dump(accepted_formulas, fp, indent=2)

    print(f"Campionamento tracce positive e negative ({args.traces_per_formula} tracce per formula)...")
    build_dataset(
        formulas=accepted_formulas,
        output_dir=resolved_output,
        traces_per_formula=args.traces_per_formula,
        seed=args.seed,
        min_len=args.min_len,
        max_len=args.max_len,
        use_fast_sampler=True,
    )
    print(f"🎉 Processo completato con successo in {time.perf_counter() - t0:.1f}s!")
    print("=" * 75)


if __name__ == "__main__":
    main()
