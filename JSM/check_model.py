import torch

# Carico il file mappandolo sulla CPU per sicurezza
loaded_file = torch.load("ltlfbert_best.pt", map_location="cpu", weights_only=False)

print(f"Tipo di oggetto caricato: {type(loaded_file)}\n")

if isinstance(loaded_file, dict):
    print("Risultato: Il file contiene uno STATE_DICT (solo i pesi).")
    print("Prime 5 chiavi salvate nel dizionario:")
    for k in list(loaded_file.keys())[:5]:
        print(f" - {k}")
else:
    print("Risultato: Il file contiene l'INTERO OGGETTO MODELLO (la classe con l'architettura).")