import torch
from pathlib import Path
from config import get_config
from model import build_transformer
from temperature import zero_attention_head
from train import get_model, get_ds

# Test per verificare l'azzeramento della prima head del primo layer del decoder

# Config e device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = get_config()

# Costruisci il modello
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, validation=True)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

# Carica i pesi
model_path = Path('opus_books_weights') / 'tmodel_30.pt'
state = torch.load(model_path, map_location=device)
model.load_state_dict(state['model_state_dict'])

print("✓ Modello caricato con successo")
print(f"Percorso: {model_path}")

# Prendi la head del layer e prima dell'azzeramento
layer_0 = model.decoder.layers[0].cross_attention_block
head_dim = layer_0.d_k

print(f"\n=== Prima dell'azzeramento ===")
print(f"w_q[0:{head_dim}, 0:10] (prime righe della prima head):")
print(layer_0.w_q.weight.data[0:head_dim, 0:10])

print(f"\nw_k[0:{head_dim}, 0:10]:")
print(layer_0.w_k.weight.data[0:head_dim, 0:10])

print(f"\nw_v[0:{head_dim}, 0:10]:")
print(layer_0.w_v.weight.data[0:head_dim, 0:10])

# Azzera la prima head del primo layer
print(f"\n=== Azzeramento in corso ===")
zero_attention_head(model, layer_idx=0, head_idx=0, h=8)

# Controlla dopo l'azzeramento
print(f"\n=== Dopo l'azzeramento ===")
print(f"w_q[0:{head_dim}, 0:10]:")
print(layer_0.w_q.weight.data[0:head_dim, 0:10])

print(f"\nw_k[0:{head_dim}, 0:10]:")
print(layer_0.w_k.weight.data[0:head_dim, 0:10])

print(f"\nw_v[0:{head_dim}, 0:10]:")
print(layer_0.w_v.weight.data[0:head_dim, 0:10])

# Verifica che siano tutti zero
w_q_zeros = (layer_0.w_q.weight.data[0:head_dim, :] == 0).all().item()
w_k_zeros = (layer_0.w_k.weight.data[0:head_dim, :] == 0).all().item()
w_v_zeros = (layer_0.w_v.weight.data[0:head_dim, :] == 0).all().item()

print(f"\n✓ Verifica completata:")
print(f"  - w_q head 0 azzerata: {w_q_zeros}")
print(f"  - w_k head 0 azzerata: {w_k_zeros}")
print(f"  - w_v head 0 azzerata: {w_v_zeros}")
