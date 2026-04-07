import csv
from pathlib import Path
import pandas as pd

def splice_heads(A, h=8):
    n = A.shape[0]
    return A.view(n, h, n//h).permute(1, 0, 2).contiguous() # shape: (h, n, n//h)

def initialize_temperature_files(temp_dir="temperature"):
    """
    Inizializza i file CSV per salvare le norme dei gradienti della cross-attention.
    Crea la directory se non esiste e inizializza i file con header.
    
    Args:
        temp_dir: directory dove salvare i file
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)
    
    files = {
        'query': temp_path / 'crossAttentionQuery.csv',
        'key': temp_path / 'crossAttentionKey.csv',
        'value': temp_path / 'crossAttentionValue.csv',
        'output': temp_path / 'crossAttentionOutput.csv'
    }
    
    # Crea i file con header se non esistono
    for file_path in files.values():
        if not file_path.exists():
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'gradient_norm'])
    
    return files


def save_cross_attention_temperatures(model, global_step, frequency=50, temp_dir="temperature", h=8):
    """
    Salva la norma dei gradienti della cross-attention ogni `frequency` iterazioni.
    Per ogni matrice di pesi (w_q, w_k, w_v, w_o), salva la norma del gradiente
    per ogni layer decoder separatamente su CSV.
    
    Args:
        model: il modello Transformer
        global_step: numero dell'iterazione globale
        frequency: ogni quanti step salvare i gradienti
        temp_dir: directory dove salvare i file
    """
    
    # Controlla se è il momento di salvare
    if global_step % frequency != 0:
        return

    try:
            # Itera su tutti i layer del decoder
        for layer_idx, layer in enumerate(model.decoder.layers):
            cross_attn = layer.cross_attention_block
            
            temperatures = {
                'query': [],
                'key': [],
                'value': [],
                'output': []
            }

            # Estrai e accumula le norme dei gradienti
            if cross_attn.w_q.weight.grad is not None:
                Wq_split = splice_heads(cross_attn.w_q.weight.grad, h)  # (h, 512, 64)
                for head_idx in range(h):
                    head_grad_norm = Wq_split[head_idx].norm().item()
                    temperatures['query'].append(head_grad_norm ** 2)
            
            if cross_attn.w_k.weight.grad is not None:
                Wk_split = splice_heads(cross_attn.w_k.weight.grad, h)  # (h, 512, 64)
                for head_idx in range(h):
                    head_grad_norm = Wk_split[head_idx].norm().item()
                    temperatures['key'].append(head_grad_norm ** 2)
            
            if cross_attn.w_v.weight.grad is not None:
                Wv_split = splice_heads(cross_attn.w_v.weight.grad, h)  # (h, 512, 64)
                for head_idx in range(h):
                    head_grad_norm = Wv_split[head_idx].norm().item()
                    temperatures['value'].append(head_grad_norm ** 2)
            
            if cross_attn.w_o.weight.grad is not None:
                Wo_split = splice_heads(cross_attn.w_o.weight.grad, h)  # (h, 512, 64)
                for head_idx in range(h):
                    head_grad_norm = Wo_split[head_idx].norm().item()
                    temperatures['output'].append(head_grad_norm ** 2)

            # Salva tutte le norme (una colonna per ogni layer)
            temp_path = Path(temp_dir)
            temp_path.mkdir(exist_ok=True)
            
            for key, norms in temperatures.items():
                if norms:
                    file_path = temp_path / f'crossAttention{key.capitalize()}_{layer_idx}.csv'
                    
                    # Se il file non esiste, crea l'header
                    if not file_path.exists():
                        with open(file_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            header = ['iteration'] + [f'head_{i}' for i in range(h)]
                            writer.writerow(header)
                    
                    # Scrivi i dati
                    with open(file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        row = [global_step] + norms
                        writer.writerow(row)
                        
    except Exception as e:
        print(f"Errore in save_cross_attention_temperatures: {e}")


def mean_temperature(temp_dir="temperature", h=8):
    """
    Calcola la temperatura media per ogni head e layer a partire dai file CSV.
    
    Args:
        temp_dir: directory dove sono salvati i file
        h: numero di heads
    """
    temp_path = Path(temp_dir)
    results = []

    for file_path in sorted(temp_path.glob('crossAttention*.csv')):
        df = pd.read_csv(file_path)
        head_columns = [col for col in df.columns if col.startswith('head_')][:h]
        mean_temps = df[head_columns].mean()
        results.append(mean_temps.values)

    output_df = pd.DataFrame(results, columns=[f'head_{j}' for j in range(h)])
    output_df.index.name = 'file_index'
    output_df.to_csv(temp_path / 'mean_temperatures.csv')