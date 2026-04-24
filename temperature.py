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

def equilibrium_temperature(temp_dir="temperature", h=8):
    """
    Calcola la temperatura di equilibrio (media sulle iterazioni) per ogni head e layer.
    La temperatura di equilibrio è ottenuta mediando i valori di temperatura su tutte
    le iterazioni presenti nei file CSV per ogni matrice (Query, Key, Value) e layer.
    
    Args:
        temp_dir: directory dove sono salvati i file
        h: numero di heads
    """
    temp_path = Path(temp_dir)
    
    # Itera su ogni tipo di matrice: Query, Key, Value
    for matrix_type in ['Query', 'Key', 'Value']:
        # Trova tutti i file per questo tipo di matrice
        file_pattern = f'crossAttention{matrix_type}_*.csv'
        files = sorted(temp_path.glob(file_pattern))
        
        for file_path in files:
            # Estrai il numero di layer dal nome del file
            layer_idx = int(file_path.stem.split('_')[-1])
            
            # Leggi il file CSV
            df = pd.read_csv(file_path)
            
            # Seleziona solo le colonne dei heads
            head_columns = [col for col in df.columns if col.startswith('head_')][:h]
            
            # Calcola la media su tutte le iterazioni
            equilibrium_temps = df[head_columns].mean().values
            
            # Crea il file di output
            output_file = temp_path / f'EquilibriumTemperatureCrossAttention{matrix_type}_{layer_idx}.csv'
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Scrivi header
                header = [f'head_{i}' for i in range(h)]
                writer.writerow(header)
                # Scrivi i dati di equilibrio
                writer.writerow(equilibrium_temps)

def head_temperature(temp_dir="temperature", h=8):
    """
    Calcola la temperatura di ogni attention head facendo la media sulle rispettive
    heads di Query, Key e Value. Crea un file per ogni layer.
    
    Args:
        temp_dir: directory dove sono salvati i file
        h: numero di heads
    """
    temp_path = Path(temp_dir)
    
    # Trova tutti i file EquilibriumTemperature per Query e estrai i layer indices
    query_files = sorted(temp_path.glob('EquilibriumTemperatureCrossAttentionQuery_*.csv'))
    
    for query_file in query_files:
        # Estrai il numero di layer dal nome del file
        layer_idx = int(query_file.stem.split('_')[-1])
        
        # Leggi i file di equilibrio per Query, Key, Value
        query_path = temp_path / f'EquilibriumTemperatureCrossAttentionQuery_{layer_idx}.csv'
        key_path = temp_path / f'EquilibriumTemperatureCrossAttentionKey_{layer_idx}.csv'
        value_path = temp_path / f'EquilibriumTemperatureCrossAttentionValue_{layer_idx}.csv'
        
        try:
            df_query = pd.read_csv(query_path)
            df_key = pd.read_csv(key_path)
            df_value = pd.read_csv(value_path)
            
            # Seleziona le colonne dei heads
            head_columns = [col for col in df_query.columns if col.startswith('head_')][:h]
            
            # Per ogni head, calcola la media su Q, K, V
            head_temps = []
            for head_col in head_columns:
                temp_q = df_query[head_col].values[0]
                temp_k = df_key[head_col].values[0]
                temp_v = df_value[head_col].values[0]
                head_temp = (temp_q + temp_k + temp_v) / 3
                head_temps.append(head_temp)
            
            # Crea il file di output per questo layer
            output_file = temp_path / f'HeadTemperatureCrossAttention_{layer_idx}.csv'
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Scrivi header
                header = [f'head_{i}' for i in range(h)]
                writer.writerow(header)
                # Scrivi i dati aggregati
                writer.writerow(head_temps)
                
        except FileNotFoundError as e:
            print(f"Errore: file mancante per layer {layer_idx}: {e}")

def zero_attention_head(model, layer_idx, head_idx, h=8):
    """
    Azzera i pesi di una attention head specifica (Query, Key, Value) in un determinato layer.
    
    Args:
        model: il modello Transformer
        layer_idx: indice del layer decoder (0-5)
        head_idx: indice della attention head da azzerare (0-7)
        h: numero totale di heads
    """
    cross_attn = model.decoder.layers[layer_idx].cross_attention_block
    
    # Calcola la dimensione per head
    head_dim = 512 // h  # 512 / 8 = 64
    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim
    
    # Azzera i pesi di Query, Key e Value per questa head
    for matrix in [cross_attn.w_q, cross_attn.w_k, cross_attn.w_v]:
        matrix.weight.data[:, start_idx:end_idx].zero_()
    
    print(f"Head {head_idx} del layer {layer_idx} azzerata (colonne {start_idx}:{end_idx})")

def rank_heads_by_temperature(temp_dir="temperature", num_layers=6, h=8):
    """
    Legge i file HeadTemperature per ogni layer e restituisce 6 DataFrames
    ordinati per temperatura crescente.
    
    Args:
        temp_dir: directory dove sono salvati i file
        num_layers: numero di layer (default 6)
        h: numero di heads per layer (default 8)
    
    Returns:
        Dizionario {layer_idx: DataFrame} dove ogni DataFrame ha colonne [layer, head, temperature]
        ordinato in modo crescente per temperatura
    """
    temp_path = Path(temp_dir)
    rank_dict = {}
    
    for layer_idx in range(num_layers):
        head_temp_file = temp_path / f'HeadTemperatureCrossAttention_{layer_idx}.csv'
        
        try:
            # Leggi il file
            df = pd.read_csv(head_temp_file)
            
            # Converti in forma "lunga" con colonne [layer, head, temperature]
            data = []
            for head_idx in range(h):
                head_col = f'head_{head_idx}'
                if head_col in df.columns:
                    temp_value = df[head_col].values[0]
                    data.append({'layer': layer_idx, 'head': head_idx, 'temperature': temp_value})
            
            # Crea il DataFrame e ordinalo per temperature crescente
            df_layer = pd.DataFrame(data)
            df_layer = df_layer.sort_values('temperature', ascending=True).reset_index(drop=True)
            
            rank_dict[layer_idx] = df_layer

            
        except FileNotFoundError:
            print(f"✗ Errore: file HeadTemperatureCrossAttention_{layer_idx}.csv non trovato")
    
    return rank_dict

if __name__ == "__main__":

    # equilibrium_temperature()
    # head_temperature()

    rank_heads_by_temperature(temp_dir="temperature", num_layers=6, h=8)
    ranking = rank_heads_by_temperature()
    for layer in range(4):
        print(f"\n=== Layer {layer} ===")
        print(ranking[layer].to_string(index=False))
