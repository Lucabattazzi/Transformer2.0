import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def compute_head_temperature_evolution(temp_dir="temperature", num_layers=4, h=8):
    """
    Calcola l'evoluzione della temperatura di ogni attention head nel tempo.
    Per ogni iterazione e ogni head, calcola la media delle temperature di Q, K, V.

    Output: un singolo file HeadTemperatureEvolution.csv con formato:
    - Colonna 1: iteration (numero dell'iterazione)
    - Colonne 2-33: layer_0_head_0, layer_0_head_1, ..., layer_0_head_7,
                    layer_1_head_0, layer_1_head_1, ..., layer_1_head_7,
                    layer_2_head_0, layer_2_head_1, ..., layer_2_head_7,
                    layer_3_head_0, layer_3_head_1, ..., layer_3_head_7

    Totale: 1 + (num_layers × h) = 1 + 32 = 33 colonne

    Args:
        temp_dir: directory dove sono salvati i file grezzi
        num_layers: numero di layer decoder (default 4)
        h: numero di heads per layer (default 8)
    """
    temp_path = Path(temp_dir)

    # Inizializza il dizionario per raccogliere i dati
    result_data = {}
    iterations = None

    # Itera su tutti i layer
    for layer_idx in range(num_layers):
        # Percorsi dei file per Query, Key, Value
        query_path = temp_path / f'crossAttentionQuery_{layer_idx}.csv'
        key_path = temp_path / f'crossAttentionKey_{layer_idx}.csv'
        value_path = temp_path / f'crossAttentionValue_{layer_idx}.csv'

        try:
            # Leggi i file grezzi
            df_query = pd.read_csv(query_path)
            df_key = pd.read_csv(key_path)
            df_value = pd.read_csv(value_path)

            # Estrai le iterazioni dal primo layer
            if iterations is None:
                iterations = df_query['iteration'].values
                result_data['iteration'] = iterations

            # Seleziona le colonne dei heads (escludendo la colonna 'iteration')
            head_columns = [col for col in df_query.columns if col.startswith('head_')][:h]

            # Calcola la media per ogni head ad ogni iterazione
            for head_idx, head_col in enumerate(head_columns):
                temp_q = df_query[head_col].values
                temp_k = df_key[head_col].values
                temp_v = df_value[head_col].values
                # Media delle temperature di Q, K, V per questo head
                head_avg = (temp_q + temp_k + temp_v) / 3
                col_name = f'layer_{layer_idx}_head_{head_idx}'
                result_data[col_name] = head_avg

        except FileNotFoundError as e:
            print(f"✗ Errore: file mancante per layer {layer_idx}: {e}")
            return
        except Exception as e:
            print(f"✗ Errore nel processare layer {layer_idx}: {e}")
            return

    # Crea il DataFrame e salvalo
    try:
        df_result = pd.DataFrame(result_data)
        output_file = temp_path / 'HeadTemperatureEvolution.csv'
        df_result.to_csv(output_file, index=False)
        print(f"✓ Creato {output_file.name} con {len(df_result)} iterazioni e {len(df_result.columns)} colonne")
    except Exception as e:
        print(f"✗ Errore nel salvare il file: {e}")

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
    
    # print(f"Head {head_idx} del layer {layer_idx} azzerata (colonne {start_idx}:{end_idx})")

def rank_heads_by_temperature(temp_dir="temperature", num_layers=4, h=8):
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

def plot_head_temperature_evolution(temp_dir="temperature", num_layers=4, h=8, output_file=None):
    """
    Grafica l'evoluzione della temperatura di ogni attention head nel tempo.
    Crea 4 sottografici (uno per layer), ciascuno mostrando l'andamento delle 8 head.

    Args:
        temp_dir: directory dove è salvato il file HeadTemperatureEvolution.csv
        num_layers: numero di layer decoder (default 4)
        h: numero di heads per layer (default 8)
        output_file: percorso dove salvare il grafico (default: temperature/HeadTemperatureEvolution.pdf)
    """
    temp_path = Path(temp_dir)
    evolution_file = temp_path / 'HeadTemperatureEvolution.csv'

    if not evolution_file.exists():
        print(f"✗ File {evolution_file} non trovato. Esegui compute_head_temperature_evolution() prima.")
        return

    try:
        # Leggi il file
        df = pd.read_csv(evolution_file)

        # Crea figura con 4 sottografici (uno per layer)
        fig, axes = plt.subplots(num_layers, 1, figsize=(14, 12))
        if num_layers == 1:
            axes = [axes]

        # Colori per le head
        colors = plt.cm.tab10(np.linspace(0, 1, h))

        # Per ogni layer
        for layer_idx in range(num_layers):
            ax = axes[layer_idx]

            # Per ogni head
            for head_idx in range(h):
                col_name = f'layer_{layer_idx}_head_{head_idx}'
                if col_name in df.columns:
                    ax.plot(df['iteration'], df[col_name],
                           label=f'Head {head_idx}',
                           color=colors[head_idx],
                           linewidth=1.5,
                           alpha=0.8)

            ax.set_xlabel('Iterazione', fontsize=11)
            ax.set_ylabel('Temperatura (media Q, K, V)', fontsize=11)
            ax.set_title(f'Layer {layer_idx} - Evoluzione Temperatura Heads', fontsize=12, fontweight='bold')
            ax.legend(loc='best', ncol=4, fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Salva il grafico
        if output_file is None:
            output_file = temp_path / 'HeadTemperatureEvolution.pdf'
        else:
            output_file = Path(output_file)

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Grafico salvato in {output_file}")
        plt.close()

    except Exception as e:
        print(f"✗ Errore nel creare il grafico: {e}")

def plot_head_temperature_evolution_moving_avg(temp_dir="temperature", num_layers=4, h=8, window_size=50, output_file=None):
    """
    Grafica l'evoluzione della temperatura con MOVING AVERAGE per ridurre il rumore.
    Crea 4 sottografici (uno per layer).

    Args:
        temp_dir: directory dove è salvato il file HeadTemperatureEvolution.csv
        num_layers: numero di layer decoder (default 4)
        h: numero di heads per layer (default 8)
        window_size: dimensione della finestra mobile (default 50 iterazioni)
        output_file: percorso dove salvare il grafico
    """
    temp_path = Path(temp_dir)
    evolution_file = temp_path / 'HeadTemperatureEvolution.csv'

    if not evolution_file.exists():
        print(f"✗ File {evolution_file} non trovato. Esegui compute_head_temperature_evolution() prima.")
        return

    try:
        # Leggi il file
        df = pd.read_csv(evolution_file)

        # Crea figura con 4 sottografici
        fig, axes = plt.subplots(num_layers, 1, figsize=(14, 12))
        if num_layers == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, h))

        # Per ogni layer
        for layer_idx in range(num_layers):
            ax = axes[layer_idx]

            # Per ogni head
            for head_idx in range(h):
                col_name = f'layer_{layer_idx}_head_{head_idx}'
                if col_name in df.columns:
                    # Calcola moving average
                    smoothed = df[col_name].rolling(window=window_size, center=True).mean()
                    ax.plot(df['iteration'], smoothed,
                           label=f'Head {head_idx}',
                           color=colors[head_idx],
                           linewidth=2,
                           alpha=0.85)

            ax.set_xlabel('Iterazione', fontsize=11)
            ax.set_ylabel('Temperatura (media Q, K, V)', fontsize=11)
            ax.set_title(f'Layer {layer_idx} - Moving Average (window={window_size})', fontsize=12, fontweight='bold')
            ax.legend(loc='best', ncol=4, fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Salva il grafico
        if output_file is None:
            output_file = temp_path / f"HeadTemperatureEvolution_MovingAvg_{window_size}.pdf"
        else:
            output_file = Path(output_file)

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Grafico Moving Average salvato in {output_file}")
        plt.close()

    except Exception as e:
        print(f"✗ Errore nel creare il grafico: {e}")

def plot_head_temperature_evolution_binned(temp_dir="temperature", num_layers=4, h=8, num_bins=20, output_file=None):
    """
    Grafica l'evoluzione della temperatura con BINNING (divisione in intervalli).
    Crea 4 sottografici (uno per layer).
    Divide il training in num_bins intervalli equidistanziati e calcola la media su ogni intervallo.

    Args:
        temp_dir: directory dove è salvato il file HeadTemperatureEvolution.csv
        num_layers: numero di layer decoder (default 4)
        h: numero di heads per layer (default 8)
        num_bins: numero di intervalli in cui dividere il training (default 20)
        output_file: percorso dove salvare il grafico
    """
    temp_path = Path(temp_dir)
    evolution_file = temp_path / 'HeadTemperatureEvolution.csv'

    if not evolution_file.exists():
        print(f"✗ File {evolution_file} non trovato. Esegui compute_head_temperature_evolution() prima.")
        return

    try:
        # Leggi il file
        df = pd.read_csv(evolution_file)

        # Dividi in bin e calcola la media per ogni bin
        df['bin'] = pd.cut(df['iteration'], bins=num_bins, labels=False)
        bin_means = df.groupby('bin').agg({'iteration': 'mean'})  # iterazione media per bin

        # Crea figura con 4 sottografici
        fig, axes = plt.subplots(num_layers, 1, figsize=(14, 12))
        if num_layers == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, h))

        # Per ogni layer
        for layer_idx in range(num_layers):
            ax = axes[layer_idx]

            # Per ogni head
            for head_idx in range(h):
                col_name = f'layer_{layer_idx}_head_{head_idx}'
                if col_name in df.columns:
                    # Calcola media per ogni bin
                    bin_temps = df.groupby('bin')[col_name].mean().values
                    bin_iters = bin_means['iteration'].values

                    ax.plot(bin_iters, bin_temps,
                           label=f'Head {head_idx}',
                           color=colors[head_idx],
                           linewidth=2.5,
                           marker='o',
                           markersize=5,
                           alpha=0.85)

            ax.set_xlabel('Iterazione', fontsize=11)
            ax.set_ylabel('Temperatura (media Q, K, V)', fontsize=11)
            ax.set_title(f'Layer {layer_idx} - Binned Average ({num_bins} intervalli)', fontsize=12, fontweight='bold')
            ax.legend(loc='best', ncol=4, fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Salva il grafico
        if output_file is None:
            output_file = temp_path / 'HeadTemperatureEvolution_Binned.pdf'
        else:
            output_file = Path(output_file)

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Grafico Binned salvato in {output_file}")
        plt.close()

    except Exception as e:
        print(f"✗ Errore nel creare il grafico: {e}")

if __name__ == "__main__":

    # Computa l'evoluzione delle temperature
    compute_head_temperature_evolution(temp_dir="temperature", num_layers=4, h=8)

    # Grafica l'evoluzione con diverse tecniche
    # plot_head_temperature_evolution(temp_dir="temperature", num_layers=4, h=8)
    plot_head_temperature_evolution_moving_avg(temp_dir="temperature", num_layers=4, h=8, window_size=250)
    # plot_head_temperature_evolution_binned(temp_dir="temperature", num_layers=4, h=8, num_bins=20)