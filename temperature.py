import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import math


def splice_heads(A, h=8):
    A = A.transpose(0, 1).contiguous()
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
    Per ogni matrice di pesi (w_q, w_k, w_v), salva la norma del gradiente
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
                'value': []
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

    # Usa la configurazione reale del blocco attenzione, senza hardcode su d_model.
    num_heads = cross_attn.h
    head_dim = cross_attn.d_k
    if head_idx < 0 or head_idx >= num_heads:
        raise ValueError(f"head_idx {head_idx} fuori range [0, {num_heads - 1}]")

    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim

    # In nn.Linear: weight ha shape (out_features, in_features).
    # Ogni head corrisponde a un blocco di out_features, quindi si azzerano le RIGHE.
    with torch.no_grad():
        for matrix in [cross_attn.w_q, cross_attn.w_k, cross_attn.w_v]:
            matrix.weight[start_idx:end_idx, :].zero_()

        # Opzionale ma coerente: elimina anche il contributo della head nell'output projection.
        cross_attn.w_o.weight[:, start_idx:end_idx].zero_()
    
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


def plot_head_temperature_evolution_moving_avg(
    temp_dir="temperature",
    num_layers=4,
    h=8,
    window_size=100,
    output_file=None
):
    """
    Grafica l'evoluzione della temperatura delle attention heads applicando
    una moving average per ridurre il rumore.

    I layer vengono disposti in una griglia a 2 colonne (tipicamente 2x2
    nel caso num_layers=4), con una legenda unica posta sopra tutti i grafici.

    Args:
        temp_dir: directory contenente HeadTemperatureEvolution.csv
        num_layers: numero di layer decoder
        h: numero di attention heads per layer
        window_size: dimensione della finestra della moving average
                     (in numero di punti del CSV)
        output_file: percorso del file in cui salvare il grafico
    """


    temp_path = Path(temp_dir)
    evolution_file = temp_path / "HeadTemperatureEvolution.csv"

    if not evolution_file.exists():
        print(
            f"✗ File {evolution_file} non trovato. "
            "Esegui compute_head_temperature_evolution() prima."
        )
        return

    try:
        df = pd.read_csv(evolution_file)

        if "iteration" not in df.columns:
            print("✗ La colonna 'iteration' non è presente nel CSV.")
            return

        # Griglia con 2 colonne: per 4 layer diventa naturalmente 2x2
        ncols = 2
        nrows = math.ceil(num_layers / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(17, 4.6 * nrows + 1.6),
            sharex=True
        )

        # axes deve essere sempre un array 1D facile da indicizzare
        axes = np.array(axes).reshape(-1)

        # Colori coerenti: stessa head = stesso colore in tutti i layer
      
        palette = [
    "#332288",  # blu-violaceo
    "#88CCEE",  # azzurro chiaro
    "#44AA99",  # verde acqua
    "#117733",  # verde scuro
    "#999933",  # oliva
    "#DDCC77",  # ocra
    "#CC6677",  # rosso spento
    "#882255",  # bordeaux 
    ]
        
        if h > len(palette):
            raise ValueError(
        f"La palette contiene {len(palette)} colori, "
        f"ma sono state richieste {h} heads.")

        colors = palette[:h]

        for layer_idx in range(num_layers):
            ax = axes[layer_idx]

            for head_idx in range(h):
                col_name = f"layer_{layer_idx}_head_{head_idx}"

                if col_name not in df.columns:
                    continue

                smoothed = df[col_name].rolling(
                    window=window_size,
                    center=True
                ).mean()

                ax.plot(
                    df["iteration"],
                    smoothed,
                    label=f"Head {head_idx + 1}",
                    color=colors[head_idx],
                    linewidth=2,
                    alpha=0.88
                )

            # Titolo del pannello
            ax.set_title(
                f"Layer {layer_idx + 1}",
                fontsize=16,
                pad=10
            )

            # Griglia leggera
            ax.grid(
    True,
    color="#D9D9D9",
    linewidth=0.7,
    alpha=0.85
)

            # Tick più leggibili
            ax.tick_params(axis="both", labelsize=16)

            # Piccolo margine sui bordi orizzontali
            ax.margins(x=0.02)

            # Label asse y solo per la colonna sinistra
            if layer_idx % ncols == 0:
                ax.set_ylabel("Temperature", fontsize=17)

            # Label asse x solo per l'ultima riga
            if layer_idx >= (nrows - 1) * ncols:
                ax.set_xlabel("Step", fontsize=17)

        # Disattiva eventuali subplot inutilizzati
        for idx in range(num_layers, len(axes)):
            axes[idx].axis("off")

        # Legenda unica ricavata dal primo subplot attivo
        handles, labels = axes[0].get_legend_handles_labels()




        handles, labels = axes[0].get_legend_handles_labels()
        plot_area_center = 0.43
        fig.suptitle(
    "Attention head temperatures",
    fontsize=24,
    x=plot_area_center,
    y=0.93
)

        if handles:
            legend = fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(0.805, 0.49),
                ncol=1,
                fontsize=12,
                title_fontsize=11,
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                borderaxespad=0.0,
                handlelength=2.7,
                handletextpad=0.7,
                labelspacing=0.65
                )


        fig.subplots_adjust(
    left=0.08,
    right=0.78,
    bottom=0.08,
    top=0.86,
    wspace=0.22,
    hspace=0.34
)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#BDBDBD")
        legend.get_frame().set_linewidth(0.8)


        # Allinea le y-label tra i subplot
        visible_axes = [ax for ax in axes[:num_layers]]
        fig.align_ylabels(visible_axes)

        # Salvataggio
        if output_file is None:
            output_file = temp_path / f"HeadTemperatureEvolution_MovingAvg_{window_size}.pdf"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.08
        )

        print(f"✓ Grafico Moving Average salvato in {output_file}")

        plt.show()
        plt.close(fig)

    except Exception as e:
        print(f"✗ Errore nel creare il grafico: {e}")
if __name__ == "__main__":

    # equilibrium_temperature()
    # head_temperature()

    # equilibrium_temperature(temp_dir="temperature", h=8)
    # head_temperature(temp_dir="temperature", h=8)
    # print(rank_heads_by_temperature(temp_dir="temperature", num_layers=4, h=8))

    # compute_head_temperature_evolution(temp_dir="temperature", num_layers=4, h=8)
    plot_head_temperature_evolution_moving_avg(temp_dir="temperature", num_layers=4, h=8, window_size=150)
