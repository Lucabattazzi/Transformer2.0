import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURAZIONE - Personalizza qui i parametri di visualizzazione
# ============================================================================

# Dimensioni e colori dei marker
MARKER_SIZE = 0.5           # Dimensione dei marker
MARKER_COLOR = '#1f77b4'  # Colore dei marker (hex o nome: 'blue', 'red', 'green', ecc.)
MARKER_STYLE = 'o'        # Stile marker: 'o', 's', '^', 'D', '*', 'x', ecc.

# Stile della linea
LINE_COLOR = '#1f77b4'    # Colore della linea
LINE_WIDTH = 2            # Spessore della linea
LINE_ALPHA = 0.7          # Trasparenza della linea (0-1)

# Aspetto del grafico
FIGURE_SIZE = (12, 6)     # Dimensione della figura (larghezza, altezza)
TITLE = 'Training Loss History'
X_LABEL = 'Global Step'
Y_LABEL = 'Loss'
GRID = True               # Mostrare griglia
DPI = 100                 # Risoluzione del grafico

# ============================================================================
# SCRIPT PRINCIPALE
# ============================================================================

def main():
    # Carica i dati
    df = pd.read_csv('loss_partial.txt')
    
    # Crea la figura
    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    
    plt.scatter(df['global_step'], df['loss'],
                s=MARKER_SIZE**2,           # scatter usa s (area), quindi quadriamo la size
                color=MARKER_COLOR,
                marker=MARKER_STYLE,
                linewidth=0.5,
                zorder=5)
    
    # Configurazione del grafico
    plt.title(TITLE, fontsize=14, fontweight='bold')
    plt.xlabel(X_LABEL, fontsize=12)
    plt.ylabel(Y_LABEL, fontsize=12)
    
    if GRID:
        plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Mostra il grafico
    plt.show()

if __name__ == '__main__':
    main()
