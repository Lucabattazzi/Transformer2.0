import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

path = "opus_books_weights"
train = ''
val = "val_"

def view_loss(csv_path, var):

    df = pd.read_csv(f"{csv_path}/{var}loss_history.csv")

    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(df['global_step'], df['loss'], marker='.', linestyle='', alpha=0.7)
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title(f"{var.capitalize()} Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(csv_path).parent /"loss_plot.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Salvato: {output_path}")

    plt.show()

if __name__ == "__main__":
    view_loss(path, val)

