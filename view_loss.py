import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

path = "opus_books_weights"
train = 'train'
val = "val"

def view_loss(var, csv_path):
    """Plot a single loss series from CSV file."""
    df = pd.read_csv(f"{csv_path}/{var}_loss_history.csv")

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

def compare_losses(var1, var2, csv_path):
    """Plot two loss series on the same graph."""
    df1 = pd.read_csv(f"{csv_path}/{var1}_loss_history.csv")
    df2 = pd.read_csv(f"{csv_path}/{var2}_loss_history.csv")

    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(df1['global_step'], df1['loss'], marker='.', linestyle='', alpha=0.7, label=f"{var1.capitalize()} Loss")
    plt.plot(df2['global_step'], df2['loss'], marker='.', linestyle='', alpha=0.7, label=f"{var2.capitalize()} Loss")
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title(f"Loss Comparison: {var1.capitalize()} vs {var2.capitalize()}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(csv_path).parent / "loss_comparison.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Salvato: {output_path}")

    plt.show()

if __name__ == "__main__":
    # Plot singolo
    # view_loss(train, path)
    
    # Plot comparativo (due serie)
    compare_losses(train, val, path)

