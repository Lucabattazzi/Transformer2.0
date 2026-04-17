import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "opus_books_weights/loss_history.csv"

df = pd.read_csv(CSV_PATH)

plt.figure(figsize=(12, 6), dpi=100)
plt.plot(df['global_step'], df['loss'], marker='.', linestyle='', alpha=0.7)
plt.xlabel('Global Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = Path(CSV_PATH).parent / "loss_plot.png"
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"Salvato: {output_path}")

plt.show()


