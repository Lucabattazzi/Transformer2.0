import matplotlib.pyplot as plt
import numpy as np

# Config
d_model = 512
warmup_steps = 4000
num_epochs = 30
batch_size = 14
steps_per_epoch = 300000 // batch_size  # ~21429
total_steps = steps_per_epoch * num_epochs

# Formula dello scheduler
def lr_schedule(step, d_model, warmup_steps):
    step = step + 1  # 1-based indexing
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)

# Calcola LR per ogni step
steps = np.arange(0, total_steps, max(1, total_steps // 2000))  # 2000 punti
lrs = [lr_schedule(s, d_model, warmup_steps) for s in steps]

# Plot
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(steps, lrs, linewidth=2)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.5, label=f'Warmup End ({warmup_steps})')
plt.legend()
plt.tight_layout()
plt.savefig('scheduler_plot.png', dpi=100, bbox_inches='tight')
print(f"Salvato: scheduler_plot.png")
print(f"Total steps: {total_steps:,}")
print(f"Initial LR: {lr_schedule(0, d_model, warmup_steps):.2e}")
print(f"LR at warmup end: {lr_schedule(warmup_steps-1, d_model, warmup_steps):.2e}")
print(f"Final LR: {lr_schedule(total_steps-1, d_model, warmup_steps):.2e}")
plt.show()
