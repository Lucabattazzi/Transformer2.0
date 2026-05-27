"""
Legge i valori di temperatura da `temperature_heads.txt` e produce
un grafico a barre raggruppate per `head` (gruppi) con 4 barre per
gruppo (una per layer). I colori sono costanti per layer.

Il file di input deve essere nella stessa cartella dello script.
Il grafico viene salvato in `temperature_heads_grouped.png` nella stessa cartella.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_temperatures(file_path):
	data = []
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				# Skip separators and headers
				if line.startswith('===') or 'layer' in line.lower() and 'head' in line.lower():
					continue
				# Try to split into three numeric columns
				parts = [p for p in line.split() if p]
				if len(parts) >= 3:
					try:
						layer = int(parts[0])
						head = int(parts[1])
						temp = float(parts[2])
						data.append((layer, head, temp))
					except ValueError:
						# ignore non-numeric lines
						continue
	except FileNotFoundError:
		print(f"File non trovato: {file_path}")
		raise
	return data


def plot_grouped_heads(data, out_path):
	# costruisci DataFrame
	df = pd.DataFrame(data, columns=['layer', 'head', 'temperature'])
	if df.empty:
		print('[temp_histo] Nessun dato da plottare.')
		return

	# pivot: righe=head, colonne=layer
	df_pivot = df.pivot(index='head', columns='layer', values='temperature')

	# ordina heads e layers
	heads = sorted(df_pivot.index.tolist())
	layers = sorted(df_pivot.columns.tolist())

	x = np.arange(len(heads))
	num_layers = len(layers)

	# palette: un colore per layer
	colors = ['#4A90E2', '#FF9F43', '#2ECC71', '#E74C3C']

	width = 0.8 / max(1, num_layers)

	fig, ax = plt.subplots(figsize=(12, 6))

	for i, layer in enumerate(layers):
		offset = (i - (num_layers - 1) / 2) * width
		values = [df_pivot.at[h, layer] for h in heads]
		ax.bar(x + offset, values, width, label=f'Layer {layer}', color=colors[i % len(colors)], edgecolor='black', linewidth=0.6)

	# Etichette: mostra Head come 1-based (Head 1, Head 2, ...)
	ax.set_xticks(x)
	ax.set_xticklabels([f'Head {h+1}' for h in heads])
	ax.set_xlabel('Head', fontsize=12, fontweight='bold')
	ax.set_ylabel('Temperature', fontsize=12, fontweight='bold')
	ax.set_title('Temperature per Head (gruppate per Head, barre=Layer)', fontsize=14, fontweight='bold')
	ax.grid(axis='y', linestyle='--', alpha=0.5)
	ax.legend(title='Layer')

	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	plt.close()
	print(f'Plot salvato in: {out_path}')


if __name__ == '__main__':
	script_dir = os.path.dirname(__file__)
	file_path = os.path.join(script_dir, 'temperature_heads.txt')
	out_path = os.path.join(script_dir, 'temperature_heads_grouped.png')

	print(f'[temp_histo] Leggo: {file_path}')
	data = read_temperatures(file_path)
	print(f'[temp_histo] Record letti: {len(data)}')
	plot_grouped_heads(data, out_path)