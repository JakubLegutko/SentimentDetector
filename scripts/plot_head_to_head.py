import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the log file path relative to this script
log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'telemetry.log')
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

# Read telemetry data
data = []
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

df = pd.DataFrame(data)

# Map labels to numeric (1 for Objective, -1 for Subjective)
# Pamiętajmy, że w logach (telemetry.log) wartości to "Objective" oraz "Subjective"
label_map = {'Objective': 1, 'Subjective': -1}
df['label_numeric'] = df['label'].map(label_map)

# Pivot tables using input_length as the unique identifier for the text
# aggfunc='mean' handles cases where the same text was evaluated multiple times by the same model
df_labels = df.pivot_table(index='input_length', columns='model', values='label_numeric', aggfunc='mean')
df_scores = df.pivot_table(index='input_length', columns='model', values='score', aggfunc='mean')

# Sort index so texts are ordered by length
df_labels = df_labels.sort_index()
df_scores = df_scores.sort_index()

sns.set_theme(style="white")

# 1. Heatmap etykiet (Label Agreement)
plt.figure(figsize=(12, 10))
# Create a custom colormap: Red for Subjective (0), Blue for Objective (1)
cmap_labels = sns.color_palette(["#e74c3c", "#3498db"])
ax = sns.heatmap(df_labels, cmap=cmap_labels, cbar=False, linewidths=1, linecolor='white')

# Add custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498db', label='Obiektywny'),
                   Patch(facecolor='#e74c3c', label='Subiektywny'),
                   Patch(facecolor='white', edgecolor='black', label='Brak danych')]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

plt.title("Porównanie decyzji modeli dla poszczególnych tekstów", pad=20, fontsize=14)
plt.ylabel("Długość tekstu (identyfikator)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "head_to_head_labels.png"), dpi=300)
plt.close()


# 2. Korelacja wyników (Score Correlation Matrix)
plt.figure(figsize=(8, 6))
# Calculate correlation matrix
corr_matrix = df_scores.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
plt.title("Macierz korelacji wyników pomiędzy modelami", pad=20, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "head_to_head_scores_corr.png"), dpi=300)
plt.close()


# 3. Znormalizowane wyniki (Normalized Scores Heatmap)
# Ręczna normalizacja wyników do skali od -1 do 1 na podstawie logiki modeli
def normalize_score(row):
    model = row['model']
    score = row['score']
    if 'Vader' in model:
        # Vader działa w skali ok. -100 do 100, dzielimy przez 100
        return score / 100.0
    elif '1DCNN' in model or 'DeBERTa' in model:
        # 1DCNN i DeBERTa zwracają prawdopodobieństwo od 0 do 1
        # Zamieniamy na skalę -1 do 1: (score - 0.5) * 2
        return (score - 0.5) * 2
    else:
        # Gemini i Local LLM zwracają już od ok. -1 do 1
        return score

df['score_norm'] = df.apply(normalize_score, axis=1)

# Pivot znormalizowanych wyników
df_scores_norm = df.pivot_table(index='input_length', columns='model', values='score_norm', aggfunc='mean')
df_scores_norm = df_scores_norm.sort_index()

plt.figure(figsize=(12, 10))
sns.heatmap(df_scores_norm, cmap='viridis', annot=False, linewidths=0.5, vmin=-1, vmax=1)
plt.title("Znormalizowane wyniki modeli", pad=20, fontsize=14)
plt.ylabel("Długość tekstu (identyfikator)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "head_to_head_scores_norm.png"), dpi=300)
plt.close()

print(f"Pomyślnie wygenerowano 3 wykresy porównawcze i zapisano je w: {out_dir}")
