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

# Set up the plotting style
sns.set_theme(style="whitegrid")

# Plot 1: Latency vs Input Length (Log Scale)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='input_length', y='latency', hue='model', style='model', s=100)
plt.yscale('log')
plt.title("Czas generacji odpowiedzi w zależności od długości tekstu (skala log)")
plt.xlabel("Długość tekstu (znaki)")
plt.ylabel("Czas generacji odpowiedzi (sekundy, skala log)")
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "latency_vs_input_length.png"), dpi=300)
plt.close()

# Plot 2: Latency vs Token Usage (Log Scale)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='token_usage', y='latency', hue='model', style='model', s=100)
plt.yscale('log')
plt.title("Czas generacji odpowiedzi w zależności od liczby tokenów (skala log)")
plt.xlabel("Liczba tokenów")
plt.ylabel("Czas generacji odpowiedzi (sekundy, skala log)")
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "latency_vs_token_usage.png"), dpi=300)
plt.close()

# Plot 3: Latency vs Input Length (Linear Scale)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='input_length', y='latency', hue='model', style='model', s=100)
plt.title("Czas generacji odpowiedzi w zależności od długości tekstu (skala liniowa)")
plt.xlabel("Długość tekstu (znaki)")
plt.ylabel("Czas generacji odpowiedzi (sekundy)")
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "latency_vs_input_length_linear.png"), dpi=300)
plt.close()

# Plot 4: Latency vs Token Usage (Linear Scale)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='token_usage', y='latency', hue='model', style='model', s=100)
plt.title("Czas generacji odpowiedzi w zależności od liczby tokenów (skala liniowa)")
plt.xlabel("Liczba tokenów")
plt.ylabel("Czas generacji odpowiedzi (sekundy)")
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "latency_vs_token_usage_linear.png"), dpi=300)
plt.close()

# Calculate average time usage per token and per input length (character)
df['latency_per_token'] = df['latency'] / df['token_usage']
df['latency_per_char'] = df['latency'] / df['input_length']

# Group by model to get the mean
avg_stats = df.groupby('model')[['latency_per_token', 'latency_per_char']].mean().reset_index()

# Save to CSV table
csv_path = os.path.join(out_dir, "average_latency_stats.csv")
avg_stats.to_csv(csv_path, index=False)

print(f"Plots generated successfully and saved in: {out_dir}")
print(f"Average latency statistics table saved to: {csv_path}")
