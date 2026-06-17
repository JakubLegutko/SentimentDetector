import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

# Identify the judge files
judge_files = glob.glob(os.path.join(datasets_dir, 'average_review_no_score_judged_*_trim.json'))

results = []
all_deviations = []

for file_path in judge_files:
    filename = os.path.basename(file_path)
    # Extract judge name, e.g., average_review_no_score_judged_bielik_trim.json -> bielik
    judge_name = filename.split('_judged_')[1].split('_trim')[0]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    selections = {}
    total_valid = 0
    deviations = []
    perfect_agreements = 0
    
    for item in data:
        selected = item.get('judge_selected_model')
        if selected:
            selections[selected] = selections.get(selected, 0) + 1
            total_valid += 1
            
        # Calculate standard deviation of the 4 models for this item
        scores = []
        if 'model_responses' in item:
            for model_name, response in item['model_responses'].items():
                score = response.get('predicted_score')
                if score is not None:
                    try:
                        scores.append(float(score))
                    except ValueError:
                        pass
        
        if len(scores) > 1:
            dev = np.std(scores)
            deviations.append(dev)
            
        if len(scores) == 4 and len(set(scores)) == 1:
            perfect_agreements += 1
            
    avg_dev = np.mean(deviations) if deviations else 0
    all_deviations.extend(deviations) # aggregate all for global stats
    
    # Calculate self-selection bias if applicable
    self_selected_pct = 0
    for k, v in selections.items():
        if judge_name.lower() in k.lower() or (judge_name == 'llama3' and 'llama' in k.lower()):
            self_selected_pct = (v / total_valid * 100) if total_valid else 0
            break

    results.append({
        'Judge': judge_name.capitalize(),
        'Total_Evaluated': len(data),
        'Total_Valid_Selections': total_valid,
        'Perfect_Agreement_Pct': (perfect_agreements / len(data) * 100) if data else 0,
        'Selections': selections,
        'Avg_Score_Deviation': avg_dev,
        'Self_Selection_Bias_Pct': self_selected_pct
    })

# Save text report to file
report_path = os.path.join(out_dir, "llm_judge_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Raport Analizy LLM-as-a-Judge\n\n")
    f.write("Poniżej znajdują się statystyki dla każdego modelu pełniącego rolę sędziego.\n\n")
    
    for r in results:
        f.write(f"## Sędzia: {r['Judge']}\n")
        f.write(f"- **Przeanalizowane teksty:** {r['Total_Evaluated']}\n")
        f.write(f"- **Odsetek pełnej zgodności (wszystkie 4 modele dały ten sam score):** {r['Perfect_Agreement_Pct']:.2f}%\n")
        f.write(f"- **Średnie odchylenie standardowe ocen modeli (disagreement):** {r['Avg_Score_Deviation']:.4f}\n")
        f.write(f"- **Bias faworyzowania samego siebie (Self-enhancement bias):** {r['Self_Selection_Bias_Pct']:.2f}%\n")
        f.write("- **Wybrane modele (kogo sędzia uznał za najlepszego)**:\n")
        for model, count in sorted(r['Selections'].items(), key=lambda x: x[1], reverse=True):
            pct = count / r['Total_Valid_Selections'] * 100
            f.write(f"  - `{model}`: {count} razy ({pct:.1f}%)\n")
        f.write("\n")

# Prepare data for plotting selections
plot_data = []
for r in results:
    for model, count in r['Selections'].items():
        plot_data.append({'Sędzia': r['Judge'], 'Wybrany_Model': model, 'Liczba': count})

# Plotting
sns.set_theme(style="whitegrid")
df_plot = pd.DataFrame(plot_data)
plt.figure(figsize=(10, 6))
sns.barplot(data=df_plot, x='Sędzia', y='Liczba', hue='Wybrany_Model', palette='Set2')
plt.title("Rozkład wyborów najlepszego modelu przez poszczególnych sędziów (LLM-as-a-judge)", pad=20, fontsize=14)
plt.xlabel("Model w roli Sędziego", fontsize=12)
plt.ylabel("Ilość przyznanych wygranych", fontsize=12)
plt.legend(title="Model wskazany jako zwycięzca", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plot_path = os.path.join(out_dir, "llm_judge_selections.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Raport z analizy LLM-as-a-Judge został zapisany w {report_path}")
print(f"Wykres z zestawieniem wyborów został zapisany w {plot_path}")
