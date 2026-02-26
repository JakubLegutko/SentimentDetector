# Dataset Management Instruction

This folder contains scripts for managing, consolidating, and regenerating datasets.
All output files are saved in the `datasets/` folder.

## 1. Consolidation (Creating a dataset of average ratings)

The `consolidate_datasets.py` script is used to combine ratings from multiple models.

### Basic usage:
```bash
python scripts/dataset_management/consolidate_datasets.py datasets/dataset_labeled_model1.json datasets/dataset_labeled_model2.json
```
Creates `datasets/average_review.json`.

### "Trim" Version (Economical)
To remove the content of the articles and replace it with a link (URL) and a date (useful for reducing file size):
```bash
python scripts/dataset_management/consolidate_datasets.py datasets/dataset_labeled_*.json -trim
```
Creates `datasets/average_review_trim.json`.

### "No Score" Version (Without average rating)
To create a dataset containing ratings from individual models, but WITHOUT a calculated average (e.g., for later evaluation by an LLM-Judge):
```bash
python scripts/dataset_management/consolidate_datasets.py datasets/dataset_labeled_*.json -trim -no_score
```
Creates `datasets/average_review_trim_no_score.json`.

## 2. Text Regeneration

The `regenerator.py` script is used to restore the full content of articles in "Trim" type files by downloading them from their original URLs.

### Usage:
```bash
python scripts/dataset_management/regenerator.py datasets/average_review_trim.json
```
The script will download the content for each link and save the result in `datasets/average_review_recreated.json`.

## 3. LLM as a Judge (LLM-as-a-judge)

The `judge.py` script uses a local language model to evaluate which model response is the best.

### Usage:
Requires the `LLM_server.py` server to be running.

```bash
python scripts/dataset_management/judge.py datasets/average_review_trim_no_score.json -model model_name
```
(It is recommended to use the file after text regeneration if the judge is to have access to the article content, e.g., `datasets/average_review_recreated.json`).

The `-model` argument (default "bielik") specifies the name of the model used as a judge.
The result will be saved as `datasets/average_review_trim_no_score_judged_{model_name}.json`.
