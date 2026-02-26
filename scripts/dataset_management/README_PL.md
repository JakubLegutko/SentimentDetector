# Instrukcja Zarządzania Zbiorem Danych

Ten folder zawiera skrypty do zarządzania, konsolidacji i regeneracji zbiorów danych.
Wszystkie pliki wynikowe są zapisywane w folderze `datasets/`.

## 1. Konsolidacja (Tworzenie zbioru średnich ocen)

Skrypt `consolidate_datasets.py` służy do łączenia ocen z wielu modeli.

### Podstawowe użycie:
```bash
python scripts/dataset_management/consolidate_datasets.py datasets/dataset_labeled_model1.json datasets/dataset_labeled_model2.json
```
Tworzy `datasets/average_review.json`.

### Wersja "Trim" (Oszczędna)
Aby usunąć treść artykułów i zastąpić ją linkiem (URL) oraz datą (przydatne do zmniejszenia rozmiaru pliku):
```bash
python scripts/dataset_management/consolidate_datasets.py datasets/dataset_labeled_*.json -trim
```
Tworzy `datasets/average_review_trim.json`.

### Wersja "No Score" (Bez średniej oceny)
Aby stworzyć zbiór zawierający oceny poszczególnych modeli, ale BEZ wyliczonej średniej (np. w celu późniejszej oceny przez LLM-Sędziego):
```bash
python scripts/dataset_management/consolidate_datasets.py datasets/dataset_labeled_*.json -trim -no_score
```
Tworzy `datasets/average_review_trim_no_score.json`.

## 2. Regeneracja tekstu

Skrypt `regenerator.py` służy do przywracania pełnej treści artykułów w plikach typu "Trim" poprzez pobranie ich z oryginalnych adresów URL.

### Użycie:
```bash
python scripts/dataset_management/regenerator.py datasets/average_review_trim.json
```
Skrypt pobierze treść dla każdego linku i zapisze wynik w `datasets/average_review_recreated.json`.

## 3. LLM jako Sędzia (LLM-as-a-judge)

Skrypt `judge.py` wykorzystuje lokalny model językowy do oceny, która z odpowiedzi modeli jest najlepsza.

### Użycie:
Wymaga uruchomionego serwera `LLM_server.py`.

```bash
python scripts/dataset_management/judge.py datasets/average_review_trim_no_score.json -model model_name
```
(Zalecane jest użycie pliku po regeneracji tekstu, jeśli sędzia ma mieć dostęp do treści artykułu, np. `datasets/average_review_recreated.json`).

Argument `-model` (domyślnie "bielik") określa nazwę modelu używanego jako sędzia.
Wynik zostanie zapisany jako `datasets/average_review_trim_no_score_judged_{model_name}.json`.
