# Text Classification with MLP – IMDB & 20 Newsgroups

Projekt porównuje różne architektury sieci MLP do klasyfikacji tekstu na dwóch datasetach. Bada wpływ reprezentacji tekstu (TF-IDF vs. Embedding), augmentacji danych generowanej przez LLM oraz hiperparametrów (architektura sieci, funkcja aktywacji, learning rate) na dokładność klasyfikacji.

---

## Pliki

| Plik | Opis |
|---|---|
| `2705final.py` | Główny skrypt treningowy – definicje modeli, grid search, trening, ewaluacja, wykresy |
| `augmented_imdb.json` | 100 syntetycznych recenzji filmowych (50 pozytywnych, 50 negatywnych) wygenerowanych przez LLM jako augmentacja danych IMDB |
| `augmented_newsgroups.json` | 100 syntetycznych postów newsgroup w 8 kategoriach wygenerowanych przez LLM jako augmentacja danych 20 Newsgroups |

---

## Datasety

- **IMDB** – binarna klasyfikacja sentymentu recenzji filmowych (`label=1` pozytywna, `label=0` negatywna); pobierany przez bibliotekę `datasets` (HuggingFace)
- **20 Newsgroups** – klasyfikacja postów do 20 kategorii tematycznych; pobierany przez `sklearn.datasets.fetch_20newsgroups`

---

## Modele

### `TextMLP`
Wielowarstwowa sieć MLP przyjmująca na wejściu wektor TF-IDF (10 000 cech). Warstwy ukryte konfigurowalne, wyjście przez `CrossEntropyLoss`.

### `TextMLPWithEmbedding`
MLP z wbudowaną warstwą `nn.Embedding`. Tokeny są uśredniane po wymiarze sekwencji przed podaniem do sieci. Wymaga własnego tokenizera (`SimpleTokenizer`).

### `SimpleTokenizer`
Prosty tokenizer word-level z ograniczonym słownikiem (`max_vocab=10 000`) i paddingiem do stałej długości (`max_len=200`).

---

## Konfiguracja eksperymentu (grid search)

| Parametr | Wartości |
|---|---|
| Architektury ukryte | `[64]`, `[256]`, `[128, 64, 32]`, `[512, 256, 128]` |
| Funkcje aktywacji | `relu`, `tanh`, `leakyrelu` |
| Learning rate | `0.01`, `0.001`, `0.0001` |
| Epoki | 5 |
| Batch size | 64 |

Dla każdego datasetu uruchamiane są **4 warianty**: TF-IDF bez augmentacji, TF-IDF z augmentacją, Embedding bez augmentacji, Embedding z augmentacją – łącznie 4 × 4 × 3 × 3 = **144 konfiguracje na dataset**.

---

## Augmentacja danych

Pliki JSON zostały wygenerowane przez model językowy przy użyciu promptów zdefiniowanych w skrypcie (`AUGMENTATION_PROMPT_IMDB`, `AUGMENTATION_PROMPT_NEWSGROUPS`). Prompty są wypisywane na starcie programu i widoczne w logu. Dane augmentowane są doklejane do zbioru treningowego przed uczeniem.

Kategorie w `augmented_newsgroups.json`:
`comp.graphics`, `rec.sport.hockey`, `sci.med`, `talk.politics.guns`, `alt.atheism`, `rec.autos`, `sci.space`, `talk.religion.misc`

---

## Wyniki

Po zakończeniu treningu skrypt zapisuje:

- **`newsgroups_results.png`** – wykres top-10 konfiguracji + heatmapa accuracy (funkcja aktywacji × learning rate) dla 20 Newsgroups
- **`imdb_results.png`** – analogiczny wykres dla IMDB
- **`saved_models/`** – najlepszy model dla każdego datasetu (plik `.pt` z state dict, nazwą konfiguracji i dokładnością)

---

## Instalacja zależności

```bash
pip install torch scikit-learn datasets matplotlib numpy
```

---

## Uruchomienie

```bash
python 2705final.py
```

Skrypt domyślnie szuka plików augmentacyjnych na `~/Desktop/`. Ścieżki można zmienić na początku pliku:

```python
AUG_IMDB_PATH       = os.path.join(os.path.expanduser("~"), "Desktop", "augmented_imdb.json")
AUG_NEWSGROUPS_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "augmented_newsgroups.json")
```

Jeśli pliki nie zostaną znalezione, trening uruchamia się bez augmentacji (z ostrzeżeniem w logu).
