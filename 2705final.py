import matplotlib
matplotlib.use('Agg')   # backend bez GUI – brak okien, tylko zapis do pliku
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from datasets import load_dataset


import numpy as np
import os
import json

print("START")


# =========================
# ŚCIEŻKI DO PLIKÓW AUGMENTACJI
# Zmień jeśli pliki są w innym miejscu
# =========================
AUG_IMDB_PATH       = os.path.join(os.path.expanduser("~"), "Desktop", "augmented_imdb.json")
AUG_NEWSGROUPS_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "augmented_newsgroups.json")


# =========================
# KONFIGURACJA EKSPERYMENTU
# =========================
EXPERIMENT_CONFIG = {
    "hidden_layer_configs": [
        ([64],            "2 layers"),
        ([256],           "2 layers"),
        ([128, 64, 32],   "4 layers"),
        ([512, 256, 128], "4 layers"),
    ],
    "activations":    ["relu", "tanh", "leakyrelu"],
    "learning_rates": [0.01, 0.001, 0.0001],
    "epochs":         5,
    "save_dir":       "saved_models",
}

os.makedirs(EXPERIMENT_CONFIG["save_dir"], exist_ok=True)


# =========================
# PROMPTY AUGMENTACYJNE
# =========================
AUGMENTATION_PROMPT_IMDB = """
Jesteś generatorem danych treningowych do klasyfikacji recenzji filmowych (IMDB).
Wygeneruj {n} krótkich recenzji filmowych w języku angielskim.
Połowa powinna być pozytywna (label=1), połowa negatywna (label=0).

Format odpowiedzi (tylko JSON, bez żadnego dodatkowego tekstu):
{{
  "reviews": [
    {{"text": "treść recenzji", "label": 0}},
    {{"text": "treść recenzji", "label": 1}}
  ]
}}

Recenzje powinny być naturalne, zróżnicowane, podobne do typowych recenzji IMDB.
"""

AUGMENTATION_PROMPT_NEWSGROUPS = """
Jesteś generatorem danych treningowych do klasyfikacji postów newsgroup (20newsgroups).
Wygeneruj {n} krótkich postów w języku angielskim.
Kategorie do użycia (wybierz losowo):
  comp.graphics, rec.sport.hockey, sci.med, talk.politics.guns,
  alt.atheism, rec.autos, sci.space, talk.religion.misc

Format odpowiedzi (tylko JSON, bez żadnego dodatkowego tekstu):
{{
  "posts": [
    {{"text": "treść posta", "category": "nazwa_kategorii"}}
  ]
}}

Posty powinny być naturalne i zawierać słowa kluczowe typowe dla danej kategorii.
"""


def get_augmentation_prompt(dataset_name, n=100):
    if dataset_name == "imdb":
        return AUGMENTATION_PROMPT_IMDB.format(n=n)
    elif dataset_name == "newsgroups":
        return AUGMENTATION_PROMPT_NEWSGROUPS.format(n=n)
    raise ValueError(f"Nieznany dataset: {dataset_name}")


# =========================
# WCZYTANIE DANYCH AUGMENTACYJNYCH
# =========================
def load_augmented_data(json_path, dataset_name):
    if not os.path.exists(json_path):
        print(f"  ⚠  Plik nie znaleziony: {json_path} – pomijam augmentację.")
        return [], []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []

    if dataset_name == "imdb":
        for item in data.get("reviews", []):
            texts.append(item["text"])
            labels.append(int(item["label"]))

    elif dataset_name == "newsgroups":
        # kategoria → indeks liczbowy (te same co fetch_20newsgroups)
        cat2idx = {
            "alt.atheism": 0, "comp.graphics": 1, "rec.autos": 11,
            "rec.sport.hockey": 13, "sci.med": 15, "sci.space": 16,
            "talk.politics.guns": 18, "talk.religion.misc": 19,
        }
        for item in data.get("posts", []):
            cat = item.get("category", "")
            if cat in cat2idx:
                texts.append(item["text"])
                labels.append(cat2idx[cat])

    print(f"  ✓ Wczytano {len(texts)} próbek augmentowanych z: {json_path}")
    return texts, labels


# =========================
# MODELE
# =========================
class TextMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes, activation="relu"):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), self._act(activation)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def _act(self, name):
        return {"relu": nn.ReLU(), "tanh": nn.Tanh(),
                "leakyrelu": nn.LeakyReLU(0.01)}[name]

    def forward(self, x):
        return self.net(x)


class TextMLPWithEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_layers, num_classes, activation="relu"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layers, prev = [], embed_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), self._act(activation)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def _act(self, name):
        return {"relu": nn.ReLU(), "tanh": nn.Tanh(),
                "leakyrelu": nn.LeakyReLU(0.01)}[name]

    def forward(self, x):
        return self.net(self.embedding(x).mean(dim=1))


class SimpleTokenizer:
    def __init__(self, max_vocab=10000, max_len=200):
        self.max_vocab = max_vocab
        self.max_len   = max_len
        self.word2idx  = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, texts):
        from collections import Counter
        counter = Counter(w for t in texts for w in t.lower().split())
        for word, _ in counter.most_common(self.max_vocab - 2):
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        return self

    def transform(self, texts):
        out = []
        for t in texts:
            idx = [self.word2idx.get(w.lower(), 1) for w in t.split()][:self.max_len]
            idx += [0] * (self.max_len - len(idx))
            out.append(idx)
        return np.array(out, dtype=np.int64)


# =========================
# TRENING / EWALUACJA / ZAPIS
# =========================
def train_model(model, train_loader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}, loss={total:.4f}")


def evaluate(model, test_loader):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for X, y in test_loader:
            preds.extend(torch.argmax(model(X), dim=1).numpy())
            true.extend(y.numpy())
    return accuracy_score(true, preds)


def save_model(model, config_name, dataset_name, accuracy):
    fname = (f"{EXPERIMENT_CONFIG['save_dir']}/{dataset_name}_{config_name}"
             f"_acc{accuracy:.4f}.pt")
    fname = fname.replace("[","").replace("]","").replace(", ","-").replace(" ","_")
    torch.save({"model_state_dict": model.state_dict(),
                "config_name": config_name,
                "accuracy": accuracy,
                "dataset": dataset_name}, fname)
    print(f"  ✓ Model zapisany: {fname}")


# =========================
# GRID SEARCH – jedna runda
# (use_augmentation=True dołącza dane z JSON)
# =========================
def run_experiments(dataset_name, X_train, X_test, y_train, y_test,
                    texts_train, texts_test, num_classes,
                    use_embedding, use_augmentation,
                    aug_texts, aug_labels):

    configs     = EXPERIMENT_CONFIG["hidden_layer_configs"]
    activations = EXPERIMENT_CONFIG["activations"]
    lrs         = EXPERIMENT_CONFIG["learning_rates"]
    epochs      = EXPERIMENT_CONFIG["epochs"]

    # --- budowanie loaderów ---
    if use_embedding:
        all_train_texts  = list(texts_train)
        all_train_labels = list(y_train.numpy())

        if use_augmentation and aug_texts:
            all_train_texts  += aug_texts
            all_train_labels += aug_labels
            print(f"  Augmentacja: +{len(aug_texts)} próbek → łącznie {len(all_train_texts)}")

        tokenizer = SimpleTokenizer().fit(all_train_texts)
        vocab_size, embed_dim = len(tokenizer.word2idx), 64

        X_tr = torch.tensor(tokenizer.transform(all_train_texts), dtype=torch.long)
        X_te = torch.tensor(tokenizer.transform(list(texts_test)),  dtype=torch.long)
        y_tr = torch.tensor(all_train_labels, dtype=torch.long)
        y_te = y_test
    else:
        X_tr = torch.tensor(X_train, dtype=torch.float32)
        X_te = torch.tensor(X_test,  dtype=torch.float32)

        if use_augmentation and aug_texts:
            # TF-IDF: przelicz augmentowane teksty tym samym wektoryzerem
            # (vectorizer przekazywany przez run_* przez closure)
            aug_X = _vectorizer.transform(aug_texts).toarray()
            X_tr  = torch.cat([X_tr, torch.tensor(aug_X, dtype=torch.float32)])
            y_tr  = torch.cat([y_train,
                               torch.tensor(aug_labels, dtype=torch.long)])
            print(f"  Augmentacja TF-IDF: +{len(aug_texts)} próbek → łącznie {len(X_tr)}")
        else:
            y_tr = y_train
        y_te = y_test

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=64)

    results, best_acc, best_cfg, best_model = [], 0.0, None, None
    total = len(configs) * len(activations) * len(lrs)

    for hidden_layers, layer_label in configs:
        for activation in activations:
            for lr in lrs:
                cfg = f"{hidden_layers}_{activation}_lr{lr}"
                aug_tag = "+aug" if (use_augmentation and aug_texts) else ""
                emb_tag = "emb" if use_embedding else "tfidf"
                print(f"\n[{len(results)+1}/{total}] {dataset_name} | {emb_tag}{aug_tag} | "
                      f"arch={hidden_layers} | act={activation} | lr={lr}")

                if use_embedding:
                    model = TextMLPWithEmbedding(vocab_size, embed_dim,
                                                hidden_layers, num_classes, activation)
                else:
                    model = TextMLP(10000, hidden_layers, num_classes, activation)

                train_model(model, train_loader, epochs=epochs, lr=lr)
                acc = evaluate(model, test_loader)
                print(f"  → Accuracy: {acc:.4f}")

                results.append({
                    "hidden_layers": str(hidden_layers),
                    "layer_label":   layer_label,
                    "activation":    activation,
                    "lr":            lr,
                    "accuracy":      acc,
                    "config_name":   cfg,
                    "augmented":     bool(use_augmentation and aug_texts),
                    "embedding":     use_embedding,
                })

                if acc > best_acc:
                    best_acc, best_cfg, best_model = acc, cfg, model

    if best_model:
        save_model(best_model, best_cfg, dataset_name, best_acc)

    print(f"\n{'='*50}")
    print(f"NAJLEPSZA ({dataset_name}): {best_cfg}  →  {best_acc:.4f}")
    print(f"{'='*50}")
    return results, best_cfg, best_acc


# =========================
# WYKRES
# =========================
def plot_results(all_results, title, filename):
    """
    all_results: lista słowników z kluczem 'augmented' i 'embedding'
    Rysuje top-10 ranking + heatmapę act×lr.
    """
    sorted_r = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)
    top10    = sorted_r[:10]

    labels = [
        f"{r['hidden_layers']}\n{r['activation']} lr={r['lr']}"
        + (" +aug" if r["augmented"] else "")
        + (" emb"  if r["embedding"] else "")
        for r in top10
    ]
    accs = [r["accuracy"] for r in top10]

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Ranking
    bars = axes[0].barh(labels[::-1], accs[::-1],
                        color=["tomato" if r["augmented"] else "steelblue"
                               for r in top10[::-1]])
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title(f"{title} – Top 10  (czerwony = z augmentacją)")
    axes[0].set_xlim(min(accs) - 0.02, max(accs) + 0.02)
    axes[0].bar_label(bars, fmt="%.4f", padding=3)

    # Heatmapa act × lr (średnia po wszystkich wariantach)
    acts = EXPERIMENT_CONFIG["activations"]
    lrs  = EXPERIMENT_CONFIG["learning_rates"]
    matrix = np.zeros((len(acts), len(lrs)))
    for i, act in enumerate(acts):
        for j, lr in enumerate(lrs):
            vals = [r["accuracy"] for r in all_results
                    if r["activation"] == act and r["lr"] == lr]
            matrix[i, j] = np.mean(vals) if vals else 0

    im = axes[1].imshow(matrix, cmap="YlGn", aspect="auto")
    axes[1].set_xticks(range(len(lrs)));  axes[1].set_xticklabels([str(l) for l in lrs])
    axes[1].set_yticks(range(len(acts))); axes[1].set_yticklabels(acts)
    axes[1].set_xlabel("Learning Rate"); axes[1].set_ylabel("Activation")
    axes[1].set_title(f"{title} – Avg accuracy (act × lr)")
    plt.colorbar(im, ax=axes[1])
    for i in range(len(acts)):
        for j in range(len(lrs)):
            axes[1].text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  ✓ Wykres: {filename}")
    plt.close()
    #plt.show()


# =========================
# 20 NEWSGROUPS
# =========================
_vectorizer = None  # globalny, żeby run_experiments mógł z niego skorzystać

def run_20newsgroups():
    global _vectorizer

    print("\n" + "="*50 + "\n=== 20 Newsgroups ===\n" + "="*50)

    data   = fetch_20newsgroups(subset='all')
    texts  = data.data
    labels = data.target

    _vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X = _vectorizer.fit_transform(texts).toarray()

    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        X, labels, texts, test_size=0.2, random_state=42)

    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_te = torch.tensor(y_te, dtype=torch.long)

    aug_texts, aug_labels = load_augmented_data(AUG_NEWSGROUPS_PATH, "newsgroups")

    all_results = []

    # 1) TF-IDF, bez augmentacji
    print("\n--- TF-IDF | bez augmentacji ---")
    r, _, _ = run_experiments("ng_tfidf", X_tr, X_te, y_tr, y_te,
                               t_tr, t_te, 20, False, False, [], [])
    all_results += r

    # 2) TF-IDF, z augmentacją
    print("\n--- TF-IDF | z augmentacją ---")
    r, _, _ = run_experiments("ng_tfidf_aug", X_tr, X_te, y_tr, y_te,
                               t_tr, t_te, 20, False, True, aug_texts, aug_labels)
    all_results += r

    # 3) Embedding, bez augmentacji
    print("\n--- Embedding | bez augmentacji ---")
    r, _, _ = run_experiments("ng_emb", X_tr, X_te, y_tr, y_te,
                               t_tr, t_te, 20, True, False, [], [])
    all_results += r

    # 4) Embedding, z augmentacją
    print("\n--- Embedding | z augmentacją ---")
    r, _, _ = run_experiments("ng_emb_aug", X_tr, X_te, y_tr, y_te,
                               t_tr, t_te, 20, True, True, aug_texts, aug_labels)
    all_results += r

    plot_results(all_results, "20 Newsgroups", "newsgroups_results.png")

    # Podsumowanie
    best = max(all_results, key=lambda x: x["accuracy"])
    print("\n=== NAJLEPSZA KONFIGURACJA (20 Newsgroups) ===")
    print(f"  Arch      : {best['hidden_layers']}")
    print(f"  Activation: {best['activation']}")
    print(f"  LR        : {best['lr']}")
    print(f"  Embedding : {best['embedding']}")
    print(f"  Augmented : {best['augmented']}")
    print(f"  Accuracy  : {best['accuracy']:.4f}")


# =========================
# IMDB
# =========================
def run_imdb():
    global _vectorizer

    print("\n" + "="*50 + "\n=== IMDB ===\n" + "="*50)

    dataset = load_dataset("imdb")
    texts   = dataset['train']['text']
    labels  = dataset['train']['label']

    _vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X = _vectorizer.fit_transform(texts).toarray()

    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        X, labels, texts, test_size=0.2, random_state=42)

    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_te = torch.tensor(y_te, dtype=torch.long)

    aug_texts, aug_labels = load_augmented_data(AUG_IMDB_PATH, "imdb")

    all_results = []

    # 1) TF-IDF, bez augmentacji
    print("\n--- TF-IDF | bez augmentacji ---")
    r, _, _ = run_experiments("imdb_tfidf", X_tr, X_te, y_tr, y_te,
                               t_tr, t_te, 2, False, False, [], [])
    all_results += r

    # 2) TF-IDF, z augmentacją
    print("\n--- TF-IDF | z augmentacją ---")
    r, _, _ = run_experiments("imdb_tfidf_aug", X_tr, X_te, y_tr, y_te,
                               t_tr, t_te, 2, False, True, aug_texts, aug_labels)
    all_results += r

    # 3) Embedding, bez augmentacji
    print("\n--- Embedding | bez augmentacji ---")
    r, _, _ = run_experiments("imdb_emb", X_tr, X_te, y_tr, y_te,
                               t_tr, t_te, 2, True, False, [], [])
    all_results += r

    # 4) Embedding, z augmentacją
    print("\n--- Embedding | z augmentacją ---")
    r, _, _ = run_experiments("imdb_emb_aug", X_tr, X_te, y_tr, y_te,
                               t_tr, t_te, 2, True, True, aug_texts, aug_labels)
    all_results += r

    plot_results(all_results, "IMDB", "imdb_results.png")

    # Podsumowanie
    best = max(all_results, key=lambda x: x["accuracy"])
    print("\n=== NAJLEPSZA KONFIGURACJA (IMDB) ===")
    print(f"  Arch      : {best['hidden_layers']}")
    print(f"  Activation: {best['activation']}")
    print(f"  LR        : {best['lr']}")
    print(f"  Embedding : {best['embedding']}")
    print(f"  Augmented : {best['augmented']}")
    print(f"  Accuracy  : {best['accuracy']:.4f}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("=" * 60)
    print("PROMPT – IMDB (użyty do wygenerowania augmented_imdb.json)")
    print("=" * 60)
    print(get_augmentation_prompt("imdb", n=100))

    print("\n" + "=" * 60)
    print("PROMPT – 20 Newsgroups (użyty do wygenerowania augmented_newsgroups.json)")
    print("=" * 60)
    print(get_augmentation_prompt("newsgroups", n=100))

    print("\n" + "=" * 60)
    print("Startuje trening...")
    print("=" * 60 + "\n")

    run_20newsgroups()
    run_imdb()

    print("\n✓ Trening zakończony. Otwieranie wykresów...")
    for fname in ["newsgroups_results.png", "imdb_results.png"]:
        img = plt.imread(fname)
        plt.figure(figsize=(20, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
    plt.show()   # ← jedno show() na końcu otwiera oba wykresy naraz