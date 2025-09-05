-----

# Hepsiburada Address → Label (TR) — End-to-End Pipeline

Bu repo, Hepsiburada yarışması için geliştirilmiş, Türkçe adres metinlerinden **10.000'den fazla sınıfa sahip etiketleri tahmin eden uçtan uca bir makine öğrenimi boru hattını** özetler.

## Amaç

Türkçe adres metninden **10.000'den fazla sınıflı etiket tahmini** yapmaktır.

## Yaklaşım

  * **Ana Model:** fastText (normalize edilmiş metin + word/char n-gram)
  * **Kural Katmanı:** normalize/fingerprint/libpostal ile majority-map
  * **Zayıf Örneklerde Düzeltme:** TF-IDF centroid ve/veya SBERT+FAISS rerank
  * **Gating:** Yalnızca belirsiz fastText tahminlerini değiştir (p@1 ve margin eşikleri)

Bu repoda, yarışma sürecinde denenen tüm yöntemler özetlenmiştir: fastText, kural katmanı, TF-IDF / SBERT rerank, BERTurk proto-centroid ve fine-tune (LLRD/EMA), kalibrasyon ve dağılım kontrolleri.

-----
flowchart TD
    %% =============== INPUT ===============
    A[[Raw Address Texts\n(data/train.csv, data/test.csv)]]

    %% =============== NORMALIZATION ===============
    subgraph N[Normalization & Keys (src/normalize.py)]
        N1[Expand Abbrev:\nmah/mh→mahalle,\ncad/cd→cadde, sk→sokak, blv→bulvar...]
        N2[Number Std:\nno:12→no 12, kat5→kat 5, d12→d 12]
        N3[Split Alnum:\na12→a 12, 12a→12 a]
        N4[ASCII View (unidecode)]
        N5[Fingerprint:\nunique sorted tokens]
    end

    A --> N1 --> N2 --> N3 --> N4 --> N5

    K1[[norm]]
    K2[[norm_ascii]]
    K3[[fp]]
    N1 -->|build| K1
    N4 -->|build| K2
    N5 -->|build| K3

    %% =============== RULE LAYER ===============
    subgraph R[Rule Layer (src/rules.py)]
        R1[Majority-Map by Key\n(norm / norm_ascii / fp)]
        R2{Key Hit?}
    end

    K1 --> R1
    K2 --> R1
    K3 --> R1
    R1 --> R2

    R2 -->|Yes| L_rule[[Label (Rule)]]
    R2 -->|No| MLPATH[Go to ML]

    %% =============== ML: FASTTEXT ===============
    subgraph F[ML Path — fastText]
        F0[[ft_train.txt]]
        F1[Train fastText\nlr=0.7, epoch=23,\nwordNgrams=4, dim=300]
        F2[Predict k=5\n(top-k labels & probs)]
        F3{Gating\np@1 < P1_THR OR\n(p1-p2) < M_THR?}
    end

    A --> F0
    F0 --> F1
    F1 --> F2
    MLPATH --> F2
    F2 --> F3

    F3 -->|No (Strong)| L_ft[[Label = Top-1 (fastText)]]

    %% =============== RERANKERS ===============
    subgraph RR[Rerankers (Weak Only)]
        direction LR
        subgraph T[TF-IDF Centroid]
            T1[char(3–5) TF-IDF\ntrain → per-label centroids]
            T2[cosine(X_weak, C_label)\nselect best in ft top-k]
        end
        subgraph S[SBERT + FAISS (optional)]
            S1[Encode with paraphrase-multilingual-mpnet-base-v2]
            S2[FAISS search (K)] 
            S3[Blend score = α·FAISS + (1-α)·p_fastText]
        end
        Rpick{Best over candidates?}
    end

    F3 -->|Yes (Weak)| T1 --> T2 --> Rpick
    F3 -->|Yes (Weak)| S1 --> S2 --> S3 --> Rpick
    Rpick --> L_rerank[[Label = Reranked]]

    %% =============== MERGE & OUTPUT ===============
    subgraph MRG[Merge & Submission]
        M1{Has Rule Label?}
        M2[Final Label = Rule else ML/Rerank]
        M3[(Distribution Checks:\nlabel freq, uniques)]
        M4[[submission.csv]]
    end

    L_rule --> M1
    L_ft --> M1
    L_rerank --> M1
    M1 -->|Yes (Rule)| M2
    M1 -->|No (Use ML)| M2
    M2 --> M3 --> M4

    %% =============== CACHE & MODELS ===============
    subgraph C[Artifacts & Caching]
        C1[(runs/: ft_probs.npy,\nft_labels.npy,\npred_ml_top1.csv)]
        C2[(models/: fastText .bin/.ftz)]
        C3[(Drive save/load\noptional)]
    end

    F2 --> C1
    L_ft --> C1
    L_rerank --> C1
    F1 --> C2
    C1 --> C3
    C2 --> C3

-----
## İçindekiler

  * [Kurulum](https://www.google.com/search?q=%23kurulum)
  * [Veri Düzeni](https://www.google.com/search?q=%23veri-d%C3%BCzeni)
  * [Proje Yapısı](https://www.google.com/search?q=%23proje-yap%C4%B1s%C4%B1)
  * [Normalize & Anahtarlar](https://www.google.com/search?q=%23normalize--anahtarlar)
  * [Kural Katmanı (Majority-Map)](https://www.google.com/search?q=%23kural-katman%C4%B1-majority-map)
  * [fastText — Eğitim / Inference / Autotune](https://www.google.com/search?q=%23fasttext--e%C4%9Fitim--inference--autotune)
  * [Gating (Belirsizlik Eşikleri)](https://www.google.com/search?q=%23gating-belirsizlik-e%C5%9Fikleri)
  * [Reranker A: TF-IDF Centroid](https://www.google.com/search?q=%23reranker-a-tf-idf-centroid)
  * [Reranker B: SBERT + FAISS (opsiyonel)](https://www.google.com/search?q=%23reranker-b-sbert--faiss-opsiyonel)
  * [Submission & Tanı](https://www.google.com/search?q=%23submission--tan%C4%B1)
  * [Drive’a Kaydet / Geri Yükle](https://www.google.com/search?q=%23drivea-kaydet--geri-y%C3%BCkle)
  * [Denenen Yöntemler](https://www.google.com/search?q=%23denenen-y%C3%B6ntemler)
  * [FAQ / Hatalar](https://www.google.com/search?q=%23faq--hatalar)

-----

## Kurulum

```bash
pip install fasttext==0.9.2 unidecode scikit-learn pandas numpy

# (opsiyonel) libpostal kuralı için
# apt-get install -y libpostal && pip install postal

# (opsiyonel) SBERT + FAISS reranker
pip install sentence-transformers faiss-cpu

# (opsiyonel) BERT deneyleri için
pip install torch transformers accelerate
```

-----

## Veri Düzeni

Verilerinizi aşağıdaki yapıda tutun. **Verileri git'e koymayın; `.gitignore`'a ekleyin.**

```
data/
  ├─ train.csv               # kolonlar: address, label
  ├─ test.csv                # kolonlar: id, address
  └─ sample_submission.csv
```

-----

## Proje Yapısı

```
.
├─ notebooks/
│   └─ cyberknights-t3-hepsiburada.ipynb
├─ src/
│   ├─ normalize.py
│   ├─ rules.py
│   ├─ train_fasttext.py
│   ├─ rerank_tfidf.py
│   ├─ rerank_sbert.py
│   ├─ build_submission.py
│   └─ utils_io.py
├─ runs/              # cache: ft_probs/labels, loglar, ara çıktılar
├─ models/            # yerel modeller (bin/ftz)
└─ README.md
```

-----

## Normalize & Anahtarlar

Adresleri tek tipe indirger ve ML/kural için tutarlı girdi üretir.

  * **Kısaltma genişletme:** mah/mh/mhl→mahalle, cad/cd→cadde, sk/sok→sokak, blv→bulvar, apt/ap→apartman, sit→sitesi, osb→organize sanayi bolgesi
  * **Numara standardizasyonu:** no:12/no-12→no 12, kat5→kat 5, d12→d 12, blokb→blok b
  * **Sayı-harf ayırma:** a12→a 12, 12a→12 a
  * **ASCII görünüm:** `unidecode`
  * **Fingerprint:** Genel kelimeleri çıkarıp benzersiz sıralı tokenlar oluşturur.

<!-- end list -->

```python
# src/normalize.py (öz)
import re, unicodedata
from unidecode import unidecode

_ABBR = {
    r"\bmah\.?\b": "mahalle", r"\bmh\.?\b": "mahalle", r"\bmhl\b": "mahalle",
    r"\bsok\.?\b": "sokak",   r"\bsk\.?\b": "sokak",
    r"\bcadd?\.?\b": "cadde", r"\bcad\.?\b": "cadde", r"\bcd\.?\b": "cadde",
    r"\bblv\.?\b": "bulvar",  r"\bbulv?\.?\b": "bulvar",
    r"\bapt\.?\b": "apartman", r"\bap\.?\b": "apartman",
    r"\bsitesi?\b": "sitesi", r"\bsit\.\b": "sitesi",
    r"\bblok\b": "blok", r"\bosb\b": "organize sanayi bolgesi",
    r"\bdr\.?\b": "doktor", r"\bmerkez\b": "merkez"
}
_punct_re = re.compile(r"[^a-z0-9ğüşöçıİĞÜŞÖÇ]+", re.IGNORECASE)

def normalize_text(s:str)->str:
    if not isinstance(s,str): return ""
    s = (s or "").strip().casefold()
    for pat, rep in _ABBR.items(): s = re.sub(pat, rep, s)
    s = re.sub(r"\b(?:no|numara)\s*[:\-]?\s*([0-9]+(?:/[0-9a-z])?)\b", r"no \1", s)
    s = re.sub(r"\bd\.?\s*([0-9]+)\b", r"d \1", s)      # d12 → d 12
    s = re.sub(r"\bkat\s*([0-9]+)\b", r"kat \1", s)
    s = re.sub(r"\bblok\s*([a-z0-9]+)\b", r"blok \1", s)
    s = re.sub(r"([a-zğüşöçı])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zğüşöçı])", r"\1 \2", s)
    s = _punct_re.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_text_ascii(s:str)->str:
    return normalize_text(unidecode(s or ""))

def fingerprint(s:str)->str:
    s = normalize_text(s); toks = s.split()
    keep = {"no","kat","d","blok"}
    toks = [t for t in toks if (len(t)>1 or t in keep)]
    toks = sorted(set(toks))
    return " ".join(toks)

def build_text(df):
    return (df["norm"] + " || " + df["norm_ascii"]).astype(str)
```

-----

## Kural Katmanı (Majority-Map)

Normalize edilmiş anahtarlar üzerinden label çoğunluğunu kullanır.

```python
# src/rules.py (öz)
import pandas as pd

def majority_map(df, key_col:str, label_col:str, min_count=1, purity=0.0):
    vc = df.groupby(key_col)[label_col].value_counts().to_frame("cnt").reset_index()
    total = vc.groupby(key_col)["cnt"].transform("sum")
    vc["frac"] = vc["cnt"] / total
    vc = vc.sort_values(["cnt"]).drop_duplicates([key_col], keep="last")
    ok = (vc["cnt"]>=min_count) & (vc["frac"]>=purity)
    return pd.Series(vc.loc[ok,label_col].values, index=vc.loc[ok,key_col].values)
```

**Kullanım:**

```python
from src.normalize import normalize_text, normalize_text_ascii, fingerprint, build_text
from src.rules import majority_map

train["norm"] = train["address"].map(normalize_text)
train["norm_ascii"] = train["address"].map(normalize_text_ascii)
train["fp"] = train["address"].map(fingerprint)

norm2label       = majority_map(train, "norm", "label")
norm_ascii2label = majority_map(train, "norm_ascii", "label")
fp2label         = majority_map(train, "fp", "label")

pred_rule = pd.Series([None]*len(test), index=test.index)
for k, mp in [("norm",norm2label), ("norm_ascii",norm_ascii2label), ("fp",fp2label)]:
    m = pred_rule.isna() & test[k].isin(mp.index)
    pred_rule[m] = test.loc[m,k].map(mp)

need_ml_idx = pred_rule[pred_rule.isna()].index
print(f"[RULE] kapsama: %{(~pred_rule.isna()).mean()*100:.2f}")
```

Opsiyonel `libpostal` ek anahtar (`lp_key`) ile kapsama artışı uygulanabilir.

-----

## fastText — Eğitim / Inference / Autotune

**Eğitim dosyası:**

```python
with open("ft_train.txt","w",encoding="utf-8") as f:
    for y, txt in zip(train["label"].astype(str), build_text(train)):
        f.write(f"__label__{y} {txt}\n")
```

**Eğitim (stabil ayar):**

```python
import fasttext, os, time
CFG = dict(lr=0.7, epoch=23, wordNgrams=4, dim=300, loss="softmax", minn=2, maxn=5,
           thread=os.cpu_count(), verbose=3)
t0=time.perf_counter()
ft = fasttext.train_supervised(input="ft_train.txt", **CFG)
print(f"⏱ {time.perf_counter()-t0:.1f}s")
```

**Autotune (isteğe bağlı):**

```python
# train'ı tr/valid ayırıp: ft_train_tr.txt / ft_valid.txt üretin
ft_auto = fasttext.train_supervised(
    input="ft_train_tr.txt",
    autotuneValidationFile="ft_valid.txt",
    autotuneDuration=1800  # saniye
)
```

**Inference (yalnız ML gerekenler):**

```python
texts_need = build_text(test.loc[need_ml_idx]).tolist()
ft_labels, ft_probs = ft.predict(texts_need, k=5)
pred_ml_top1 = pd.Series(
    [labs[0].replace("__label__","") if labs else train["label"].mode()[0] for labs in ft_labels],
    index=need_ml_idx, dtype="object"
)
# cache
np.save("runs/ft_probs.npy", np.array(ft_probs, dtype=object), allow_pickle=True)
np.save("runs/ft_labels.npy", np.array(ft_labels, dtype=object), allow_pickle=True)
pred_ml_top1.to_csv("runs/pred_ml_top1.csv")
```

-----

## Gating (Belirsizlik Eşikleri)

Modelin en az emin olduğu örnekleri belirlemek için kullanılır.

```python
import numpy as np
def _p1p2(row):
    p1 = float(row[0]) if len(row)>0 else 0.0
    p2 = float(row[1]) if len(row)>1 else 0.0
    return p1, p2

ft_p1 = np.array([_p1p2(p)[0] for p in ft_probs], dtype=np.float32)
ft_p2 = np.array([_p1p2(p)[1] for p in ft_probs], dtype=np.float32)

P1_THR, M_THR = 0.50, 0.03  # başlamak için iyi band
weak_mask = (ft_p1 < P1_THR) | ((ft_p1 - ft_p2) < M_THR)
weak_idx_local = np.where(weak_mask)[0]
print(f"[CAL] zayıf oranı: {100*len(weak_idx_local)/len(ft_p1):.2f}%")
```

**Hedef:** Değiştirilen satır oranı %10–15. Çok düşükse eşiği gevşet, çok yüksekse sıkılaştır.

-----

## Reranker A: TF-IDF Centroid

En zayıf tahminler için TF-IDF tabanlı bir yeniden sıralama mekanizmasıdır.

```python
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from scipy.sparse import vstack

TOPK_FT = 5
ft_topk = [[l.replace("__label__","") for l in labs[:TOPK_FT]] for labs in ft_labels]

needed_labels = sorted({l for i in weak_idx_local for l in ft_topk[i]})
train_text = build_text(train).astype(str).tolist()
weak_texts = build_text(test.loc[need_ml_idx[weak_idx_local]]).astype(str).tolist()

tfv = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)
X_tr = tfv.fit_transform(train_text)
X_wk = tfv.transform(weak_texts)

y_lab = train["label"].astype(str).values
lab2rows = defaultdict(list)
for i, lab in enumerate(y_lab):
    if lab in needed_labels: lab2rows[lab].append(i)

rows, row_labs = [], []
for lab in needed_labels:
    idxs = lab2rows.get(lab, [])
    if not idxs: continue
    C = X_tr[idxs].mean(axis=0)          # CSR
    C = sk_normalize(C, norm="l2", copy=False)
    rows.append(C); row_labs.append(lab)

final_ml = pred_ml_top1.loc[need_ml_idx].astype(str).to_numpy()
if rows:
    C_mat = vstack(rows)                  # (L x D)
    X_wk  = sk_normalize(X_wk, norm="l2", copy=False)
    S     = X_wk @ C_mat.T                # cosine ~ dot

    changed = 0
    for pos, j in enumerate(weak_idx_local):
        cand = ft_topk[j];  base = float(ft_p1[j])
        best_lbl, best_sc = final_ml[j], -1e9
        for lbl in cand:
            try:
                r = row_labs.index(lbl)
            except ValueError:
                continue
            sc = float(S[pos, r])
            if sc > best_sc: best_lbl, best_sc = lbl, sc
        if best_lbl != final_ml[j] and best_sc > base + 1e-6:
            final_ml[j] = best_lbl; changed += 1
    print(f"[TF-IDF] değişen: {changed}/{len(weak_idx_local)}")
```

-----

## Reranker B: SBERT + FAISS (opsiyonel)

SBERT ile adres gömme vektörleri oluşturup, FAISS ile en yakın benzer label'ları bulur.

```python
import faiss, torch
from sentence_transformers import SentenceTransformer

SBERT_FAISS_DIR = "/content/drive/MyDrive/sbert_faiss_models/..."
index = faiss.read_index(f"{SBERT_FAISS_DIR}/index.faiss")
labels_ok = np.load(f"{SBERT_FAISS_DIR}/labels.npy", allow_pickle=True).tolist()

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
emb = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

weak_texts = build_text(test.loc[need_ml_idx[weak_idx_local]]).tolist()
Z = emb.encode(weak_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=128, show_progress_bar=False)

K_F = min(50, len(labels_ok))
D, I = index.search(Z.astype('float32'), K_F)

ALPHA = 0.5
final_ml = pred_ml_top1.loc[need_ml_idx].astype(str).to_numpy()
changed = 0
for pos, j in enumerate(weak_idx_local):
    cand_labs = ft_topk[j]; cand_probs = ft_probs[j][:len(cand_labs)]
    faiss_labs = [labels_ok[r] for r in I[pos] if r!=-1]; faiss_scos = D[pos]
    best_lbl, best_score = final_ml[j], -1e9; base = float(ft_p1[j])
    for lbl_idx, lbl in enumerate(cand_labs):
        sc = 0.0
        if lbl in faiss_labs:
            ridx = faiss_labs.index(lbl); sc = float(faiss_scos[ridx])
        p_ft = float(cand_probs[lbl_idx])
        score = ALPHA*sc + (1-ALPHA)*p_ft
        if score > best_score: best_lbl, best_score = lbl, score
    if best_lbl != final_ml[j] and best_score > base + 1e-6:
        final_ml[j] = best_lbl; changed += 1
print(f"[FAISS] değişen: {changed}/{len(weak_idx_local)}")
```

-----

## Submission & Tanı

Final tahminleri birleştirir ve dosya oluşturur.

```python
# final birleşim: kural + (rerank/ft)
final_pred = pred_rule.astype("object").copy()
final_pred.loc[need_ml_idx] = pd.Series(final_ml, index=need_ml_idx).astype(str).values
final_pred.fillna(str(train["label"].astype(str).value_counts().idxmax()), inplace=True)

submission = pd.DataFrame({"id": test["id"].astype(int),
                           "label": final_pred.astype(str)}).sort_values("id")
if submission["label"].str.fullmatch(r"\d+").all():
    submission["label"] = submission["label"].astype(int)
submission.to_csv("submission.csv", index=False)

vc = submission["label"].astype(str).value_counts(normalize=True)
print(f"satır={len(submission)} | tekil={submission['label'].nunique()} | top_label={vc.index[0]} pay={vc.iloc[0]*100:.2f}%")
```

-----

## Drive’a Kaydet / Geri Yükle

Modelleri ve ara çıktıları kaydetmek için kullanılır.

```python
# fastText modeli kaydet
ft.save_model("/content/drive/MyDrive/fasttext_models/fasttext_addr_YYYYMMDD_HHMMSS.bin")
# yükle
ft = fasttext.load_model("/content/drive/MyDrive/fasttext_models/fasttext_addr_YYYYMMDD_HHMMSS.bin")

# prediction cache
np.save("/content/drive/MyDrive/runs/ft_probs.npy", np.array(ft_probs, dtype=object), allow_pickle=True)
np.save("/content/drive/MyDrive/runs/ft_labels.npy", np.array(ft_labels, dtype=object), allow_pickle=True)
pred_ml_top1.to_csv("/content/drive/MyDrive/runs/pred_ml_top1.csv")
```

-----

## Denenen Yöntemler

  * **fastText:** `loss="softmax"` (stabil), alternatif `ova` (bazı sınıflar iyi ama riskli)
  * **Autotune:** `valid` set ile f1 optimizasyonu
  * **Warm-start:** Mevcut `.bin` → `.vec` çıkarıp `pretrainedVectors` ile küçük `lr` (0.2–0.3)
  * **Kural katmanı:** `norm`, `norm_ascii`, `fp` + `key_*` ve opsiyonel `libpostal` `lp_key`
  * **Rerank:** TF-IDF centroid (`char` 3–5) ve/veya SBERT+FAISS (hazır index)
  * **Gating:** `P1_THR≈0.40–0.55`, `M_THR≈0.02–0.05` hedef %10–15 değişim
  * **BERTurk:** `proto-centroid` + tam `fine-tune` (LLRD, EMA, sampler, LS, cosine)

-----

## FAQ / Hatalar

  * `pretrainedVectors dimension mismatch` → `.vec` boyutu ile `dim` aynı olmalı.
  * `Autotune time` → `autotuneDuration`’ı artır; `valid`'i küçült.
  * `KeyError 'norm'` → Normalize adımlarını çalıştırmadan ileri gitmeyin.
  * `np.matrix not supported` → TF-IDF centroid'te CSR'ı `sk_normalize` ile kullanın (örnek yukarıda).
  * `load_model uyarısı` → Normal, yeni API FastText objesi döndürüyor.
