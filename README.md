# Amazon Reviews 2023 (Video Games): Sequential + Text-aware Recommender

Recommend the next 10 products a user is likely to buy **and rate 4-5 stars**, using the order of their past
purchases and the products' text. New products with almost no reviews are recommended from their description.

Data: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (McAuley Lab), Video
Games category: 4.6M reviews and 137k products. Video Games was chosen over Beauty (23.9M ratings) and Sports
(19.6M), which are 4-5 times bigger and too heavy for CPU training on an 8 GB laptop.

| Code | What it does |
|---|---|
| `amzrec/data.py` | streaming loader (reviews + metadata -> Parquet), EDA, 5-core filter, product text |
| `amzrec/als.py` | implicit ALS with time decay (recent purchases weigh more) |
| `amzrec/itemknn.py` | item-kNN: "people who bought this also bought" |
| `amzrec/text.py` | MiniLM sentence embeddings of product text (TF-IDF fallback) |
| `amzrec/sasrec.py` | SASRec transformer and the text-aware variant |
| `amzrec/pipeline.py` | time-based splits, scoring, rank fusion, metrics |
| `amzrec/train.py` | tuning on validation, evaluation on test, serving artifacts |
| `amzrec/api.py` | FastAPI service |
| `tests/` | tests on small synthetic files with the real JSON format |

## 1. Data exploration

| | All data | 5-core (used for modelling) |
|---|---|---|
| Reviews | 4,555,500 | 814,586 (18% kept) |
| Users | 2,766,656 | 94,762 (3.4% kept) |
| Products | 137,249 | 25,612 |
| Reviews per user (median / 90th pct) | 1 / 3 | 6 / 14 |
| Reviews per product (median / 99th pct) | 4 / 537 | 12 / 340 |
| Rated 4-5 stars | 74.5% | 80.4% |
| Period | 1998 - Sep 2023 | |

- **Very sparse.** 73% of users wrote one review and only 4% wrote five or more. Sequence models need a history,
  so the standard 5-core filter is used (users and products with at least 5 reviews). The numbers match the
  official benchmark for this category (814,586 / 94,762 / 25,612). Users outside the 5-core are served by the API
  from whatever products they give (`POST /next-items`) plus popularity.
- **J-shaped ratings.** 61% are 5 stars, but 12.7% are 1 star. A recommendation only counts as correct if the user
  bought the product and rated it 4-5 stars. 1-3 star purchases still appear in the input sequence as context.
- **Long tail.** The median product has 4 reviews while the top 1% have more than 500, so results are also reported
  separately for popular and long-tail products.

## 2. Approach

**Product text** (one string per product): title, category, store, details, what buyers say (titles of the two
most recent 4-5 star reviews written before the validation period), then features and description. It is encoded
with MiniLM (all-MiniLM-L6-v2, 384 dimensions).

| Model | Idea |
|---|---|
| popularity | most liked products of the last 90 days |
| item-kNN | products bought by the same users as the user's recent purchases |
| ALS | implicit matrix factorisation; each purchase weighted by 0.5^(age / half-life) |
| content | cosine between the user's taste profile (mean text vector) and each product |
| **SASRec** | a 2-layer transformer reads the user's last 50 purchases in order and predicts the next one |
| **SASRec + text** | product vector = ID embedding + projected MiniLM vector. Training randomly switches off 20% of the ID embeddings, so the model also learns from text, and a new product can be scored from its text alone |
| **fusion** | weighted reciprocal-rank fusion of all of the above; weights tuned on validation |

**Time decay.** The test period is 2023, and old purchases describe old tastes. Every purchase is weighted by
0.5^(age / half-life):
- in ALS, the half-life is chosen on validation (365 days won)
- in SASRec's training loss, the same half-life is used, so recent purchases count more

**Evaluation (no leakage):**
- **Validation:** train on everything before 2022-09-08 and predict 2022-09-08 to 2023-03-06. This chooses the
  ALS half-life, the SASRec epochs (early stopping), the fusion weights and the text-to-factors mapping.
- **Test:** train on everything before 2023-03-07 and predict 2023-03-07 to 2023-09-02. Nothing is tuned here.
- **Old products:** every product that existed before the cutoff (25k) is ranked, minus what the user already bought.
- **New products:** only the products first reviewed during the test period are ranked.
- **Metrics:** Recall@10, NDCG@10 (headline), HitRate@10 and catalogue coverage.

## 3. Results (test period: 2,177 users, Mar-Sep 2023)

| Old products | Recall@10 | NDCG@10 | HitRate@10 |
|---|---|---|---|
| popularity (last 90 days) | 0.028 | 0.0209 | 0.043 |
| item-kNN | 0.018 | 0.0109 | 0.029 |
| ALS (time decay) | 0.029 | 0.0166 | 0.044 |
| content (MiniLM) | 0.008 | 0.0045 | 0.014 |
| SASRec | 0.027 | 0.0142 | 0.041 |
| **SASRec + text** | **0.046** | **0.0257** | **0.070** |
| **fusion** | **0.061** | **0.0358** | **0.092** |

- **SASRec + text is the best single model**, 23% above popularity and 81% above SASRec without text.
- **Fusion is 71% above popularity.** It hits at least one liked product in the top 10 for 9.2% of users, vs 4.3%.
- **Time decay helped.** Before adding it to SASRec's loss: SASRec 0.0121, SASRec + text 0.0204, fusion 0.0335. After:
  0.0142 (+17%), 0.0257 (+26%) and 0.0358 (+7%).
- **Long-tail products:** popularity finds none (recall 0.000), while SASRec + text reaches 0.022 and ALS 0.011.

| NDCG@10 by user history | Users | ALS | SASRec + text | Fusion |
|---|---|---|---|---|
| 1-4 likes | 1,077 | 0.015 | 0.026 | 0.034 |
| 5-9 likes | 757 | 0.019 | 0.024 | 0.038 |
| 10-19 likes | 256 | 0.010 | 0.020 | 0.025 |
| 20+ likes | 87 | 0.033 | 0.054 | 0.070 |

| New products (150 first reviewed in the test period) | Recall@10 | NDCG@10 |
|---|---|---|
| random | 0.065 | 0.039 |
| content | 0.181 | 0.109 |
| text -> ALS factors | 0.186 | 0.112 |
| SASRec + text (text only) | 0.169 | 0.095 |
| **fusion** | **0.205** | **0.119** |

- **New products are recommended about 3 times better than random**, from their description alone.

**Why the absolute numbers are low:**
- Only 2,177 of the 94,762 5-core users bought (and liked) something in the last 6 months of the data.
- They have to be found among 25k products.

The same order of magnitude is normal for temporal splits on Amazon data. The comparison between models, all on the
same users and products, is what matters.

**Runtime on an 8 GB laptop (CPU):**
- One-time conversion of the 2.7 GB file: about 2 minutes.
- MiniLM encoding of 25.6k products: about 17 minutes, cached for later runs.
- Training and evaluation: about 40 minutes, mostly the four SASRec trainings (validation and test phases).

## 4. Run (Windows PowerShell)

```powershell
cd amazon-recsys
python -m pytest                                   # offline tests
python -m amzrec.train --raw data\raw --eda-only   # data exploration
python -m amzrec.train --raw data\raw              # full training + evaluation (~40 min after the first run)
python -m uvicorn amzrec.api:app --port 8007       # API -> http://localhost:8007/docs
```

Data files go in `data\raw\`: `Video_Games.jsonl` and `meta_Video_Games.jsonl` (plain or `.gz`), from the Hugging Face
dataset (`raw/review_categories/` and `raw/meta_categories/`).

## 5. API

| Endpoint | Purpose |
|---|---|
| `GET /users/sample` | real users with a purchase in the last year of the data |
| `GET /users/{user_id}/recommendations` | the last 5 purchases (with stars), 10 recommendations with reasons, and new products for this user |
| `POST /next-items` | a list of product ids (a new visitor or the current basket) -> what to buy next, without retraining |
| `GET /items/{item_id}` | product details |
| `GET /items/{item_id}/similar` | bought together (item-kNN), bought next (SASRec), similar description (MiniLM) |
| `GET /search?q=...` | search by meaning, optionally re-ranked for a user |
| `GET /model-info`, `GET /health` | EDA, results and status |

Each recommendation lists its reasons, for example:
- "people who bought 'X' also bought this"
- "fits the order of your recent purchases (sequence model)"
- "similar to 'Y' (text match 0.63)"
- "popular in the last 90 days"
- "new product, picked from its description"

`user_id` values start with "A"; product ids (`parent_asin`) start with "B0".

## 6. Limitations and next steps

- **Users with fewer than 5 reviews** (96% of all users) are not in training. They are served from their basket
  and popularity; a model trained on all users with short sequences would cover them better.
- **More data per model:** SASRec trains on CPU in about 50 s per epoch. A GPU would allow bigger models, longer
  training and more categories (Beauty, Sports).
- **Better text:** a stronger embedding model, or one fine-tuned on "bought together" pairs, would improve new-product
  recommendations.
- **Learned ranker:** a LightGBM ranker over the fused candidates (with features such as time since last purchase,
  price and rating) would replace the fixed fusion weights.
