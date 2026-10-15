# Reddit Content Opportunity Finder

Finds content opportunities for a brand in scraped Reddit comments using Apify in a previous stage. The framework cleans the raw export, groups comments into topics using HDBSCAN clustering, scores each comment with Jev (TypeSafe), and flags comments where the brand could publish new content and no competitor is mentioned.

## How it works

The pipeline has two stages. Keep them in **two separate scripts** (each has its own `main()`).

### Stage 1: Clean (`clean_reddit.py`)

Input: `scraped_reddit.xlsx` (raw export). Output: `scraped_reddit_clean.xlsx`.

1. Maps column names (`text`, `selftext`, `upvotes`, `created_utc`, ...) to `body`, `score`, `timestamp`.
2. Drops deleted/removed comments and bots (by author name if present, otherwise by boilerplate text).
3. Cleans the body: HTML entities, spoiler tags, quoted replies (`>`), Markdown, URLs, smart punctuation, non-ASCII.
4. Drops rows left empty after cleaning, and converts timestamps to UTC.

### Stage 2: Cluster and score (`cluster_reddit.py`)

Input: `scraped_reddit_clean.xlsx`. Output: an Excel workbook and an HTML heatmap.

1. Asks for the target brand, competitors and extra terms.
2. Drops rows with a Reddit `score` below `MIN_REDDIT_SCORE` (10), including rows with no score.
3. Verifies brand and competitor mentions: a regex finds candidates, then Google Natural Language confirms they are real organisation or product entities (falls back to regex only if the API is unavailable).
4. Embeds comments (`all-MiniLM-L6-v2`), reduces with UMAP and clusters with HDBSCAN.
5. Removes the noise cluster (HDBSCAN label `-1`).
6. Names each cluster with Gemini (`gemini-2.5-flash`), falling back to TF-IDF terms if Gemini fails.
7. Scores comments that do not mention the target brand with Jev. The score is the probability (0-1) that the brand can write new, useful content about the comment.
8. Sets `content_opportunity` to **Yes** when `body_score >= OPPORTUNITY_THRESHOLD` (0.8) **and** no competitor is mentioned.
9. Writes the workbook and a cluster x term mentions heatmap.

## Requirements

- Google Colab
- Packages:

```bash
pip install sentence-transformers hdbscan umap-learn scikit-learn openpyxl \
            typesafe-sdk numpy pandas plotly google-genai google-cloud-language
```

- Credentials:

| Service | Used for | How to provide |
|---|---|---|
| Gemini | Cluster names | `GEMINI_API_KEY` (env var or Colab Secret) |
| TypeSafe (Jev) | `body_score` | `TYPESAFE_API_KEY` (env var or Colab Secret) |
| Google Cloud Natural Language | Brand entity check | OAuth / Application Default Credentials plus a GCP project with **Cloud Natural Language API** enabled and billing on. The API **does not accept API keys.** |

In Colab, the script opens a Google sign-in popup. Elsewhere run `gcloud auth application-default login`. Set `GCP_PROJECT` in the script, or the `GOOGLE_CLOUD_PROJECT` secret, or enter it when prompted.

Missing keys degrade gracefully: no Gemini key gives TF-IDF cluster names, no TypeSafe key leaves `body_score` empty, and no NLP access falls back to regex matching.

## Input format

An `.xlsx` with at least these columns (after stage 1):

| Column | Required | Notes |
|---|---|---|
| `body` | yes | Comment text |
| `score` | yes | Reddit score; numeric |
| `timestamp` | no | Created to NaN if missing |

## Usage

1. Run stage 1 to produce `scraped_reddit_clean.xlsx`.
2. Set `IN_PATH`, `OUT_PATH` and `HEATMAP_PATH` at the top of stage 2.
3. Run stage 2 and answer the prompts:

```
Brand to find content opportunities for: {}
Competitors to verify and count, comma-separated: {}
Other terms to count without entity check, e.g. destinations or nothing ('-' = none): -
```

Empty competitor or term answers fall back to `DEFAULT_COMPETITORS` and `DEFAULT_EXTRA_TERMS`. Brands other than TUI also ask for a one-line description, which is sent to Jev as context.

## Configuration

| Constant | Default | Meaning |
|---|---|---|
| `MIN_REDDIT_SCORE` | `10` | Rows below this (or with no score) are dropped |
| `OPPORTUNITY_THRESHOLD` | `0.8` | Minimum `body_score` for a content opportunity |
| `MIN_CLUSTER_SIZE` / `MIN_SAMPLES` | `5` / `2` | HDBSCAN density settings |
| `UMAP_NEIGHBORS` / `UMAP_COMPONENTS` | `15` / `5` | UMAP settings |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Cluster naming model |
| `JEV_MODEL` | `jev-1.13.0` | Pinned Jev model |
| `JEV_WORKERS` | `8` | Parallel Jev calls |
| `HIDE_EMPTY_HEATMAP_ROWS` | `True` | Hide clusters with no tracked mentions |

## Output

### Workbook: sheet `bodies`

| Column | Description |
|---|---|
| `body` | Cleaned comment text |
| `score` | Reddit score (always >= 10) |
| `timestamp` | Comment time (UTC) |
| `cluster_name` | Gemini-generated topic name |
| `branded_mention` | Yes if the target brand is mentioned (entity-verified) |
| `competitor_mention` | Yes if any listed competitor is mentioned |
| `body_score` | Jev probability 0-1; blank for branded comments or when no TypeSafe key |
| `content_opportunity` | Yes / No; blank when there is no score |

Example rows (illustrative, shortened):

| body | score | timestamp | cluster_name | branded_mention | competitor_mention | body_score | content_opportunity |
|---|---|---|---|---|---|---|---|
| We did the Basel - Amsterdam tour last month. Only on the Rhine from Basel to Strasbourg, then a long coach day... | 24 | 2026-07-03 | Rhine itinerary expectations | No | No | 0.91 | Yes |
| Viking was worth the extra money for the cabins and food... | 41 | 2026-06-18 | Comparing river cruise lines | No | Yes | 0.88 | No |
| Is travel insurance included with TUI river cruises? | 12 | 2026-08-02 | Booking and insurance questions | Yes | No | | |
| Rhine in April, is it too cold for the sun deck? | 15 | 2026-05-11 | Best season for river cruises | No | No | 0.62 | No |

### Heatmap

`brand_mentions_heatmap.html` shows clusters (rows) against brands, competitors and extra terms (columns), with mention counts in each cell. The brand and competitor columns use entity-verified counts.


## Choosing the score threshold

Jev answers a yes/no question, so scores tend to bunch high. Label about 100 comments Yes/No by hand and pick `OPPORTUNITY_THRESHOLD` from precision and recall instead of trusting 0.8. With those labels you can also train a SetFit or logistic-regression classifier on the existing MiniLM embeddings.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `API keys are not supported by this API` | Natural Language needs OAuth/ADC, not an API key |
| Gemini `Unterminated string` | Thinking tokens eat the output budget; the script sets `thinking_budget=0` and retries |
| Jev `400 Noul question must have criteria or instructions` | The question sent to Jev was empty; make sure `DEFAULT_JEV_INSTRUCTION` is defined and used when the question is blank |
| Every comment is noise | Lower `MIN_CLUSTER_SIZE` or supply more rows |
| Scores like `"1.2k"` all dropped | `score` must be numeric; convert before running |
