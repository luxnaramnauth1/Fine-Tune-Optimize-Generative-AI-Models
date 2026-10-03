# Computed Metrics Analysis

Data: 10 articles x 3 model versions (V1, V2, V3) scored against human reference summaries.
BLEU = corpus BLEU (sacrebleu, 0-1 scale); ROUGE = F-measure averaged over articles (rouge_score, no stemming).
These are equivalent implementations of the Hugging Face `evaluate` metrics; values may differ slightly.

## Metric scores

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|---|
| V1 | 0.0493 | 0.4026 | 0.1281 | 0.3152 | pending (run compute_bertscore.py) |
| V2 | 0.4878 | 0.7404 | 0.5553 | 0.6879 | pending (run compute_bertscore.py) |
| V3 | 0.1202 | 0.4835 | 0.1899 | 0.3686 | pending (run compute_bertscore.py) |

BERTScore could not be computed here because it needs to download roberta-large from Hugging Face, which was unreachable. Run `python compute_bertscore.py "<Module 2 Lab folder>"` locally and paste the three F1 values into the last column.

## Highest-scoring model per metric

| Metric | Highest | What it measures |
|---|---|---|
| BLEU | V2 | Exact n-gram precision against the reference |
| ROUGE-1 / ROUGE-2 | V2 | Unigram / bigram recall of reference content |
| ROUGE-L | V2 | Longest common subsequence (word order and content) |
| BERTScore F1 | pending | Semantic similarity from contextual embeddings |

## Comparison (156 words)

BLEU, ROUGE-1, ROUGE-2 and ROUGE-L agree completely: V2 ranks first, V3 second and V1 last. V2 leads by a wide margin (BLEU 0.488 against 0.120 for V3 and 0.049 for V1), so on this dataset the metrics do identify the best summarizer. The disagreement is in what the ranking implies. V3 beats V1 on every metric, yet my manual review found factual errors in six of V3's ten summaries (wrong revenue, wrong dates, a false claim about stolen Social Security numbers) and none in V1, whose problem is vagueness. Within V3, summaries with errors score almost the same ROUGE-L as correct ones (0.358 against 0.385). BLEU counts exact n-gram overlap and is harshest on paraphrase; ROUGE measures how much reference content is recalled; BERTScore compares contextual embeddings and should reward paraphrases that BLEU punishes. None of them checks facts against the source. BERTScore must be run locally with compute_bertscore.py; its ranking should be added before submission.
