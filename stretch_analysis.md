# Stretch 6B-S1 — Embedding Space Explorer: Analysis

## Dimensionality Reduction Choice: t-SNE

I chose **t-SNE** over PCA for both visualizations. t-SNE optimizes to preserve
*local* structure — points that are nearby in the original high-dimensional space
remain nearby in 2D — which makes semantic clusters visually sharp and
interpretable. PCA, by contrast, preserves global variance and tends to produce
overlapping blobs when the categories are not linearly separable. Since neither
GloVe semantic categories nor DistilBERT document topics are expected to be
linearly separable, t-SNE is the better fit. For the word plot I used
`perplexity=30` (appropriate for ~200 points); for the document plot
`perplexity=5` (required to stay below the 20-point sample size).

---

## Word Embedding Space (GloVe, 219 words, 5 categories)

The t-SNE projection reveals that GloVe's 50-dimensional vectors encode clear
semantic structure across all five categories. The **Sports** cluster (green) is
the tightest and most cohesive group in the plot, appearing in the far left
region with words like *football*, *tennis*, and *marathon* sitting close
together — reflecting that sport vocabulary consistently co-occurs in similar
athletic contexts in the training corpus.

**Countries** (red) form a well-defined cluster in the upper-right region.
Country names share strong geopolitical and geographic co-occurrence patterns,
which GloVe captures reliably. **Technology** (blue) occupies the central region,
with *algorithm* and *internet* annotated near the cluster core and *cloud*
appearing slightly lower — suggesting that *cloud* has broader contextual usage
beyond pure technology.

The **Emotions** cluster (orange) is positioned in the lower-center region.
Words like *love* and *happiness* are annotated within the cluster, and the group
shows moderate internal spread, reflecting the diversity of emotional vocabulary
in context. **Finance** (purple) appears in the right-center region, partially
overlapping with the Countries cluster at the boundary — words like *stock*,
*inflation*, and *bankruptcy* are annotated and clearly within the finance zone,
but the proximity to Countries suggests that financial news frequently references
nations in the training corpus.

The most notable outlier pattern is the boundary overlap between Finance and
Countries, which makes semantic sense: financial reporting is heavily
geography-dependent. All five categories are nonetheless visually separable,
confirming that GloVe captures meaningful domain-level structure in its
50-dimensional vectors.

---

## Document Embedding Space (DistilBERT, 20 BBC News articles)

The DistilBERT document embeddings produce clear category separation across the
20 articles. **Sport** (green) forms the tightest and most isolated cluster,
positioned in the upper-center region with four articles grouped closely together
— headlines like *"sydney return for henin-hardenne"* and *"republic to face
china and italy"* sit near each other, confirming that sports articles share
strong vocabulary and framing patterns that DistilBERT captures reliably.

**Tech** (purple) clusters in the left region with articles like *"sony psp
console hits us"*, *"halo 2 heralds traffic explosion"*, and *"mobiles not media
players yet"* appearing near each other, though with slightly more spread than
Sport — reflecting that tech journalism covers a broader range of sub-topics
(hardware, software, mobile).

**Entertainment** (orange) is the most dispersed category, with articles
scattered across the right side of the plot. Headlines like *"duran duran show
set for us tv"* and *"uk tv channel rapped for csi ad"* appear far apart,
confirming that entertainment news is heterogeneous in vocabulary (music, film,
television) and does not cluster as tightly as domain-specific categories.

**Business** (red) and **Politics** (blue) appear in adjacent regions on the
left-bottom area of the plot — *"us trade gap ballooned"* and *"india calls for
fair trade rules"* sit near politics articles like *"cabinet anger at brown cash
raid"* and *"blunkett unveils policing plans"*. This proximity reflects genuine
content overlap: economic policy reporting blends financial and political
language, which DistilBERT correctly encodes as semantic similarity.

Overall, the clear separation of Sport and Tech, the dispersion of Entertainment,
and the adjacency of Business and Politics all validate that DistilBERT's
768-dimensional embeddings capture strong topic-level signal that survives t-SNE
dimensionality reduction to 2D.