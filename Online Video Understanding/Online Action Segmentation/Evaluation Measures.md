### Acc
Accuracy or MoF (mean over frames) are an per-frame accuracy measure that calculates the ratio of frames that are correctly recognized by the temporal action model:

$$Acc=\frac {\text{number of correct frames}}{\text{number of all frames}}$$

### F1 score
The F1-score, or F1@$\tau$ compares the Intersection over Union (IoU) of each segment with respect to the corre- sponding ground truth based on some threshold τ/100. τ are set to 10,25,50. A segment is considered a true positive if its score with respect to the ground truth exceeds the threshold. If there is more than one correct segment within the span of a single ground truth action, then only one is marked as a true positive and the others are marked as false positives.

$$F1=2\cdot \frac{precision \times recall}{precision+recall}$$

### Edit score
The Edit Score is computed using the Levenshtein distance e, which quantifies **how similar two sequences are to each other** by counting the minimum number of operations required to convert one (segment) string into another.

$$Edit=(1−\frac{e(X,Y)}{max(|X|,|Y|)})⋅100$$