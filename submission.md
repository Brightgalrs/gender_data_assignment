# Submission: Data assignment \#1: Gender classification from text

## Natural Language Processing and Psychology (Corpus Analysis)

### Name: Robert

### Results:

Feature set | Accuracy
------------|---------
Function words | 65.80%
POS | 72.28%
Lexical | 58.62%
Complexity | 55.04%
Topic models* | 56.88%
All features | 72.85%

*Extra credit

### How do your results compare to the Argamon and Ottenbacher papers? (Refer to both accuracies and which feature sets performed better)

Aramon:
part of speech | 0.76
function words  | 0.56

We underperformed in POS compared to Aramon, but outperform in function words.

Ottenbacher:
Feature(s) | Accuracy
Style | 64.55
Content | 65.03
Metadata | 72.95
Utility | 72.46
All | 73.71

All performed best in Ottenbacher's.

Ottenbacher's "Content" seems to be comparable to our "Topic models" - we underperform though.
Their "Style" seems to be comparable to our "Complexity" - we still underperformed.
