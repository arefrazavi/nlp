# NLP

## Terminology

## How to turn the text content into numerical feature vectors.

- Bags of Words (BOW):     
  The most intuitive way to do so is to use a bags of words representation: 
  1) Assign a fixed integer id to each word occurring in any document of the training set (for instance by building a dictionary from words to integer indices).    
  2) For each document #i, count the number of occurrences of each word w and store it in X[i, j] as the value of feature #j where j is the index of word w in the dictionary.
    
## Scikit-learn:
Vectorizers (Transformers):

- CountVectorizer: 
Builds a dictionary of features and transforms documents to feature vectors.    
Convert a collection of text documents to a matrix of token counts.

- TfifVectorizer (Term Frequency times Inverse Document Frequency):
  Occurrence count (CountVectorizer) is a good start but there is an issue: longer documents will have higher average count values than shorter documents, even though they might talk about the same topics.   
  To avoid these potential discrepancies it suffices to divide the number of occurrences of each word in a document by the total number of words in the document: 
  these new features are called tf for Term Frequencies.        
  Another refinement on top of tf is to downscale weights for words that occur in many documents in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus.  

## N-grams
N-gram: means a sequence of N words
Now if we assign a probability to the occurrence of an N-gram or the probability of a word occurring next in a sequence of words
an N-gram model predicts the occurrence of a word based on the occurrence of its N – 1 previous words. 


## Deep Learning Frameworks
