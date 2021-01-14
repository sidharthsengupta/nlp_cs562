#!/usr/bin/env python3

# Sidharth Sengupta
# CS 562 HW 1 Part 3
# Usage: cat CORPUS_FILE | ./sengupta_part3.py
#
# Reads corpus text from stdin and performs analysis relevant to
# part 3 of HW, including counting unigram and bigram frequencies, 
# bigram PMI, plotting, and various common corpus processing steps.

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.collocations import *

def wordCollocation(tokens, filter_n, bigram_n):
    """
    Takes list of tokens and uses nltk frequency distribution function
    to count unique unigrams and bigrams in corpus. Then uses nltk 
    finder class to calcualte pointwise mutual information scores for 
    top bigrams, printing all results to stdout. Contains optional 
    filtering step to remove low-occuring bigrams.

    Args:
        tokens (list): list of word token strings
        filter_n: threshold of bigram frequency
        bigram_n: number of top bigrams to output

    Returns:
        None
    """
    unigram_freq = nltk.FreqDist(tokens)
    bigram_freq = nltk.FreqDist(ngrams(tokens, 2))
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(filter_n)
    print("\nTop 30 highest-PMI word pairs with filter of ",
            filter_n,
            ":\n") 

    print("W1\t\t\t\tPMI\tW1 Freq\tW2 Freq\tBG Freq")
    for bg in finder.nbest(bigram_measures.pmi, bigram_n):
        print("{: <32}{: <8}{: <8}{: <8}{: <8}".format(bg[0] + ", " + bg[1],
            round(finder.score_ngram(bigram_measures.pmi, bg[0], bg[1]), 3),
            unigram_freq[bg[0]],
            unigram_freq[bg[1]],
            bigram_freq[bg]))

def generateRankFreqPlot(tokens):
    """
    Takes list of tokens and uses nltk frequency distribution function
    to count unique unigrams in corpus. Then uses matplotlib to plot
    log transformed rank and frequency of each unigram to a 'plot.png'
    file within the working directory.

    Args:
        tokens (list): list of word token strings

    Returns:
        None
    """
    unigram_freq = nltk.FreqDist(tokens)
    x = np.log10(list(range(1, len(unigram_freq.most_common()) + 1)))
    y = np.log10([(freq[1]) for freq in unigram_freq.most_common()])
    plt.scatter(x,y)
    plt.plot(x,y)
    plt.title("Rank-Frequency Plot")
    plt.xlabel("log(rank)")
    plt.ylabel("log(frequency)")
    plt.savefig('./plot.png')
    plt.savefig('foo.png')

def mostCommonTokens(tokens):
    """
    Takes list of tokens and uses nltk frequency distribution function
    to count unique unigrams in corpus. Then uses most_common member
    function to print top 30 highest occuring unigrams and their counts
    to stdout.

    Args:
        tokens (list): list of word token strings

    Returns:
        None
    """
    unigram_freq = nltk.FreqDist(tokens)
    print("Top 30 Words:")
    for freq in unigram_freq.most_common(30):
        print(freq)

def countUniqueNgrams(tokens, n):
    """
    Takes list of tokens and uses nltk frequency distribution function
    to count unique n-grams in corpus, printing result to stdout.
    
    Args:
        tokens (list): list of word token strings
        n (int): number for n-gram

    Returns:
        None
    """
    ngram_freq = nltk.FreqDist(ngrams(tokens, n))
    print("Unique ", n, "-grams:\t", len(ngram_freq), sep = '')

def processText(text):
    """
    Uses nltk Whitespace tokenizer to tokenize raw corpus text into 
    list of word tokens. Then proceeds to perform analysis relevant 
    to part 3 of HW, including:
    - Count unique bigrams
    - Count unqiue unigrams
    - Count top 30 common unigrams
    - Create Rank-Freq Plot
    - Remove tokens in nltk stopword english set
    - Count unique bigrams
    - Count unique unigrams
    - Count top 30 common unigrams
    - Calculate top bigrams scored by PMI using 0, 5, 10, 20, 100
        frequency thresholds
    - Calculate PMI of "NEW YORK" bigram

    Args:
        text (str): corpus text

    Returns:
        None
    """
    tokens = WhitespaceTokenizer().tokenize(text)

    print("WITH STOPWORDS:\n")
    countUniqueNgrams(tokens, 2)
    countUniqueNgrams(tokens, 1)
    mostCommonTokens(tokens)

    generateRankFreqPlot(tokens)
  
    stop_engl = stopwords.words('english')
    print("\nEnglish Stopwords:")
    print(stop_engl, '\n')

    print("WITHOUT STOPWORDS:")
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t not in stop_engl]
    tokens = [t.upper() for t in tokens]
    print("here")
    countUniqueNgrams(tokens, 2)
    countUniqueNgrams(tokens, 1)
    mostCommonTokens(tokens)

    print("\nWord Collocation:\n")
    wordCollocation(tokens, 0, 30)
    wordCollocation(tokens, 5, 30)
    wordCollocation(tokens, 10, 30)
    wordCollocation(tokens, 20, 30)
    wordCollocation(tokens, 100, 10)

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    print("New York PMI:")
    print(finder.score_ngram(bigram_measures.pmi, "NEW", "YORK"))

def main():
    if (sys.stdin):
        processText(sys.stdin.read())

if __name__ == "__main__":
    main()
