#!/usr/bin/env python3

# Sidharth Sengupta
# CS 562 HW 1 Part 2
# Usage: cat TEXT_FILE | ./sengupta_tokenization.py
#
# Reads text from stdin and performs sentence and word-level tokenization, 
# storing tokens in one sentence per line format in "corpus.txt" file.

import sys
from nltk import sent_tokenize, word_tokenize
import string

def saveCorpus(tokens):
    """
    Takes nested list of tokens and saves one sentence per line with 
    word tokens seperated by single white space character to 
    "corpus.txt" file.

    Args:
        tokens (list): nested list of token strings 

    Returns:
        None
    """
    with open("corpus.txt", "w") as f:
        for sent in tokens:
            print(*sent, sep = ' ', end = '\n', file = f)

def removePuncTokens(sent_tokens):
    """
    Takes nested list of tokens and removes any token that is a member of 
    nltk punctation set or any token that begins with a character in nltk
    punctuation set.

    Args:
        sent_tokens (list): nested list of sentence tokens separted into 
            word tokens

    Returns:
        Nested list of tokens with punctuation tokens removed
    """
    punct_set = set(string.punctuation)
    tokens = [[t for t in sent if t not in punct_set] for sent in sent_tokens]
    return [[t for t in sent if not t[0] in punct_set] for sent in sent_tokens]

def processText(unformatted_text):
    """
    Handles processing raw text into sentence and word tokens. Removes any
    newlines from supplied text before using nlkt's Punkt sentence 
    tokenization to separate text into sentences, and then using nltk's 
    defualt word tokenization procedure to tokenize each sentence into words. 
    Saves these tokens to 'corpus.txt' output file.

    Args:
        unformatted_text (str): string of unformatted text

    Returns:
        None
    """
    text = unformatted_text.replace('\n', ' ')
    sent_tokens = sent_tokenize(text)
    sent_tokens = [word_tokenize(sent) for sent in sent_tokens]
    sent_tokens = removePuncTokens(sent_tokens)
    sent_tokens = [[t.upper() for t in sent] for sent in sent_tokens]
    saveCorpus(sent_tokens)

def main():
    if (sys.stdin):
        processText(sys.stdin.read())

if __name__ == "__main__":
    main()
