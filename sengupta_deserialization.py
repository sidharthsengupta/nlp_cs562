#!/usr/bin/env python3

# Sidharth Sengupta
# CS 562 HW 1 Part 1
# Usage: ./sengupta_deserialization.py INPUT_GZIPPED_XML_FILES
#
# Extracts paragraph text from input gzipped xml files that have DOC tag
# with type attribute equal to 'story'. Prints all text to stdout.

import sys
import gzip
from lxml import etree

def processFiles(fileList):
    """
    Extracts paragraph text from Gzipped XML files. 
    Uses lxml etree for xml parsing to convert raw XML string into
    tree structure and XPath syntax to extract text from elements
    with paragraph tag 'P' from document elements with tag 'DOC' 
    that have type attribute equal to 'story'. Prints all text to
    stdout, separating text between files with newline.

    Args:
        fileList (list): list of Gzipped XML filename strings

    Returns:
        None
    """
    all_text = []
    for iFile in fileList:
        with gzip.open(iFile) as f:
            tree = etree.fromstring(f.read())
            text = tree.xpath("//DOC[@type='story']/TEXT/P/text()")
            text = [p for p in text if p]
            all_text = all_text + text
    print(*all_text, sep = '\n')

def main(): 
    # Checks if a command line arg was passed
    if len(sys.argv) > 1:
        processFiles(sys.argv[1:])
    else:
        print("Input files not specified.")

if __name__ == "__main__":
    main()
