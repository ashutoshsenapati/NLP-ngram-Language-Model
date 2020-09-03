import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator

from collections import defaultdict,Counter

# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    text=[i.lower() for i in text]
    text.append('</s>')
    for i in range(len(text)):
        context = []
        j=n-1
        while j>0:
            if i-j <0:
                context.append('<s>')
            else:
                context.append(text[i-j])
            j-=1
        yield (text[i],tuple(context))

# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:
    with open(corpus_path) as file:
        data = file.read()
        sentences = data.split("\n")
        res=[]
        for t in sentences:
            res.extend(sent_tokenize(t))
        sentences=res
        sentences  = [word_tokenize(t) for t in sentences]
        return sentences



# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    # pass
    data = load_corpus(corpus_path)
    lm = NGramLM(n)
    for i in data:
        lm.update(i)
    return lm


# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        # pass
        ngrams = get_ngrams(self.n, text)
        for i in ngrams:
            self.vocabulary.add(i[0])
            self.ngram_counts[i]+=1
            self.context_counts[i[1]]+=1
        

    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta= .0) -> float:
        # pass
        # print(self.ngram_counts[(word,context)]/self.context_counts[context] if self.context_counts[context]!=0 else 1/len(self.vocabulary))
        # print(word,context)
        cvw = self.ngram_counts[(word,context)]
        cv = self.context_counts[context]
        voc = len(self.vocabulary)
        if not delta:
            return cvw/cv if cv!=0 else 1/voc
        return (cvw+delta)/ (cv+ delta*voc) if cv!=0 else 1/voc
        


    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        # pass
        # sent = sent.lower()
        ngrams = get_ngrams(self.n,sent)
        res = 0
        for i in ngrams:
            if i[0] not in self.vocabulary:
                return float("-inf")
            res+=math.log2(self.get_ngram_prob(i[0],i[1])) if self.get_ngram_prob(i[0],i[1])!=0 else 0
        return res

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]]) -> float:
        res = 0
        for sentence in corpus:
            res+=self.get_sent_log_prob(sentence)
        res/=len(self.vocabulary)
        res = math.pow(2,-res)
        return res
        
        

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        sorted_vocab = sorted(self.vocabulary)
        rand = random.random()
        curr = 0
        for word in sorted_vocab:
            curr+=self.get_ngram_prob(word,context,delta)
            if curr>rand:
                return word.strip()
        pass

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        curr = []
        if max_length==0:
            return ""
        while len(curr)==0 or (len(curr)<max_length and curr[-1] !='</s>'):
            if len(curr)<(self.n-1):
                context = tuple(['<s>']*(self.n-1 - len(curr)) + curr)
            else:
                context = tuple(curr[-(self.n-1):])
            curr.append(self.generate_random_word(context,delta))
        return " ".join(curr)



def main(corpus_path: str, delta: float, seed: int):
    c1=create_ngram_lm(1,"shakespeare.txt")
    c3=create_ngram_lm(3,"shakespeare.txt")
    c5=create_ngram_lm(5,"shakespeare.txt")

    rand = [random.randint(1,10000) for i in range(5)]
    i=1
    for ran in rand:
        random.seed(ran)
        print(i, c1.generate_random_text(10,0))
        i+=1
    print()
    i=1
    for ran in rand:
        random.seed(ran)
        print(i, c3.generate_random_text(10,0))
        i+=1
    print()
    i=1
    for ran in rand:
        random.seed(ran)
        print(i, c5.generate_random_text(10,0))
        i+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    parser.add_argument('corpus_path', nargs="?", type=str, default='/Users/ashutoshsenapati/Study/NLP/hw1/hw/shakespeare.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=0, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=123456, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
