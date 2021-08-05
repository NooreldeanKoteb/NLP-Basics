#!/usr/bin/env python
# coding: utf-8



from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.util import ngrams
from scipy.spatial.distance import cdist
import numpy as np
import os
import io
import re


################
# Part 1 A & B #
################
words = ['free', 'market', 'language', 'shortage', 'reagan', 'president', 'administration', 'carter',
         'profit', 'net', 'loss', 'language', 'endangered', 'of', ('the', 'language'), 'accord', 
         ('the', 'paris')]


with open('reuters.train.txt', "r", encoding="utf8") as f:
    lines = f.readlines()

lines  = ' '.join(lines)
lines = ' '.join(sent_tokenize(lines))



unigram = word_tokenize(lines.lower())
bigram = ngrams(unigram, 2)
trigram = ngrams(unigram, 3)

N = len(unigram)

unigram_freq = dict(Counter(unigram))
bigram_freq = dict(Counter(bigram))
trigram_freq = dict(Counter(trigram))
V = len(unigram_freq)

def word_prob(w0, w1=None, w2=None, n_gram=1, k=0):
        if n_gram == 1:
            try:
                if k > 0:
                    print('p('+str(w0)+'): {:.15f}'.format((unigram_freq[w0]+k)/(N+(k*V))))
                else:
                    print('p('+str(w0)+'): {:.15f}'.format(unigram_freq[w0]/N))
            except:
                if k > 0:
                    print('p('+str(w0)+'): {:.15f}'.format(k/(N+ (k*V))))
                else:
                    print('p('+str(w0)+'): {:.15f}'.format(0/N))
        
        elif n_gram == 2:
            try:
                if k > 0:
                    print('p('+str(w1)+' | '+str(w0)+'): {:.15f}'.format((bigram_freq[(w0, w1)]+k)/(unigram_freq[w0]+(k*V))))
                else:
                    print('p('+str(w1)+' | '+str(w0)+'): {:.15f}'.format(bigram_freq[(w0, w1)]/unigram_freq[w0]))
            except:
                if k > 0:
                    print('p('+str(w1)+' | '+str(w0)+'): {:.15f}'.format(k/(unigram_freq[w0]+(k*V))))
                else:
                    print('p('+str(w1)+' | '+str(w0)+'): {:.15f}'.format(0/unigram_freq[w0]))
                    
        elif n_gram == 3:
            try:
                if k > 0:
                    print('p('+str(w2)+' | '+str(w0)+' '+str(w1)+'): {:.15f}'.format((trigram_freq[(w0, w1, w2)]+k)/(
                                                                                bigram_freq[(w0, w1)]+(k*V))))
                else:
                    print('p('+str(w2)+' | '+str(w0)+' '+str(w1)+'): {:.15f}'.format(trigram_freq[(w0, w1, w2)]/
                                                                                bigram_freq[(w0, w1)]))
            except:
                if k > 0:
                    print('p('+str(w2)+' | '+str(w0)+' '+str(w1)+'): {:.15f}'.format(k/(bigram_freq[(w0, w1)]+(k*V))))
                else:
                    print('p('+str(w2)+' | '+str(w0)+' '+str(w1)+'): {:.15f}'.format(0/bigram_freq[(w0, w1)]))


print('Unigram:')
word_prob('free', n_gram=1)
word_prob('market', n_gram=1)
word_prob('language', n_gram=1)

print('\nBigram:')
word_prob('free', 'market', n_gram=2)
word_prob('market', 'shortage', n_gram=2)
word_prob('president', 'reagan', n_gram=2)
word_prob('carter', 'administration', n_gram=2)
word_prob('net', 'profit', n_gram=2)
word_prob('net', 'loss', n_gram=2)
word_prob('programming', 'language', n_gram=2)
word_prob('endangered', 'language', n_gram=2)

print('\nTrigram:')
word_prob('the', 'language', 'of', n_gram=3)
word_prob('the', 'paris', 'accord', n_gram=3)


print('\n\nadd-1 smoothing applied:\n')
print('Unigram:')
word_prob('free', n_gram=1, k=1)
word_prob('market', n_gram=1, k=1)
word_prob('language', n_gram=1, k=1)

print('\nBigram:')
word_prob('free', 'market', n_gram=2, k=1)
word_prob('market', 'shortage', n_gram=2, k=1)
word_prob('president', 'reagan', n_gram=2, k=1)
word_prob('carter', 'administration', n_gram=2, k=1)
word_prob('net', 'profit', n_gram=2, k=1)
word_prob('net', 'loss', n_gram=2, k=1)
word_prob('programming', 'language', n_gram=2, k=1)
word_prob('endangered', 'language', n_gram=2, k=1)

print('\nTrigram:')
word_prob('the', 'language', 'of', n_gram=3, k=1)
word_prob('the', 'paris', 'accord', n_gram=3, k=1)
    
        


############
# Part 1 C #
############
with open('reuters.train.txt', "r", encoding="utf8") as f:
    lines = f.readlines()

lines  = ' '.join(lines).lower()
lines = re.sub("\d", "", lines)

# unique words = []

# words = dict(Counter(lines.split()))
# words = dict(sorted(words.items(),key=lambda x:x[1]))
# key_list = list(words.keys())
# val_list = list(words.values())

lines = sent_tokenize(lines)



def start_stop(sentances, n_gram):
    new_sentances = []
    for i in sentances:
        
        start = ''
        for n in range(n_gram-1):
            start +='<s> '
        
        mid = re.sub("\.(?!\d)", "", i)
        
        new_sentances.append(start+mid+' </s>')
        
    lines = ' '.join(new_sentances).split()

    return lines




unigram = ngrams(start_stop(lines,1), 1)
bigram = ngrams(start_stop(lines,2), 2)
trigram = ngrams(start_stop(lines,3), 3)
quadgram = ngrams(start_stop(lines,4), 4)
quintgram = ngrams(start_stop(lines,5), 5)
hexgram = ngrams(start_stop(lines,6), 6)

unigram_freq = dict(Counter(unigram))
bigram_freq = dict(Counter(bigram))
trigram_freq = dict(Counter(trigram))
quadgram_freq = dict(Counter(quadgram))
quintgram_freq = dict(Counter(quintgram))
hexgram_freq = dict(Counter(hexgram))

N = 0
for i in unigram_freq.keys():
    N += unigram_freq[i]

V = len(unigram_freq)

with open('reuters.test.txt', "r", encoding="utf8") as f:
    test = f.readlines()

test  = ' '.join(test).lower()
test = sent_tokenize(test)


test_unigram = list(ngrams(start_stop(test,1), 1))
test_bigram = list(ngrams(start_stop(test,2), 2))
test_trigram = list(ngrams(start_stop(test,3), 3))
test_quadgram = list(ngrams(start_stop(test,4), 4))
test_quintgram = list(ngrams(start_stop(test,5), 5))
test_hexgram = list(ngrams(start_stop(test,6), 6))



def word_prob2(words, n_gram, n_freq, uni_freq, k=1):
    if n_gram == 1:
        try:
            if k > 0:
                return ((uni_freq[i]+k)/(N+(k*V)))
        except:
            return (k/(N+(k*V)))
        
    else:
        if n_gram == 2:
            sec_freq = unigram_freq
        elif n_gram == 3:
            sec_freq = bigram_freq
        elif n_gram == 4:
            sec_freq = trigram_freq
        elif n_gram == 5:
            sec_freq = quadgram_freq
        elif n_gram == 6:
            sec_freq = quintgram_freq
            
        try:
            if k > 0:
                return ((n_freq[i]+k)/(sec_freq[i[:-1]]+(k*V)))
        except:
            try:
                return (k/(sec_freq[i[:-1]]+(k*V)))
            except:
                return (k/(N+(k*V)))
    return prob


def perplexity(test_set, n_freq, uni_freq, n_gram):
    sent_prob = 0
    sent_len = len(test_set)
    for i in test_set:
        if '</s>' in i:
                continue
        else:
            sent_prob += np.log2(word_prob2(i, n_gram=n_gram, n_freq=n_freq, uni_freq=uni_freq, k=1))


    avg_prob = sent_prob/sent_len
    return np.power(2, -avg_prob)
        



print('n-gram 1: '+str(perplexity(test_unigram, unigram_freq, unigram_freq, n_gram=1)))
print('n-gram 2: '+str(perplexity(test_bigram, bigram_freq, unigram_freq, n_gram=2)))
print('n-gram 3: '+str(perplexity(test_trigram, trigram_freq, unigram_freq, n_gram=3)))
print('n-gram 4: '+str(perplexity(test_quadgram, quadgram_freq, unigram_freq, n_gram=4)))
print('n-gram 5: '+str(perplexity(test_quintgram, quintgram_freq, unigram_freq, n_gram=5)))
print('n-gram 6: '+str(perplexity(test_hexgram, hexgram_freq, unigram_freq, n_gram=6)))



###########
#Part 2-1 #
###########
#Piazza Code from Samuel Blouir
glove_file = "glove\glove.6B.300d.txt"
wiki_file = "wiki-news-300d-1M-subword.vec"
analogy_file = 'analogy.txt'
new_dir = "processed\wiki"


def faster_loading(path):
    curr = open(path, 'r').read().split(" ")
    name, curr = curr[0], curr[1:]
    curr = np.array(list(map(float, curr)), dtype=np.float)
    return curr


def fetch_file(name, in_file=wiki_file, out_dir='wiki'):
    new_dir = f"processed/{out_dir}"
    target_file = f"{new_dir}/{name}.vec"
    if not os.path.exists(target_file):
        os.system(f"mkdir {new_dir} > nul 2> nul") 
        print(f"\tFetching {name}...")        
        os.system(f"findstr /B /S /C:\"{name} \" {in_file} > {target_file}")
    return faster_loading(target_file)



def read_from_disc(name, use_glove=False):
    if use_glove:
        return [fetch_file(name, in_file=glove_file, out_dir='glove')]
    else:
        return [fetch_file(name, in_file=wiki_file, out_dir='wiki')]


def part_one(glove=False):
    words = ["horseradish", "spinach", "lingonberry", "strawberries", "pikachu", "charizard", "charmander", "math", "algorithm"]
    word_vectors_dict = {each: read_from_disc(each, glove) for each in words}
    print('done')
    return word_vectors_dict

def part_two(glove=False):
    analogies = open(analogy_file, 'r').read().split()
    word_vectors_dict = {each: read_from_disc(each, glove) for each in analogies}
    print('done')
    return word_vectors_dict



dist_vecs = part_one()


# In[4]:


def cos_sim(word1, word2, vecs, verbose=True):
    dist = 0
    try:
        dist = cdist(vecs[word1], vecs[word2], metric='cosine')[0][0]
    except:
        print(word1)
        print(word2)
    
    if verbose:
        print(word1+' | '+word2+' --> {:.5f}'.format(dist))
    else:
        return dist



print('Cosine Similarity:')
cos_sim('horseradish','spinach', dist_vecs)
cos_sim('lingonberry','strawberries', dist_vecs)
cos_sim('pikachu','charizard', dist_vecs)
cos_sim('charizard','charmander', dist_vecs)
cos_sim('math','algorithm', dist_vecs)



###########
#Part 2-2 #
###########

analogy_vecs = part_two(True)



def cos_sim_compare(words, vecs, verbose=True):
    dist = 0

    try:
        new_word = (np.array(vecs[words[1]]) - np.array(vecs[words[0]])) + np.array(vecs[words[2]])
        dist = cdist(new_word, np.array(vecs[words[3]]), metric='cosine')[0][0]
    except:
        print('Warning: Vector not found')
    
    if verbose:
        print('Distance from words to '+words[3]+' {:.5f}'.format(dist))
    else:
        return dist




with open("analogy.txt") as f:
    results = []
    for line in f:
        words = line.split()
        if 'thirstiness' in words:
            continue
        results.append([cos_sim_compare(words, analogy_vecs, False), words[3]])



results = np.array(results)
values = results[:,0].astype(np.float)
print('Max: '+str(max(values)))
print('Min: '+str(min(values)))
print('Mean: '+str(np.mean(values)))
print('Median: '+str(np.median(values)))
print('STD: '+str(np.std(values)))
top, bot = np.percentile(values, [79, 51])
print('Range of possible Wrong: '+str([top, bot]))
print('Range of possible Wrong: '+str(top-bot))
import matplotlib.pyplot as plt
plt.plot(range(0,994), values)
plt.show()



file = open('analogy-predictions.txt','w')
wrong = 0
with open("analogy.txt") as f:
    for line in f:
        words = line.split()
        
        if 'thirstiness' in words:
            file.write('Correct\n')
            continue

        if cos_sim_compare(words, analogy_vecs, False) <= .89:
            file.write('Correct\n')
        else:
            wrong +=1
            file.write('Wrong\n')

