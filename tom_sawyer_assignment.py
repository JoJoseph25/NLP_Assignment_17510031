import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def msplit(string, sep):
    docs = [string]
    for char in sep:
        words = []
        for substr in docs:
            words.extend(substr.split(char))
        docs = words
    return docs

def inverse_dict(d):
    invr_dict={}
    for key in d:
        check = invr_dict.get(d[key],0)
        if check==0:
            invr_dict[d[key]] = [key]
        else:
            invr_dict[d[key]].append(key)
    return invr_dict

def histogram(word_list):
    # word_list=list(s.split())
    word_dict= {}
    for word in word_list:
        word_dict[word]= word_dict.get(word,0)+1
    return word_dict

def word_freq_descending(L):
    d1={}
    for word in L:
        d1[word] = d1.get(word,0) + 1
    inverse_d1= inverse_dict(d1)
    sorted_freq=list(inverse_d1.keys())
    sorted_freq.sort()
    sorted_freq=sorted_freq[::-1]
    Final_list=[]
    for freq in sorted_freq:
        repeat= freq
        for word in inverse_d1[freq]:
            for i in range(repeat):
                Final_list.append(word)
    word_dict=histogram(Final_list)
    return word_dict

def n_def(x, dictonary):
    count=0
    if len(x)==0:
        count=0
    else:
        for i in range(len(x)-1):
            re_dict=dictonary[x[i]:x[i+1]]
            for word in re_dict:
                if word.isdigit() or word== "Defn":
                    count+=1
                else:
                    continue
    return count

def search(word, dictonary):
    check=1
    i=0
    re_dict=dictonary[i:]
    poss_word=[]
    def_word=[]
    while check==1:
        re_dict=dictonary[i:]
        if len(re_dict)<=4:
            check=0
        if word in re_dict:
            ind=re_dict.index(word)
            poss_word.append(i+ind)
            if word == re_dict[ind] and (not(re_dict[ind-1].isupper()) and not(re_dict[ind+1].isupper())):
                def_word.append(i+ind)
                i+=ind+1
            else:
                i+=ind+1
        else:
            check=0
    if def_word!=[]:
        ind=poss_word.index(def_word[0])
        def_word.append(poss_word[ind+len(def_word)])

    out=n_def(def_word,dictonary)
    return out

def count_vocab(list_words):
    N = []
    i=0
    V=[]
    v = set()
    for word in words:
        i=i+1
        if word not in V:
            v.add(word)
        V.append(len(v))
        N.append(i)
        print("Computing Vocab "+ str(int(i/len(words)*100))+"%"+" complete")
    return N,V

file = open("Tom_Sawyer.txt", 'r')
doc=msplit(file.read(),(" ","\n",",","(",")","-","--","\"","[","]","!","?","_",".",";",":","”","“"))
file.close()
words=list(filter(lambda x: x != "", doc))
word_dict=word_freq_descending(words)

types=len(word_dict.keys())
token=len(words)
TTR=types/token

word_df=pd.DataFrame.from_dict(word_dict,orient="index")
word_df.columns=["Freq"]
word_df["Rank"]= [i for i in range(1,len(word_df)+1)]
word_df["f.r"]= word_df["Freq"]*word_df["Rank"]
word_df["prob_r"]= word_df["Freq"]/token
word_df["A"]=word_df["prob_r"]*word_df["Rank"]


word_len=[len(word) for word in word_dict.keys()]
word_df["word_len"]= word_len


# x=pd.Series(word_df["Rank"]).values
# y=pd.Series(word_df["Freq"]).values

# plt.plot(x,y, linewidth=2)
# plt.xlabel("Rank")
# plt.ylabel("Frequency")
# plt.title("Zipf Law to Word Rank and their corresponding Frequency")
# plt.show()
file = open("dictonary.txt", 'r')
doc_dict=msplit(file.read(),(" ","\n",",","(",")","-","--","\"","[","]","!","?","_",".",";",":","”","“"))
file.close()
dictonary=list(filter(lambda x: x != "", doc_dict))
word_df["meaning"]=word_df.index

print(word_df)






# N=np.array(N)
# V=np.array(V)

# N=N/10**3
# V=V/10**2

# def func(n,k,b):
#     return k*n**b

# popt, pcov = curve_fit(func, N, V, bounds=([0,0.0], [100,1.0]))

# k= popt[0]
# beta=popt[1]
# print("k= "+str(k)+" & beta= " +str(beta))

# plt.plot(N,V, label="Text Data")
# plt.plot(N, func(N, *popt), 'g--',label='Model')
# plt.xlabel("Words in Collection, N (Thousands)")
# plt.ylabel("Words in Vocabulary, V (Hundreds)")
# plt.title("Heap's Law")
# plt.legend()
# plt.show()
