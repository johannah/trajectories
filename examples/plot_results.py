import numpy as np
import matplotlib.pyplot as plt
import os, sys
from glob import glob
from IPython import embed
import pickle



def get_steps_won(pdict):
    key = 'reward'
    seeds = np.array([p for p in pdict.keys() if type(p) == int])
    seeds_won = np.array([p for p in seeds if pdict[p]['reward'] > 0])
    seeds_lost = np.array([p for p in seeds if pdict[p]['reward'] < 0])
    steps = [len(pdict[seed]['actions']) for seed in seeds_won]
    seeds_tout = np.array([p for p in seeds if pdict[p]['reward'] == 0])
    z = zip(seeds_won, steps)
    steps_sorted = sorted(z, key=lambda x: x[1])
    tms = [pdict[seed]['full_end_time'] - pdict[seed]['full_start_time'] for seed in seeds_won]

    return {'mean steps':np.mean(steps), 'median steps':np.median(steps),
            'max steps':np.max(steps), 'min_steps':np.min(steps),
            'var_steps':np.std(steps), 'num_tout':len(seeds_tout), 
            'mean times':np.mean(tms), 'var times':np.std(tms),
             #'seed steps sorted':steps_sorted,
            'seeds_lost':seeds_lost}




def get_num_games_won(pdict):
    key = 'reward'
    vals = np.array([pdict[p][key] for p in pdict.keys() if type(p) == int])
    r = vals>0
    #print(key, r)
    return {'total won':r.sum(), 'num played':r.shape[0], 'frac won':r.sum()/float(r.shape[0])}

def get_reward_won_games(pdict):
    key = 'reward'
    vals = np.array([pdict[p][key] for p in pdict.keys() if type(p) == int])
    r = vals>0
    return np.mean(vals[r])

def get_avg_reward(pdict):
    key = 'reward'
    vals = [pdict[p][key] for p in pdict.keys() if type(p) == int]
    mea = np.mean(vals)
    med = np.median(vals)
    std = np.std(vals)
    ma = np.max(vals)
    mi = np.min(vals)
    return {'reward mean':mea, 'reward median':med, 'reward std':std, 'reward max':ma, 'reward min':mi}

def get_avg_time(pdict):
    key = 'decision_ts'
    vals = [np.mean(pdict[p][key]) for p in pdict.keys() if type(p) == int]
    r = np.mean(vals)
    #print(key, r)
    return r


files = sorted(glob('all_results_model_*.pkl'))
loaded = [(f,pickle.load(open(f,'r'))) for f in files ]
for (f,l) in loaded:
    print(f)
    print(get_avg_reward(l))
    print(get_num_games_won(l))
    print(get_steps_won(l))
    #print(f,get_reward_won_games(l))
    print('------------------------------')

embed()
#for (f,l) in loaded:
#    print(f, get_avg_time(l))



