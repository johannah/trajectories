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
    seeds_died = [p for p in seeds if pdict[p]['reward'] < 0]
    seeds_lost = np.array([p for p in seeds if pdict[p]['reward'] < 0])
    steps = np.array([len(pdict[seed]['actions']) for seed in seeds_won])
    seeds_tout = np.array([p for p in seeds if pdict[p]['reward'] == 0])

    z = zip(seeds_won, steps)
    steps_sorted = sorted(z, key=lambda x: x[1])
    tms = [pdict[seed]['full_end_time'] - pdict[seed]['full_start_time'] for seed in seeds_won]
    steps_won = [len(pdict[seed]['actions']) for seed in seeds_won]
    if not len(steps_won):
        steps_won = [10000.0]

    mean_times = np.mean(tms)
    print("PERCENT LOST  {} - {}/{}".format(round(len(seeds_lost)/float(len(seeds)),2),
                                                 len(seeds_lost),len(seeds)))

    print("PERCENT WON  {} - {}/{}".format(round(len(seeds_won)/float(len(seeds)),2),
                                                 len(seeds_won),len(seeds)))

    print("PERCENT TIED  {} - {}/{}".format(round(len(seeds_tout)/float(len(seeds)),2),
                                                 len(seeds_tout),len(seeds)))
    print("MEAN STEPS WON {}".format(round(np.mean(steps_won),2)))


    #return {'mean steps':np.mean(steps), 'median steps':np.median(steps),
    #        'max steps':np.max(steps), 'min_steps':np.min(steps),
    #        'var_steps':np.std(steps), 'num_tout':len(seeds_tout),
    #        'mean times':mean_times,  'var times':np.std(tms),
    #        'num_won':len(seeds_won),
    #        'num_died':len(seeds_died),
    #        'num_timeout':len(seeds_won)-len(seeds_lost),
    #         #'seed steps sorted':steps_sorted,
    #        'seeds_lost':seeds_lost}




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
    return {key:r}


files = sorted(glob('../../results/*mall*pkl'))
loaded = [(f,pickle.load(open(f,'r'))) for f in files ]
for (f,l) in loaded:
    print(f)
    get_steps_won(l)
    print(get_avg_time(l))
    #print('35 length', len(l[35]['actions']))
    #print(get_avg_reward(l))
    #print(get_num_games_won(l))
    #print(f,get_reward_won_games(l))
    print('------------------------------')

embed()
#for (f,l) in loaded:
#    print(f, get_avg_time(l))



