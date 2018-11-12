

# import standard libs
from IPython.display import display
from IPython.core.debugger import set_trace as bp
from pathlib import PurePath, Path
import sys
import time
from collections import OrderedDict as od
import re
import os
import json
os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'

# get project dir
pp = PurePath(Path.cwd()).parts[:-1]
pdir = PurePath(*pp)
script_dir = pdir / 'scripts' 
viz_dir = pdir / 'viz'
data_dir = pdir / 'data'
sys.path.append(script_dir.as_posix())

# import python scientific stack
import pandas as pd
import pandas_datareader.data as web
pd.set_option('display.max_rows', 100)
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from multiprocessing import cpu_count
pbar = ProgressBar()
pbar.register()
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from numba import jit
import math


# import visual tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

plt.style.use('seaborn-talk')
plt.style.use('bmh')
#plt.rcParams['font.family'] = 'DejaVu Sans Mono'
plt.rcParams['font.size'] = 9.5
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['figure.figsize'] = 10,7
blue, green, red, purple, gold, teal = sns.color_palette('colorblind', 6)

# import util libs
from tqdm import tqdm, tqdm_notebook
import warnings
warnings.filterwarnings("ignore")
import missingno as msno
#from utils import cprint
#from bars import *
import pymc3 as pm
from theano import shared, theano as tt
import ffn
RANDOM_STATE = 777

print()
#get_ipython().run_line_magic('watermark', '-p pandas,pandas_datareader,dask,numpy,pymc3,theano,sklearn,statsmodels,scipy,ffn,matplotlib,seaborn')


# In[174]:



from numba import jit
from tqdm import tqdm
import pandas as pd
import numpy as np

def cprint(df):
    if not isinstance(df, pd.DataFrame): df = df.to_frame()
        #try:
        #except: pass
		#raise ValueError('object cannot be coerced to df')
    print('-'*79)
    print('dataframe information')
    print('-'*79)
    print(df.tail(5))
    print('-'*50)
    print(df.info())
    print('-'*79)
    print()

get_range = lambda df, col: (df[col].min(), df[col].max())



#========================================================
def returns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))
#========================================================
def tick_bars(df, column, m):
    '''
    compute tick bars

    # args
        df: pd.DataFrame()
        column: name for price data
        m: int(), threshold value for ticks
    # returns
        idx: list of indices
    '''
    t = df[column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += 1 
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx

def tick_bar_df(df, column, m):
    idx = tick_bars(df, column, m)
    return df.iloc[idx]
#========================================================
def volume_bars(df, column, m):
    '''
    compute volume bars

    # args
        df: pd.DataFrame()
        column: name for price data
        m: int(), threshold value for volume
    # returns
        idx: list of indices
    '''
    t = df[column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx

def volume_bar_df(df, column, m):
    idx = volume_bars(df, column, m)
    return df.iloc[idx]
#========================================================
def dollar_bars(df, column, m):
    '''
    compute dollar bars

    # args
        df: pd.DataFrame()
        column: name for price data
        m: int(), threshold value for dollars
    # returns
        idx: list of indices
    '''
    t = df[column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx

def dollar_bar_df(df, column, m):
    idx = dollar_bars(df, column, m)
    return df.iloc[idx]
#========================================================

@jit(nopython=True)
def numba_isclose(a,b,rel_tol=1e-09,abs_tol=0.0):
    return np.fabs(a-b) <= np.fmax(rel_tol*np.fmax(np.fabs(a), np.fabs(b)), abs_tol)

@jit(nopython=True)
def bt(p0, p1, bs):
    #if math.isclose((p1 - p0), 0.0, abs_tol=0.001):
    if numba_isclose((p1-p0),0.0,abs_tol=0.001):
        b = bs[-1]
        return b
    else:
        b = np.abs(p1-p0)/(p1-p0)
        return b

@jit(nopython=True)
def get_imbalance(t):
    bs = np.zeros_like(t)
    for i in np.arange(1, bs.shape[0]):
        t_bt = bt(t[i-1], t[i], bs[:i-1])
        bs[i-1] = t_bt
    return bs[:-1] # remove last value


# ## Code Snippets
# 
# Below I reproduce all the relevant code snippets found in the book that are necessary to work through the excercises found at the end of chapter 3.

# ### Symmetric CUSUM Filter [2.5.2.1]

# In[175]:


def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna().abs()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = (sPos+diff.loc[i]).astype(float), (sNeg+diff.loc[i]).astype(float)
        except Exception as e:
            print(e)
            print(sPos+diff.loc[i], type(sPos+diff.loc[i]))
            print(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))
            break
        sPos, sNeg=max(0., pos.all()), min(0., neg.all())
        if sNeg<-h:
            sNeg=0;tEvents.append(i)
        elif sPos>h:
            sPos=0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


# ### Daily Volatility Estimator [3.1]

# In[176]:


def getDailyVol(close,span0=100):
    # daily vol reindexed to close
    print('close; ')
    print(close)
    print('close.index-pd.Timedelta; ')
    print(close.index-pd.Timedelta(days=1))
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    #print('\n df0 1\n', df0)
    df0=df0[df0>0]   
    #print('\n df0 2\n', df0)
    df0=(pd.Series(close.index[df0-1],index=close.index[close.shape[0]-df0.shape[0]:]))   
    #print('\n df0 3\n', df0)
    
    try:
        print('\n df0 index\n', df0.index)
        print('\n df0 values\n', df0.values)
        dfidxs=close.loc[df0.index].drop_duplicates()
        dfvals=close.loc[df0.values]#.iloc[:dfidxs.size] #.drop_duplicates()
        print('\n close.loc[df0.index]\n', dfidxs)
        print('\n close.loc[df0.values]\n', dfvals)
        #df1=(dfidxs.values/dfvals.values) -1 # daily rets
        df0=(dfidxs/dfvals.values) -1 # daily rets
        print('\n df1\n',df1)
    except Exception as e:
        print(e)
        print('adjusting shape of close.loc[df0.index]')
        cut = close.loc[df0.index].shape[0] - close.loc[df0.values].shape[0]
        df0=close.loc[df0.index].iloc[:-cut]/close.loc[df0.values].values-1
        
    #df4=pd.Series(df1).ewm(span=span0).std().dropna()
    #df0=pd.Series(df1).ewm(span=span0).std().dropna()
    df0=(df0).ewm(span=span0).std().dropna()

    #print('\n df4\n', df4)
    #df2=(pd.Series(df4.values,index=close.index[close.shape[0]-df4.shape[0]:]))
    #print('\n df2\n', df2)
    #return df2
    return df0


# ### Triple-Barrier Labeling Method [3.2]

# In[177]:


def applyPtSlOnT1(close,events,ptSl,molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_=events.loc[molecule]
    print('in applyPtSlOnT1',close,'events',events,'ptSl',ptSl,'molecule',molecule)
    out=events_[['t1']].copy(deep=True)
    print('out',out)
    if ptSl[0]>0: pt=ptSl[0]*events_['trgt']
    else: pt=pd.Series(index=events.index) # NaNs
    if ptSl[1]>0: sl=-ptSl[1]*events_['trgt']
    else: sl=pd.Series(index=events.index) # NaNs
    i=0
    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        try:
            df0=close[loc:t1] # path prices
            #print('\niter ', i, 'df0 \n',df0)
            df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
            out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss
            out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking
            #print('out \n',out)
            i=i+1
        except Exception as e:
            pass#print(e)
    return out


# ### Gettting Time of First Touch (getEvents) [3.3], [3.6]

# In[171]:


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    trgt=trgt.dropna()
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False:t1=pd.Series(pd.NaT, index=tEvents.drop_duplicates())
    #3) form events object, apply stop loss on t1
    if side is None:side_,ptSl_=pd.Series(1.,index=trgt.index), [ptSl[0],ptSl[0]]
    else: side_,ptSl_=side.loc[trgt.index],ptSl[:2]
    #trgt=trgt.iloc[trgt.size-t1.size:]
    #side_=side_.iloc[side_.size-t1.size:]
    print('\nin getEvents','trgt\n', trgt,'t1\n', t1,'side\n', side_)
    events=(pd.concat({'trgt':trgt,'side':side_}, axis=1,ignore_index=True).dropna().drop_duplicates())
    #events=pd.concat({'t1':t1,'side':events.columns[0],'trgt':events.columns[1]},ignore_index=True)
    events=events.merge(pd.DataFrame(t1),right_index=True,left_index=True).dropna().drop_duplicates()
    
    events=events.rename(columns={ events.columns[0]: "side",events.columns[1]: "trgt",events.columns[2]: "t1" })
    print('\nin getEvents','close\n', close,'events\n', events,'ptsl\n', ptSl_,'molecule\n' ,events.index)
    
    #df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),numThreads=numThreads,close=close,events=events,ptSl=ptSl_)
    df0=applyPtSlOnT1(close, events, ptSl_, events.index)
    print('df0 after applyPtSl',df0)
    events['t1']=df0.dropna()#how='all').min(axis=1) # pd.min ignores nan
    if side is None:events=events.drop('side',axis=1)
    return events
  
# ### Adding Vertical Barrier [3.4]

# In[135]:

def addVerticalBarrier(tEvents, close, numDays=1):
    t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
    return t1


# ### Labeling for side and size [3.5]

# In[136]:


def getBinsOld(events,close):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin']=np.sign(out['ret'])
    # where out index and t1 (vertical barrier) intersect label 0
    try:
        locs = out.query('index in @t1').index
        out.loc[locs, 'bin'] = 0
    except: pass
    return out


# ### Expanding getBins to Incorporate Meta-Labeling [3.7]

# In[137]:


def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    print('\n events_ \n', events_)
    #px=pd.DataFrame(px)
    
    print('\n px \n', px)
    try:
        px=close.reindex(px,method='bfill')
    except Exception as e:
        print(e,'\n trying join')
        px=px.to_frame().join(close, how='inner').drop_duplicates();
        
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    
    print('\n out \n', out)
    print('\n px \n', px)
    print('\n events_[t1]\n', events_['t1'])
    et1=events_['t1']
    px_et1=px.loc[et1].drop_duplicates()
    px_eIndex=px.loc[events_.index].drop_duplicates()
    out['ret']=(px_et1['price'].values/px_eIndex[:px_et1.shape[0]]['price'])-1
    
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    out=out.dropna().drop_duplicates()
    print('\n out \n', out)
    return out


# ### Dropping Unnecessary Labels [3.8]

# In[138]:


def dropLabels(events, minPct=.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0=events['bin'].value_counts(normalize=True)
        if df0.min()>minPct or df0.shape[0]<3:break
        print('dropped label: ', df0.argmin(),df0.min())
        events=events[events['bin']!=df0.argmin()]
    return events


# ### Linear Partitions [20.4.1]

# In[139]:


def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts


# In[140]:


def nestedParts(numAtoms,numThreads,upperTriang=False):
    # partition of atoms with an inner loop
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts


# ### multiprocessing snippet [20.7]

# In[141]:


def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func
    
    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''
    import pandas as pd
    #if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    #else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)
    
    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else: out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out:df0=df0.append(i)
    df0=df0.sort_index()
    return df0


# ### single-thread execution for debugging [20.8]

# In[142]:


def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out


# ### Example of async call to multiprocessing lib [20.9]

# In[143]:


import multiprocessing as mp
import datetime as dt

#________________________________
def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+         str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return
#________________________________
def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asyn output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out


# ### Unwrapping the Callback [20.10]

# In[144]:


def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out


# ### Pickle Unpickling Objects [20.11]

# In[145]:


def _pickle_method(method):
    func_name=method.im_func.__name__
    obj=method.im_self
    cls=method.im_class
    return _unpickle_method, (func_name,obj,cls)
#________________________________
def _unpickle_method(func_name,obj,cls):
    for cls in cls.mro():
        try:func=cls.__dict__[func_name]
        except KeyError:pass
        else:break
    return func.__get__(obj,cls)
#________________________________
import copyreg,types, multiprocessing as mp
copyreg.pickle(types.MethodType,_pickle_method,_unpickle_method)


# # Exercises

# ## Import Dataset
# 
# Note this dataset below has been resampled to `1s` and then `NaNs` removed. This was done to remove any duplicate indices not accounted for in a simple call to `pd.DataFrame.drop_duplicates()`. 

# In[146]:

path = os.getcwd()
df = pd.read_csv(path+'/bitstampUSD_21.csv', index_col=0)
cprint(df)


# ## [3.1] Form Dollar Bars

# In[18]:


dbars = dollar_bar_df(df, 'dv', 100_000).drop_duplicates().dropna()
cprint(dbars)


# ## [3.1] Form Dollar Bars

# In[178]:

dbars.index = pd.to_datetime(dbars.index)
dbars=dbars.drop_duplicates().dropna()
cprint(dbars)


# ### (a) Run cusum filter with threshold equal to std dev of daily returns

# In[148]:


close = dbars.price.copy()
dailyVol = getDailyVol(close)
#dailyVol = dailyVol.dropna()
print(dailyVol)


# In[149]:


f,ax=plt.subplots()
dailyVol.plot(ax=ax)
ax.axhline(dailyVol.mean(),ls='--',color=red)


# In[156]:


tEvents = getTEvents(close,h=dailyVol.mean())
print(tEvents)


# ### (b) Add vertical barrier

# In[157]:


t1 = addVerticalBarrier(tEvents, close)
print(t1)


# ### (c) Apply triple-barrier method where `ptSl = [1,1]` and `t1` is the series created in `1.b`

# In[172]:


# create target series
ptsl = [1,1]
target=dailyVol
# select minRet
minRet = 0.01
# get cpu count - 1
cpus = cpu_count() - 1
events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1)
events = events.dropna()

# In[170]:
print(events)

# ### (d) Apply `getBins` to generate labels

# In[109]:


labels = getBins(events, close)
print(labels)


# In[110]:


labels.bin.value_counts()


# ## [3.2] Use snippet 3.8 to drop under-populated labels

# In[113]:


clean_labels = dropLabels(labels)
cprint(clean_labels)


# In[114]:


clean_labels.bin.value_counts()


# ## [3.4] Develop moving average crossover strategy. For each obs. the model suggests a side but not size of the bet

# In[259]:


fast_window = 3
slow_window = 7

close_df = (pd.DataFrame()
            .assign(price=close)
            .assign(fast=close.ewm(fast_window).mean())
            .assign(slow=close.ewm(slow_window).mean()))
cprint(close_df)


# In[260]:


def get_up_cross(df):
    crit1 = df.fast.shift(1) < df.slow
    crit2 = df.fast > df.slow
    return df.fast[(crit1) & (crit2)]

def get_down_cross(df):
    crit1 = df.fast.shift(1) > df.slow
    crit2 = df.fast < df.slow
    return df.fast[(crit1) & (crit2)]

up = get_up_cross(close_df)
down = get_down_cross(close_df)

f, ax = plt.subplots(figsize=(11,8))

close_df.loc['2014':].plot(ax=ax, alpha=.5)
up.loc['2014':].plot(ax=ax,ls='',marker='^', markersize=7,
                     alpha=0.75, label='upcross', color='g')
down.loc['2014':].plot(ax=ax,ls='',marker='v', markersize=7, 
                       alpha=0.75, label='downcross', color='r')

ax.legend()


# ### (a) Derive meta-labels for `ptSl = [1,2]` and `t1` where `numdays=1`. Use as `trgt` dailyVol computed by snippet 3.1 (get events with sides)

# In[261]:


side_up = pd.Series(1, index=up.index)
side_down = pd.Series(-1, index=down.index)
side = pd.concat([side_up,side_down]).sort_index()
cprint(side)


# In[267]:


minRet = .01 
ptsl=[1,2]
ma_events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1,side=side)
cprint(ma_events)


# In[224]:


ma_events.side.value_counts()


# In[268]:


ma_side = ma_events.dropna().side


# In[269]:


ma_bins = getBins(ma_events,close).dropna()
cprint(ma_bins)


# In[265]:


Xx = pd.merge_asof(ma_bins, side.to_frame().rename(columns={0:'side'}),
                   left_index=True, right_index=True, direction='forward')
cprint(Xx)


# ### (b) Train Random Forest to decide whether to trade or not `{0,1}` since underlying model (crossing m.a.) has decided the side, `{-1,1}`

# In[227]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report


# In[270]:


X = ma_side.values.reshape(-1,1)
#X = Xx.side.values.reshape(-1,1)
y = ma_bins.bin.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

n_estimator = 10000
rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator,
                            criterion='entropy', random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print(classification_report(y_test, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# ## [3.5] Develop mean-reverting Bollinger Band Strategy. For each obs. model suggests a side but not size of the bet.

# In[230]:


def bbands(price, window=None, width=None, numsd=None):
    """ returns average, upper band, and lower band"""
    ave = price.rolling(window).mean()
    sd = price.rolling(window).std(ddof=0)
    if width:
        upband = ave * (1+width)
        dnband = ave * (1-width)
        return price, np.round(ave,3), np.round(upband,3), np.round(dnband,3)        
    if numsd:
        upband = ave + (sd*numsd)
        dnband = ave - (sd*numsd)
        return price, np.round(ave,3), np.round(upband,3), np.round(dnband,3)


# In[231]:


window=50
bb_df = pd.DataFrame()
bb_df['price'],bb_df['ave'],bb_df['upper'],bb_df['lower']=bbands(close, window=window, numsd=1)
bb_df.dropna(inplace=True)
cprint(bb_df)


# In[232]:


f,ax=plt.subplots(figsize=(11,8))
bb_df.loc['2014'].plot(ax=ax)


# In[233]:


def get_up_cross(df, col):
    # col is price column
    crit1 = df[col].shift(1) < df.upper  
    crit2 = df[col] > df.upper
    return df[col][(crit1) & (crit2)]

def get_down_cross(df, col):
    # col is price column    
    crit1 = df[col].shift(1) > df.lower 
    crit2 = df[col] < df.lower
    return df[col][(crit1) & (crit2)]

bb_down = get_down_cross(bb_df, 'price')
bb_up = get_up_cross(bb_df, 'price') 

f, ax = plt.subplots(figsize=(11,8))

bb_df.loc['2014':].plot(ax=ax, alpha=.5)
bb_up.loc['2014':].plot(ax=ax, ls='', marker='^', markersize=7,
                        alpha=0.75, label='upcross', color='g')
bb_down.loc['2014':].plot(ax=ax, ls='', marker='v', markersize=7, 
                          alpha=0.75, label='downcross', color='r')
ax.legend()


# ### (a) Derive meta-labels for `ptSl=[0,2]` and `t1` where `numdays=1`. Use as `trgt` dailyVol.

# In[300]:


bb_side_up = pd.Series(-1, index=bb_up.index) # sell on up cross for mean reversion
bb_side_down = pd.Series(1, index=bb_down.index) # buy on down cross for mean reversion
bb_side_raw = pd.concat([bb_side_up,bb_side_down]).sort_index()
cprint(bb_side_raw)

minRet = .01 
ptsl=[0,2]
bb_events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1,side=bb_side_raw)
cprint(bb_events)

bb_side = bb_events.dropna().side
cprint(bb_side)


# In[290]:


bb_side.value_counts()


# In[301]:


bb_bins = getBins(bb_events,close).dropna()
cprint(bb_bins)


# In[292]:


bb_bins.bin.value_counts()


# ### (b) train random forest to decide to trade or not. Use features: volatility, serial correlation, and the crossing moving averages from exercise 2.

# In[293]:


def returns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))

def df_rolling_autocorr(df, window, lag=1):
    """Compute rolling column-wise autocorrelation for a DataFrame."""

    return (df.rolling(window=window)
            .corr(df.shift(lag))) # could .dropna() here

#df_rolling_autocorr(d1, window=21).dropna().head()


# In[294]:


srl_corr = df_rolling_autocorr(returns(close), window=window).rename('srl_corr')
cprint(srl_corr)


# In[302]:


features = (pd.DataFrame()
            .assign(vol=bb_events.trgt)
            .assign(ma_side=ma_side)
            .assign(srl_corr=srl_corr)
            .drop_duplicates()
            .dropna())
cprint(features)


# In[303]:


Xy = (pd.merge_asof(features, bb_bins[['bin']], 
                    left_index=True, right_index=True, 
                    direction='forward').dropna())
cprint(Xy)


# In[297]:


Xy.bin.value_counts()


# In[305]:


X = Xy.drop('bin',axis=1).values
y = Xy['bin'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

n_estimator = 10000
rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator,
                            criterion='entropy', random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print(classification_report(y_test, y_pred, target_names=['no_trade','trade']))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# ### (c) What is accuracy of predictions from primary model if the secondary model does not filter bets? What is classification report?

# In[299]:


minRet = .01 
ptsl=[0,2]
bb_events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1)
cprint(bb_events)

bb_bins = getBins(bb_events,close).dropna()
cprint(bb_bins)

features = (pd.DataFrame()
            .assign(vol=bb_events.trgt)
            .assign(ma_side=ma_side)
            .assign(srl_corr=srl_corr)
            .drop_duplicates()
            .dropna())
cprint(features)

Xy = (pd.merge_asof(features, bb_bins[['bin']], 
                    left_index=True, right_index=True, 
                    direction='forward').dropna())
cprint(Xy)

### run model ###
X = Xy.drop('bin',axis=1).values
y = Xy['bin'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

n_estimator = 10000
rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator,
                            criterion='entropy', random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print(classification_report(y_test, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[ ]:
