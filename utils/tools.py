import pickle
import numpy as np 
import pandas as pd 
from utils.fit import *


def get_fit_param(model, fit_sub_info, method='mle', poi=None):
    if poi is None: poi = eval(model).p_names
    sub_lst = list(fit_sub_info.keys())
    if 'group' in sub_lst: sub_lst.pop(sub_lst.index('group'))
    params = {p: [] for p in poi}
    params['sub_id'] = []
    for sub_id in sub_lst:
        params['sub_id'].append(sub_id)
        for p, fn in zip(poi, eval(model).p_trans):
            idx = fit_sub_info[sub_id]['param_name'].index(p) 
            pvalue = fit_sub_info[sub_id]['param'][idx].copy()
            params[p].append(fn(pvalue))
    return pd.DataFrame.from_dict(params)

def get_llh_score(models, method, fit_sub_info,
                  use_bic=False,
                  relative=True):
    '''Get likelihood socres

    Inputs:
        models: a list of models for evaluation
    
    Outputs:
        crs: nll, aic and bic score per model per particiant
        pxp: pxp score per model per particiant
    '''
    tar = models[0] 
    fit_sub_info = []
    for i, m in enumerate(models):
        # get the subject list 
        with open(f'fits/fit_sub_info-{m}-{method}.pkl', 'rb')as handle:
            fit_info = pickle.load(handle)
        if i==0: subj_lst = fit_info.keys() 
        # get log post
        log_post = [fit_info[idx]['log_post'] for idx in subj_lst]
        bic      = [fit_info[idx]['bic'] for idx in subj_lst]
        h        = [fit_info[idx]['H'] for idx in subj_lst] if use_bic==False else 0
        n_param  = fit_info[list(subj_lst)[0]]['n_param']
        fit_sub_info.append({
            'log_post': log_post, 
            'bic': bic, 
            'n_param': n_param, 
            'H': h,
        })
    # get bms 
    bms_results = fit_bms(fit_sub_info, use_bic=use_bic)

    ## combine into a dataframe 
    cols = ['NLL', 'AIC', 'BIC', 'model', 'sub_id']
    crs = {k: [] for k in cols}
    for m in models:
        with open(f'fits/fit_sub_info-{m}-{method}.pkl', 'rb')as handle:
            fit_info = pickle.load(handle)
        # get the subject list 
        if i==0: subj_lst = fit_info.keys() 
        # get log post
        nll = [-fit_info[idx]['log_like'] for idx in subj_lst]
        aic = [fit_info[idx]['aic'] for idx in subj_lst]
        bic = [fit_info[idx]['bic'] for idx in subj_lst]
        crs['NLL'] += nll
        crs['AIC'] += aic
        crs['BIC'] += bic
        crs['model'] += [m]*len(nll)
        crs['sub_id'] += list(subj_lst)
    crs = pd.DataFrame.from_dict(crs)
    for c in ['NLL', 'BIC', 'AIC']:
        tar_crs = len(models)*list(crs.query(f'model=="{tar}"')[c].values)
        subtrack = tar_crs if relative else 0
        crs[c] -= subtrack
    pxp = pd.DataFrame.from_dict({'pxp': bms_results['pxp'], 'model': models})
    return crs, pxp 
