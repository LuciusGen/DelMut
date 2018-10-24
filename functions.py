import pandas as pd
import numpy as np
import os
import shutil as sh
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


def make_dir_if_not_exists(path, dir_name):
    '''
    This function creates a directory if there is no with such name
    
    :param path: a parent directory
    :param dir_name: a name of directory which is wanted to create
    :return: a path to created or already existed directory
    '''
    
    path = path if path[-1] == '/' else path + '/'
    
    if not dir_name in os.listdir(path):
        os.mkdir(path + dir_name)
    
    dpath = path + dir_name + '/'
    
    return dpath


def get_data(path, labels=True):
    '''
    This function loads feature table and distrubutes it to several variables
    
    :param path: a path to feature table
    :param label: a parameter which indicates whether data labeled or not
    :return: tuple of * a table which includes protein names, substitutions and their positions
                      * a table with features
                      * labels or None
                      * probabilities obtained by PolyPhen-2
                      * predictions obtained by PolyPhen-2
    '''
    
    data = pd.read_table(path, sep='\t')    
    
    subs = data.iloc[:, :4]
    
    X = data.iloc[:, 4:(data.shape[1] - 3)] if labels else data.iloc[:, 4:(data.shape[1] - 2)]
    y = data['is_del'] if labels else None
    
    pph2_prob, pph2_pred = data['pph2_prob'], data['is_del_pph2']    
    
    return subs, X, y, pph2_prob, pph2_pred


def copy_clfs(sdir, ind, ldir, is_wo_w):
    '''
    This function copies the best classifiers to a separate directory
    
    :param sdir: a path to directory contained directories with repetitions' results
    :param ind: a name of directory contained the best classifiers
    :param ldir: a path to a separate directory
    :param is_wo_w: a parameter which indicates whether classifiers fitted with weights or not
    '''
    
    cdir = sdir + '{:03}'.format(ind) + '/'
    
    if is_wo_w:
        clfs = [c for c in os.listdir(cdir) if c.find('with') == -1]
    else:
        clfs = [c for c in os.listdir(cdir) if c.find('wout') == -1]
    
    for clf in clfs:
        sh.copy(cdir + clf, ldir)
        
        
def result_table(subs, X, stdX, y, pph2_prob, pph2_pred, clfs, X_test=None):
    '''
    This function computes a resulting table which consist of an initial table, scaled features,
    distances/probabilites for ROC-analysis and predictions
    
    :param subs: a table which includes protein names, substitutions and their positions
    :param X: a table with features
    :param stdX: a table with scaled features
    :param y: labels
    :param pph2_prob: probabilities obtained by PolyPhen-2
    :param pph2_pred: predictions obtained by PolyPhen-2
    :param clfs: a set of fited classifiers in interest
    :param X_test: None or test data for A. thaliana
    :return: a resulting table
    '''
    
    tab = pd.concat([subs, X, stdX, y, pph2_prob], axis=1)
    cols = list(subs) + list(X) + ['Scaled_' + l for l in list(X)] + ['is_deleterious', 'prob_pph2']
    tab.columns = cols
    
    # Make indicator to know whether we work with A. thaliana or not
    is_arab = not type(X_test) == type(None)
    
    # Columns' names for distances/probabilites
    if is_arab:
        cols = ['prob_{}_wout_w'.format(c) for c in ['lsvm', 'gsvm', 'rf']]
    else:
        cols = ['prob_{}_wout_w'.format(c) for c in ['lsvm', 'gsvm', 'rf']] + ['prob_{}_with_w'.format(c) for c in ['lsvm', 'gsvm', 'rf']]
    
    # Computation of distances/probabilites
    for c, clf in zip(cols, clfs):
        if is_arab:
            col = np.full(tab.shape[0], float('nan'))
            col[X_test.index] = clf.predict_proba(X_test)[:, 1] if c[5:7] == 'rf' else clf.decision_function(X_test)
            tab[c] = col
        else:
            tab[c] = clf.predict_proba(stdX)[:, 1] if c[5:7] == 'rf' else clf.decision_function(stdX)
            
    tab['pred_pph2'] = pph2_pred
    
    # Columns' names for predictions
    if is_arab:
        cols = ['pred_{}_wout_w'.format(c) for c in ['lsvm', 'gsvm', 'rf']]
    else:
        cols = ['pred_{}_wout_w'.format(c) for c in ['lsvm', 'gsvm', 'rf']] + ['pred_{}_with_w'.format(c) for c in ['lsvm', 'gsvm', 'rf']]
    
    # Computation of predictions
    for c, clf in zip(cols, clfs):
        if is_arab:
            col = np.full(tab.shape[0], float('nan'))
            col[X_test.index] = clf.predict(X_test)
            tab[c] = col
        else:
            tab[c] = clf.predict(stdX)
            
    return tab


def arab_plot_accs(X_train, X_test, y_train, y_test, clfs):
    '''
    This function computes accuracies for the train and test data
    from A. thaliana dataset and creates the corresponding bar chart
    
    :param X_train: A. thaliana train data 
    :param X_test: A. thaliana test data
    :param y_train: labes of A. thaliana train data 
    :param y_test: labes of A. thaliana test data 
    :return: a table with the accuracies
    '''
    
    acc_tr = [accuracy_score(y_train, c.predict(X_train)) for c in clfs]
    acc_te = [accuracy_score(y_test, c.predict(X_test)) for c in clfs]

    ind = np.arange(len(acc_tr)) 
    width = 0.15
    y_val = np.arange(0, 1.01, 0.05)
    
    fsize = 13
    
    plt.figure(figsize=(15, 8))
    plt.barh(ind, acc_tr, width, label='Train data')
    plt.barh(ind + width, acc_te, width, label='Test data')
    plt.xlabel('Accuracy', fontsize=fsize)
    plt.yticks(ind + width / 2, ('lSVM', 'gSVM', 'RF'), fontsize=fsize)
    plt.xticks(y_val, fontsize=fsize)
    plt.grid(True, axis='x', linestyle='--')
    plt.legend(loc='best', framealpha=1, fontsize=fsize)
    plt.show()
    
    indx = ['lSVM', 'gSVM', 'RF']
    cols = ['Train data', 'Test data']
    tab = pd.DataFrame(np.vstack([acc_tr, acc_te]).T, index=indx, columns=cols).round(3)
    
    return tab
    

def mets_mats_arab(tab):
    '''
    This function makes confusion matrixes and calcutates metrics in interest related with A. thaliana
    
    :param tab: a resulting table for A. thaliana
    :return: tuple of * confusion matrixes
                      * a table with metrics in interst
    '''
    
    # Names of columns and rows
    label_clfs = ['PPh2', 'lSVM', 'gSVM', 'RF']
    label_mets  =['Accuracy', 'FPR', 'FNR', 'Sensetivity', 'Specificity', 'AUC']
    label_mat_ind, label_mat_col = ['Actual_Neu', 'Actual_Del'], ['Predicted_Neu', 'Predicted_Del']
    
    probs = tab.iloc[:, -2 * len(label_clfs):-len(label_clfs)].dropna()
    preds = tab.iloc[:, -len(label_clfs):].dropna()
    true = tab.is_deleterious.iloc[preds.index]
    
    # Make confusion matrixes and calcutate metrics
    mets, mats = [], []
    for c_probs, c_preds in zip(probs, preds):
        tn, fp, fn, tp = confusion_matrix(true, preds[c_preds]).ravel()
        acc = accuracy_score(true, preds[c_preds])
        sen = tp / (fn + tp)
        fnr = fn / (fn + tp)
        spe = tn / (fp + tn)
        fpr = fp / (fp + tn)
        auc = roc_auc_score(true, probs[c_probs])
        mets.append([acc, fpr, fnr, sen, spe, auc])
        mats.append(np.array([[tn, fp], [fn, tp]]))
    
    # Package results in tables
    mats_cols = pd.MultiIndex.from_product([label_clfs, label_mat_col])
    mats = pd.DataFrame(np.hstack(mats), index=label_mat_ind, columns=mats_cols)
    mets = pd.DataFrame(mets, index=label_clfs, columns=label_mets).round(3).transpose()
    
    return mats, mets
    
    
def mets_others(tab):
    '''
    This function calcutates metrics in interest related with the others organisms
    
    :param tab: a resulting table for the particular organism 
    :return: a table with metrics in interst
    '''
    
    # Names of columns and rows
    label_clfs = ['PPh2', 'lSVM', 'gSVM', 'RF', 'lSVM + TL', 'gSVM + TL', 'RF + TL']
    label_mets  =['Accuracy', 'FPR', 'FNR', 'AUC']
    
    true = tab.is_deleterious
    probs = tab.iloc[:, -2 * len(label_clfs):-len(label_clfs)]
    preds = tab.iloc[:, -len(label_clfs):]
    
    # Calcutate metrics
    mets = []
    for c_probs, c_preds in zip(probs, preds):
        tn, fp, fn, tp = confusion_matrix(true, preds[c_preds]).ravel()
        acc = accuracy_score(true, preds[c_preds])
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        auc = roc_auc_score(true, probs[c_probs])
        mets.append([acc, fpr, fnr, auc])
    
    # Package results in table
    mets = pd.DataFrame(mets, index=label_clfs, columns=label_mets).round(3)
    
    return mets


def roc_curves(tab, is_arab):
    '''
    This function plots ROC-curves
    
    :param tab: a resulting table for the particular organism 
    :param is_arab: a parameter which indicates whether table is for A. thaliana or not
    '''
    
    if is_arab:
        label_clfs = ['PPh2', 'lSVM', 'gSVM', 'RF']    
        preds = tab.iloc[:, -2 * len(label_clfs):-len(label_clfs)].dropna()
        true = tab.is_deleterious.iloc[preds.index]
    else:
        label_clfs = ['PPh2', 'lSVM', 'gSVM', 'RF', 'lSVM + TL', 'gSVM + TL', 'RF + TL']
        preds = tab.iloc[:, -2 * len(label_clfs):-len(label_clfs)]
        true = tab.is_deleterious
        
    curves = []
    for clf in preds:
        fpr, tpr, _ = roc_curve(true, preds[clf])        
        auc = roc_auc_score(true, preds[clf])
        curves.append([fpr, tpr, auc])
        
    fsize = 13
    
    plt.figure(figsize=(14, 14))
    for cur, clf in zip(curves, label_clfs):
        lab = clf + ': AUC = {}'.format(round(cur[2], 3))
        plt.plot(cur[0], cur[1], label=lab)
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('False Positive Rate', fontsize=fsize)
    plt.ylabel('True Positive Rate', fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.legend(loc='best', framealpha=1, fontsize=fsize)
    plt.show()

    
def cicer_table(subs, X, stdX, pph2_pred, best_clf):
    '''
    This function computes a resulting table for C. arietinum, which consist of an initial table,
    scaled features, predictions from PolyPhen-2 and the best classifier
    
    :param subs: a table which includes protein names, substitutions and their positions
    :param X: a table with features
    :param stdX: a table with scaled features
    :param pph2_pred: predictions obtained by PolyPhen-2
    :param best_clf: the best classifier
    :return: a resulting table
    '''
    
    tab = pd.concat([subs, X, stdX, pph2_pred], axis=1)
    cols = list(subs) + list(X) + ['Scaled_' + l for l in list(X)] + ['pred_pph2']
    tab.columns = cols
    
    tab['pred_best_clf'] = best_clf.predict(stdX)
            
    return tab
    