from sn_analysis.sn_tools import loadData_fakeSimu
import matplotlib.pyplot as plt
import numpy as np
from sn_analysis.sn_calc_plot import Calc_zlim, select
from sn_analysis.sn_calc_plot import effi
import pandas as pd
from . import plt, filtercolors


def gime_zlim(df, dict_sel, selvar):

    df['sigmaC'] = np.sqrt(df['Cov_colorcolor'])
    # simulation parameters
    # histSN_params(df)

    sel = select(df, dict_sel[selvar])

    """
    fig, ax = plt.subplots()
    plot_effi(df, sel, leg='test', fig='tt', ax=ax)
    plot_2D(df)
    """
    effival = effi(df, sel)
    mycl = Calc_zlim(effival)
    zlim = mycl.zlim

    # mycl.plot_zlim(zlim)

    return zlim


def plot_2D(res, varx='z', legx='$', vary='sigmaC',
            legy='$\sigma C$', fig=None, ax=None):

    if fig is None:
        fig, ax = plt.subplots()

    ax.plot(res[varx], res[vary], 'ko')

    # plt.show()


def plot_res(df, fig=None, ax=None, label='', color='b', lst='solid', mark='None', mfc='None'):

    if fig is None:
        fig, ax = plt.subplots(figsize=(11, 6))

    idx = df['moon_frac'] < 0
    zlim_ref = df[idx]['zlim'].values[0]
    df.loc[idx, 'moon_frac'] = 0.
    print(zlim_ref)
    df['delta_zlim'] = df['zlim']-zlim_ref
    print(df)

    idx = df['moon_frac'] > -0.5
    sel = df[idx]
    sel = sel.sort_values(by=['moon_frac'])
    ax.plot(sel['moon_frac'], sel['delta_zlim'],
            label=label, color=color, linestyle=lst, marker=mark, mfc=mfc)
    # ax.grid()


def get_data(theDir, fis):

    df_dict = {}
    for fi in fis:
        print('processing', fi)
        ffi = fi.split('/')[-1]
        hname = ffi.split('_')[4]
        bb = ffi.split('_')[2]
        df = loadData_fakeSimu(theDir, ffi)
        key = '{}_{}'.format(bb, hname)
        if key not in df_dict.keys():
            df_dict[key] = df
        else:
            df_dict[key] = pd.concat((df, df_dict[key]))

    return df_dict


def plot_delta_zlim(df_dict, dict_sel, selvar):

    r = []
    for key, vals in df_dict.items():
        zlim = gime_zlim(vals, dict_sel, selvar)
        tt = key.split('_')
        bb = tt[0]
        hname = tt[1]
        print(hname, zlim)
        r.append((bb, int(hname), zlim))

    df = pd.DataFrame.from_records(r, columns=['band', 'moon_frac', 'zlim'])

    print(df)

    fig, ax = plt.subplots(figsize=(12, 7))
    leg = {}
    bands = 'rizy'
    for b in bands:
        leg[b] = 'u <-> '+b

    lst = dict(zip(bands, ['solid', 'dashed', 'dotted', 'dashdot']))
    marks = dict(zip(bands, ['o', 's', '^', 'h']))
    for b in bands:
        idx = df['band'] == b
        plot_res(df[idx], fig=fig, ax=ax, label=leg[b],
                 color=filtercolors[b], lst=lst[b], mark=marks[b])

    ax.set_xlabel('Moon Phase [%]', fontweight='bold')
    zcomp = '$z_{complete}$'
    ax.set_ylabel('$\Delta $ {}'.format(zcomp))

    ax.grid()
    ax.legend()


def plot_delta_nsn(df_dict, dict_sel, selvar, zmin=0.8):

    r = []

    for key, df in df_dict.items():
        tt = key.split('_')
        bb = tt[0]
        hname = tt[1]

        # print(hname, zlim)
        # r.append((bb, int(hname), zlim))
        df['sigmaC'] = np.sqrt(df['Cov_colorcolor'])
        # plt.plot(df['z'], df['sigmaC'], 'ko')
        # check_simuparams(df)
        seldf = select(df, dict_sel[selvar])
        idx = seldf['z'] >= zmin
        print(key, len(seldf), len(seldf[idx]))
        nsn_tot = len(seldf)
        nsn_07 = len(seldf[idx])
        r.append((bb, int(hname), nsn_tot, nsn_07))

    df = pd.DataFrame.from_records(
        r, columns=['band', 'moon_frac', 'nSN', 'nSN_0{}'.format(int(10*zmin))])

    fig, ax = plt.subplots(figsize=(12, 7))
    leg = {}
    bands = 'rizy'
    for b in bands:
        leg[b] = 'u <-> '+b

    lst = dict(zip(bands, ['solid', 'dashed', 'dotted', 'dashdot']))
    marks = dict(zip(bands, ['o', 's', '^', 'h']))

    vartoplot = 'nSN_0{}'.format(int(10*zmin))
    for b in bands:
        idx = df['band'] == b
        plot_resb(df[idx], fig=fig, ax=ax, var=vartoplot, label=leg[b],
                  color=filtercolors[b], lst=lst[b], mark=marks[b])

    if vartoplot != 'nSN':
        ttit = '$z\geq$'
        ttit += '{}'.format(np.round(zmin, 1))
        fig.suptitle(ttit)
    ax.set_xlabel('Moon Phase [%]', fontweight='bold')
    zcomp = '$N_{SN}$'
    ax.set_ylabel('$\Delta $ {}'.format(zcomp))
    ax.set_ylabel(
        r'$\frac{\mathrm{N_{SN}}}{\mathrm{N_{SN}}^{\mathrm{Moon~Phase=0}}}$')
    # ax.set_ylabel('{}'.format(legb))

    ax.legend()

    ax.grid()


def plot_resb(df, fig=None, ax=None, var='nSN', label='', color='b',
              lst='solid', mark='None', mfc='None'):

    if fig is None:
        fig, ax = plt.subplots(figsize=(11, 6))

    idx = df['moon_frac'] < 0
    nSN_ref = df[idx][var].values[0]
    df.loc[idx, 'moon_frac'] = 0.
    print(nSN_ref)
    p = df[var]/nSN_ref
    df['delta_nSN'] = p
    df['var_nSN'] = np.sqrt(nSN_ref*p*(1.-p))
    df['err_nSN'] = df['var_nSN']/nSN_ref
    print(df)

    idx = df['moon_frac'] > -0.5
    sel = df[idx]
    sel = sel.sort_values(by=['moon_frac'])
    ax.plot(sel['moon_frac'], sel['delta_nSN'], label=label,
            color=color, linestyle=lst, marker=mark, mfc=mfc)
    # ax.errorbar(sel['moon_frac'], sel['delta_nSN'], yerr=sel['err_nSN'],
    #            label=label, color=color, linestyle=lst, marker=mark, mfc=mfc)
    # ax.grid()


def check_simuparams(df):

    fig, ax = plt.subplots(ncols=2, nrows=2)

    vvars = ['x1', 'color', 'daymax', 'z']
    ppos = [(0, 0), (0, 1), (1, 0), (1, 1)]

    dpos = dict(zip(vvars, ppos))

    for key, vals in dpos.items():
        i = vals[0]
        j = vals[1]
        ax[i, j].hist(df[key], histtype='step')
