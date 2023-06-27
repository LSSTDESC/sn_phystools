#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:45:41 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import operator


def selection_criteria():

    dict_sel = {}

    dict_sel['nosel'] = [('daymax', operator.ge, 0, 1)]
    # dict_sel['nosel_z0.7'] = [('z', operator.ge, 0.7)]

    sdict = {}
    sdict['phases'] = [('n_epochs_phase_minus_10', operator.ge, 1, 1),
                       # ('n_epochs_bef', operator.ge, 2),
                       ('n_epochs_phase_plus_20', operator.ge, 1, 2)]

    sdict['G10'] = [('n_epochs_m10_p35', operator.ge, 4, 3),
                    ('n_epochs_m10_p5', operator.ge, 1, 4),
                    ('n_epochs_p5_p20', operator.ge, 1, 5),
                    ('n_bands_m8_p10', operator.ge, 2, 6),
                    ('fitstatus', operator.eq, 'fitok', 7)]

    sdict['sigmaC'] = [('sigmaC', operator.le, 0.04, 8)]
    # sdict['z0.7'] = [('z', operator.ge, 0.7)]
    sdict['JLA'] = [('sigmat0', operator.le, 2., 8),
                    ('sigmax1', operator.le, 1, 9),
                    # ('sigmaC', operator.le, 0.04, 10)
                    ]

    dict_sel['G10_sigmaC'] = sdict['phases'] + sdict['G10']+sdict['sigmaC']

    # dict_sel['G10_sigmaC_z0.7'] = dict_sel['G10_sigmaC'] + sdict['z0.7']

    dict_sel['G10_JLA'] = sdict['phases']+sdict['G10']+sdict['JLA']

    # dict_sel['G10_JLA_z0.7'] = dict_sel['G10_JLA']+sdict['z0.7']

    """
    dict_sel['metric'] = [('n_epochs_bef', operator.ge, 4),
                          ('n_epochs_aft', operator.ge, 10),
                          ('n_epochs_phase_minus_10', operator.ge, 1),
                          ('n_epochs_phase_plus_20', operator.ge, 1),
                          ('sigmaC', operator.le, 0.04),
                          ]
    """

    return dict_sel
