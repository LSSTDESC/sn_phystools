#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed May 3 09:40:35 2023
@author: philippe.gris@clermont.in2p3.fr/andrea.antoniali@etu.uca.fr
"""


import numpy as np
from iminuit import Minuit
from abc import ABC, abstractmethod
import pandas as pd


class CosmoFit(ABC):
    def __init__(self, dataValues, dataNames, fitparNames=['w0', 'wa', 'Om0'],
                 cosmo_model='w0waCDM',
                 cosmo_default=dict(
                     zip(['w0', 'wa', 'Om0'], [-1.0, 0.0, 0.3])),
                 prior=pd.DataFrame(), par_protect_fit=[]):
        """
        Abstract class to estimate cosmoly parameters

        Parameters
        ----------
        dataValues : list(array)
            data to fit.
        dataNames : list(str)
            corresponding list names.
        fitparNames : str, optional
            List of parameters to fit. The default is ['w0','wa','Om0'].
        cosmo_model : str, optional
            Name of the cosmology model. The default is 'w0waCDM'.
        cosmo_default : dict, optional
            Default values for the cosmology model.
            The default is dict(zip(['w0', 'wa', 'Om0'], [-1.0, 0.0, 0.3])).
        prior : pd.DataFrame, optional
            Priors to include in the chisquare. The default is pd.DataFrame().
        par_protect_fit : list(str), optional
            List of fit parameters that have to be protected (example: Om0 > 0)
            The default is [].

        Returns
        -------
        None.

        """

        for i, vals in enumerate(dataNames):
            exec('self.{} = dataValues[{}]'.format(vals, i))

        self.h = 1.e-7
        self.fitparNames = fitparNames
        self.cosmo_model = cosmo_model
        self.cosmo_default = cosmo_default
        self.prior = prior
        self.par_protect_fit = par_protect_fit

    @ abstractmethod
    def fit_function(self,  parameters, fitparNames=[]):
        """

        Abstract method where the fit function is defined

        Parameters
        ----------
        parameters : list(float)
            parameters to fit.
        fitparNames : list(str), optional
            corresponding list of names. The default is [].

        Returns
        -------
        None.

        """

        pass

    @ abstractmethod
    def xi_square(self, *parameters):
        """
        Abstract method to define the ChiSquare

        Parameters
        ----------
        *parameters : list
            Parameters of the fit function

        Returns
        -------
        None.

        """
        pass

    def updating_a_parameter(self, i_par, i_list, *parameters):
        '''
        Updates a parameter by an small amount and calculates the Xisquare
        for this change for the change. Does this following a sequence.
        Parameters
        ----------
        i_par : integer
            the index of the parameter to be updated.

        i_list : list of numerical values.
            How much at each interation we modify the paramater by h.

        *parameters : tuple of differents types of entries.
              see the definition in function()
        Returns
        -------
        D : Array of numerical values
            Array of all the Xi_square calculated for each iterations,
            then used for fisher calculations.
        '''
        D = np.zeros(len(i_list))
        for i in range(len(i_list)):
            parameters = list(parameters)
            parameters[i_par] += i_list[i]*(self.h)
            parameters = tuple(parameters)
            D[i] = self.xi_square(*parameters)
        return D

    def updating_two_parameters(self, i_par, j_par, i_list, j_list,
                                *parameters):
        '''
        Updates two parameter by an small amount and calculates the Xisquare
        for this change for the change. Does this following a sequence.
        Parameters
        ----------
        i_par : integer
            the index of the first parameter to be updated.

        j_par : integer
            the index of the second parameter to be updated.

        i_list : list of numerical values.
            How much at each interation we modify the first paramater by h.
            *parameters
        j_list : list of numerical values.
            How much at each interation we modify the second paramater by h.

        *parameters : tuple of differents types of entries.
              see the definition in function()
        Returns
        -------
        D : Array of numerical values
            Array of all the Xi_square calculated for each iterations,
            then used for fisher calculations.
        '''
        D = np.zeros(len(i_list))
        for i in range(len(i_list)):
            parameters = list(parameters)
            parameters[i_par] += i_list[i]*self.h
            parameters[j_par] += j_list[i]*self.h
            parameters = tuple(parameters)
            D[i] = self.xi_square(*parameters)
        return D

    def diff_Xi2_twice(self, i_par, *parameters):
        '''
        Calculate the double derivative of Xi_square.
        Parameters
        ----------
        i_par : integer
            the index of the first parameter to be updated.

        *parameters : tuple of differents types of entries.
              see the definition in function()
        Returns
        -------
        Diff : numerical value
            double derivative value  of Xi_square
        '''
        i_list = [+1, -1, -1]
        D = self.updating_a_parameter(i_par, i_list, *parameters)
        Diff = (D[0]-2*D[1]+D[2])/(self.h**2)

        return Diff

    def diff_Xi2_didj(self, i_par, j_par, *parameters):
        '''
        Calculate the cross derivative of Xi_square.
        Parameters
        ----------
        i_par :  integer
            the index of the first parameter to be updated.

        j_par :  integer
            the index of the second parameter to be updated.

        *parameters : tuple of differents types of entries.
              see the definition in function()
        Returns
        -------
        Diff : numerical value
            cross derivative value of Xi_square
        '''
        i_list = [+1, -2, 0, +2]
        j_list = [+1, -2, +2, -2]
        D = self.updating_two_parameters(
            i_par, j_par, i_list, j_list, *parameters)
        Diff = (D[0]+D[1]-D[2]-D[3])/(4*(self.h**2))
        return Diff

    def fisher(self, *parameters):
        '''
        Calculate the fisher function.
        Parameters
        ----------
        *parameters : tuple of differents types of entries.
              see the definition in function()
        Returns
        -------
        F : numpy.array of size [n_param x n_param]
            The Fisher matrix.
        '''
        n_param = len(parameters)
        F = np.zeros([n_param, n_param])
        for i in range(n_param):
            for j in range(n_param):
                if (i == j):
                    F[i, j] = 0.5*self.diff_Xi2_twice(i, *parameters)
                else:
                    F[i, j] = 0.5*self.diff_Xi2_didj(i, j, *parameters)
        return F

    def covariance_fisher(self, parameters):
        '''
        Calculate the covariance matrix from the fisher matrix
        Parameters
        ----------
        *parameters : tuple of differents types of entries.
              see the definition in function()
        Returns
        -------
        covariance_matrix : numpy.array of size [n_param x n_param]
            The covariance matrix.
        '''
        cov = np.linalg.inv(self.fisher(*parameters))
        # print('mat method \n',np.mat(self.fisher(*parameters)).I)

        dict_out = {}
        for i, vala in enumerate(self.fitparNames):
            for j, valb in enumerate(self.fitparNames):
                if j <= i:
                    dict_out['Cov_{}_{}_fisher'.format(vala, valb)] = cov[i, j]

        return dict_out

    def fisher_uncertainty_matrix(self, *parameters):
        '''
        Calculate the uncertainty matrix from the fisher matrix
        Parameters
        ----------
        *parameters : tuple of differents types of entries.
              see the definition in function()
        Returns
        -------
        uncertainty_matrix : numpy.array of size [n_param x n_param]
            The uncertainty matrix.
        '''
        uncertainty_matrix = np.sqrt(self.cov_fisher(*parameters))
        return uncertainty_matrix

    def minuit_fit(self, parameters):
        """
        gives back the Minuit object from which one can get the parameters
        and the covariance matrix.

        Parameters
        ----------
        parameters : tuple(float)
            parameters to fit.

        Returns
        -------
        dict_out : dict
            output results.

        """

        m = Minuit(self.xi_square, *parameters, name=self.fitparNames)
        if self.par_protect_fit:
            for vv in self.par_protect_fit:
                m.limits[vv] = (0, None)

        m.migrad()
        m.hesse()

        # grab the results: param values
        dict_out = {}
        res = m.values
        for name in self.fitparNames:
            dict_out['{}_fit'.format(name)] = res[name]

        # covariance matrix
        cov = m.covariance
        for i, vala in enumerate(self.fitparNames):
            for j, valb in enumerate(self.fitparNames):
                if j <= i:
                    dict_out['Cov_{}_{}_fit'.format(vala, valb)] = cov[i, j]

        return dict_out


def fom(cov_a, cov_b, cov_ab, deltaXi2=6.17):
    """
    Figure of Merit estimator : FoM = pi/A
    A = pi*(deltaXi2) *sigma_a *sigma_b *sqrt(1-pearson**2)

    Parameters
    ----------
    cov_a : float
        a covariance.
    cov_b : float
        b covariance.
    cov_ab : float
        ab covariance.
    deltaXi2 : float, optional
        degree of confidence level. The default is 6.17.
        Note : 
            1 sigma : CL : 68.3%, deltaXi2 : 2.3
            2 sigma : CL : 95,4%, deltaXi2 : 6.17
            3 sigma : CL : 9np.arange(0.01, 1.11, 0.01)9,7 % deltaXi2 : 11.8
    Returns
    -------
    FoM: float
        Figure of Merit or -1 if anomalous data.

    """

    if cov_a < 0 or cov_b < 0:
        return -1

    sigma_a = np.sqrt(cov_a)
    sigma_b = np.sqrt(cov_b)
    pearson = cov_ab/(sigma_a*sigma_b)

    if np.abs(pearson) > 1:
        return -1

    A = np.pi * deltaXi2 * sigma_a*sigma_b * np.sqrt(1-pearson**2)
    FoM = np.pi/A

    return FoM
