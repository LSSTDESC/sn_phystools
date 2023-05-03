#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed May 3 09:40:35 2023
@author: philippe.gris@clermont.in2p3.fr/andrea.antoniali@etu.uca.fr
"""


import numpy as np
from iminuit import Minuit
from abc import ABC, abstractmethod


class CosmoFit(ABC):
    def __init__(self, x, y, sigma, parNames=[], cosmo_model='w0waCDM',
                 cosmo_default=dict(
                     zip(['w0', 'wa', 'Om0'], [-1.0, 0.0, 0.3])),
                 prior={}):
        '''
        Initialize the class with the differents variables.

        Parameters
        ----------
        x : array of numerical values
            Our entry data.

        y : array of numerical values
            The results of the model with a small noise
                    (e.g our observational data ?)

        sigma : array of numerical values
            The uncertainty on our data points.

        h : a float.
            Used for differentiating our Xisquare.
        '''
        self.x = x
        self.y = y
        self.sigma = sigma
        self.h = 1.e-7
        self.parNames = parNames
        self.cosmo_model = cosmo_model
        self.cosmo_default = cosmo_default
        self.prior = prior

    @ abstractmethod
    def fit_function(self,  parameters, parNames=[]):
        """
        Abstract method where the fit function is defined

        Parameters
        ----------
        *parameters : list
            Parameters to fit.

        Returns
        -------
        None.

        """
        pass

    @ abstractmethod
    def xi_square(self, *parameters):
        """
        abstract method to define the ChiSquare

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
            Array of all the Xi_square calculated for each iterations, then used
            for our fisher calculations.
        '''
        D = np.zeros(len(i_list))
        for i in range(len(i_list)):
            parameters = list(parameters)
            parameters[i_par] += i_list[i]*(self.h)
            parameters = tuple(parameters)
            D[i] = self.xi_square(*parameters)
        return D

    def updating_two_parameters(self, i_par, j_par, i_list, j_list, *parameters):
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
            Array of all the Xi_square calculated for each iterations, then used
            for our fisher calculations.
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
        covariance_matrix = np.linalg.inv(self.fisher(*parameters))
        # print('mat method \n',np.mat(self.fisher(*parameters)).I)

        return covariance_matrix

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
        '''
        gives back the Minuit object from which one can get the parameters
        and the covariance matrix.
        Returns
        -------
        m : iminuit.Minuit
            iminuit object that can be used to extract data.
        '''

        m = Minuit(self.xi_square, *parameters, name=self.parNames)
        m.limits['Om0'] = (0, None)
        m.migrad()
        m.hesse()
        return m
