# -------------------------------------------------------------------------
# Name:        Waterdemand modules
# Purpose:
#
# Author:      Yuancheng Xu
#
# Created:     26/10/2024
# Copyright:   (c) YC Xu 2024
# -------------------------------------------------------------------------

from cwatm.management_modules import globals
from cwatm.management_modules.data_handling import *

import numpy as np
import pandas as pd
from datetime import datetime

# from cwatm.management_modules.data_handling import *  # luca for testing
# import matplotlib.pyplot as plt


# def decompress(map, nanvalue=None):
#    """
#    Decompressing CWatM maps from 1D to 2D with missing values
#
#    :param map: compressed map
#    :return: decompressed 2D map
#    """
#
#    dmap = maskinfo['maskall'].copy()
#    dmap[~maskinfo['maskflat']] = map[:]
#    if nanvalue is not None:
#        dmap.data[np.isnan(dmap.data)] = nanvalue
#
#    return dmap.data

class diversion_yellow:
    """
    WATERDEMAND

    calculating water demand - irrigation
    Agricultural water demand based on water need by plants

    **Global variables**

    =====================================  ======================================================================  =====
    Variable [self.var]                    Description                                                             Unit 
    =====================================  ======================================================================  =====
    load_initial                           Settings initLoad holds initial conditions for variables                input
    topwater                               quantity of water above the soil (flooding)                             m    
    cropKC                                 crop coefficient for each of the 4 different land cover types (forest,  --   
    efficiencyPaddy                        Input, irrPaddy_efficiency, paddy irrigation efficiency, the amount of  frac 
    efficiencyNonpaddy                     Input, irrNonPaddy_efficiency, non-paddy irrigation efficiency, the am  frac 
    returnfractionIrr                      Input, irrigation_returnfraction, the fraction of non-efficient water   frac 
    alphaDepletion                         Input, alphaDepletion, irrigation aims to alphaDepletion of field capa  frac 
    minimum_irrigation                     Cover-specific irrigation in metres is 0 if less than this, currently   1/m2 
    pot_irrConsumption                     Cover-specific potential irrigation consumption                         m/m  
    fraction_IncreaseIrrigation_Nonpaddy   Input, fraction_IncreaseIrrigation_Nonpaddy, scales pot_irrConsumption  frac 
    irrPaddyDemand                         Paddy irrigation demand                                                 m    
    availWaterInfiltration                 quantity of water reaching the soil after interception, more snowmelt   m    
    ws1                                    Maximum storage capacity in layer 1                                     m    
    ws2                                    Maximum storage capacity in layer 2                                     m    
    wfc1                                   Soil moisture at field capacity in layer 1                                   
    wfc2                                   Soil moisture at field capacity in layer 2                                   
    wwp1                                   Soil moisture at wilting point in layer 1                                    
    wwp2                                   Soil moisture at wilting point in layer 2                                    
    arnoBeta                                                                                                            
    maxtopwater                            maximum heigth of topwater                                              m    
    totAvlWater                            Field capacity minus wilting point in soil layers 1 and 2               m    
    InvCellArea                            Inverse of cell area of each simulated mesh                             1/m2 
    w1                                     Simulated water storage in the layer 1                                  m    
    w2                                     Simulated water storage in the layer 2                                  m    
    totalPotET                             Potential evaporation per land use class                                m    
    fracVegCover                           Fraction of specific land covers (0=forest, 1=grasslands, etc.)         %    
    unmetDemand                            Unmet demand                                                            m    
    unmetDemandPaddy                       Unmet paddy demand                                                      m    
    unmetDemandNonpaddy                    Unmet nonpaddy demand                                                   m    
    irrDemand                              Cover-specific Irrigation demand                                        m/m  
    irrNonpaddyDemand                                                                                                   
    totalIrrDemand                         Irrigation demand                                                       m    
    =====================================  ======================================================================  =====

    **Functions**
    """

    def __init__(self, model):
        self.var = model.var
        self.model = model

    def initial(self):
        """
        Initial part of the water diversion module
        """
        self.var.maskDivYellow = loadmap('maskDivYellow')

        pathDivYellowFile = cbinding('DivYellowFile')
        self.var.DivYelTimeInterval = cbinding('DivYelTimeInterval')
        inputRaw = pd.read_excel(pathDivYellowFile)
        inputDate = inputRaw.values[:, 0]
        self.var.DivYelData = inputRaw.values[:, inputRaw.shape[1] - 1]
        self.var.subdivideDataYel = inputRaw.iloc[:, 1 : inputRaw.shape[1] - 1]

        self.var.residualDivYel = 0
        self.var.AvlDivYel = 0

        self.var.AbstractionFraction_DivYellow_Domestic = loadmap('AbstractionFraction_DivYellow_Domestic')
        self.var.AbstractionFraction_DivYellow_Livestock = loadmap('AbstractionFraction_DivYellow_Livestock')
        self.var.AbstractionFraction_DivYellow_Industry = loadmap('AbstractionFraction_DivYellow_Industry')
        self.var.AbstractionFraction_DivYellow_Irrigation = loadmap('AbstractionFraction_DivYellow_Irrigation')

        if self.var.DivYelTimeInterval == 'monthly':
            self.var.DivYelDate = []
            for i in inputDate:
                self.var.DivYelDate = np.append(self.var.DivYelDate, datetime.strptime(i.astype(int).astype(str), '%Y/%m'))
        elif self.var.DivYelTimeInterval == 'yearly':
            self.var.DivYelDate = []
            for i in inputDate:
                self.var.DivYelDate = np.append(self.var.DivYelDate, datetime.strptime(i.astype(int).astype(str), '%Y'))
        else:
            raise Exception("Error in DivYelTimeInterval!")


    def dynamic(self, wd_date):
        """
        Dynamic part of the water diversion module

        """

        if self.var.DivYelTimeInterval == 'monthly':
            timediv = globals.dateVar['daysInMonth']
        elif self.var.DivYelTimeInterval == 'yearly':
            timediv = globals.dateVar['daysInYear']
        else:
            raise Exception("Error in DivYelTimeInterval!")

        if self.var.DivYelTimeInterval == 'monthly':
            raise Exception("Error: monthly diversion input under construction!")
        elif self.var.DivYelTimeInterval == 'yearly':
            if wd_date >= datetime.strptime( str(self.var.DivYelDate[len(self.var.DivYelDate)-1].year+1), '%Y'):
                raise Exception("Date of interest out of diversion input!")
            timeID_list = np.where(self.var.DivYelDate <= wd_date)[0]
            if len(timeID_list) == 0:
                raise Exception("Date of interest out of diversion input!")
            else:
                timeID = timeID_list[len(timeID_list) - 1]
        else:
            raise Exception("Error in DivYelTimeInterval!")
        
        self.var.AvlDivYel = self.var.residualDivYel + self.var.DivYelData[timeID]/timediv
        self.var.subdivideYel = self.var.subdivideDataYel.iloc[timeID, :]
