# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 19:38:52 2018

@author: Shoshannah
"""
from astropy import constants
import numpy as np
import matplotlib.pyplot as pplt

#def constants
G = constants.G.value
c = constants.c.value
hbar = constants.hbar.value
m_e = constants.m_e.value
m_p = constants.m_p.value
k = constants.k_B.value
sigma_sb = constants.sigma_sb.value
pi = np.pi

a = 4.*sigma_sb/c
gamma = 5./3.
X = 0.55
Y = 0.4
Z = 0.05
mu = (2*X + 0.75*Y + 0.5*Z)**(-1.)

#def functions
def eps(Rho,T):     # (Eqn. 9) total specific energy generation rate 
    eps_pp = 1.07E-7*(Rho/1E5)*X**2.*(T/1E6)**4.
    eps_cno = 8.24E-26*(Rho/1E5)*0.03*X**2.*(T/1E6)**19.9
    return eps_pp + eps_cno

def kappa(Rho,T):   # (Eqn. 14) radiative opacity 
    kappa_Hminus = 2.5E-32*(Z/0.02)*(Rho/1E3)**0.5*T**9.
    kappa_ff = 1.0E24*(Z + 0.0001)*(Rho/1E3)**0.7*T**(-3.5)
    kappa_es = 0.02*(1 + X)
    max_kesff = max(kappa_es,kappa_ff)
    return (1/kappa_Hminus + 1./max_kesff)**(-1.)
    
def P(Rho,T):       # (Eqn. 5) equation of state, total pressure from  non-relativistic degenerate, ideal gas and photon gas 
    P_deg = ((3.*pi**2.)**(2./3.)/5.)*(hbar**2./m_e)*(Rho/m_p)**(5./3.)
    P_IG = Rho*k*T/(mu*m_p)
    P_PG = a*T**4./3.
    return P_deg + P_IG + P_PG
    
def dT_dR(Rho,T,R,L,M,P,kappa): # (Eqn. 2b) energy transfer equation
    rad_transf = 3.*kappa*Rho*L/(16.*pi*a*c*T**3.*R**2.)
    conv = (1 - 1./gamma)*T*G*M*Rho/(P*R**2.)
    min_dT = min(rad_transf,conv)
    return -rad_transf,-conv,-min_dT
    
def diP_diRho(Rho,T):           # (Eqn. 7a) 
    val = ((3.*pi**2.)**(2./3.)/3.)*(hbar**2./(m_e*m_p))*(Rho/m_p)**(2./3.) + k*T/(mu*m_p)
    return val
    
def diP_diT(Rho,T):             # (Eqn. 7b) Eqn. 7a and b are both partial derivative of Eqn. 5, to be used in Eqn. 2a
    val = Rho*k/(mu*m_p) + (4.*a*T**3./3.)
    return val
    
def dRho_dR(Rho,R,M,diP_diT,dT_dR,diP_diRho): # (Eqn. 2a)
    num = -(G*M*Rho/R**2. + diP_diT*dT_dR)
    denom = diP_diRho
    return num/denom
    
def L(Rho,T,R): # (Eqn. 15d) ``center'' luminosity, R input should be near the center of the star 
    return 4.*pi*R**3.*Rho*eps(Rho,T)
    
def M(Rho,R): # (Eqn. 15c) enclosed mass near center, R input should be near the center of the star
    return 4.*pi*R**3.*Rho
    
def gen_dRho_dR(Rho_0,T_0,R_0,L_0,M_0): # ``center'' density gradient, application of Eqn. 2a with center parameters
    kappa_0 = kappa(Rho_0,T_0)
    P_0 = P(Rho_0,T_0)
    dT_dR_0 = dT_dR(Rho_0,T_0,R_0,L_0,M_0,P_0,kappa_0)
    diP_diRho_0 = diP_diRho(Rho_0,T_0)
    diP_diT_0 = diP_diT(Rho_0,T_0)
    dRho_dR_0 = dRho_dR(Rho_0,R_0,M_0,diP_diT_0,dT_dR_0,diP_diRho_0)
    return dRho_dR_0
    
def EM(y0,dy0,delx): # euler's method
    val = y0 + dy0*delx
    return val
    

#def init conditions (using IC for Sun)
Rho_i = 160000 #kg/m^3
T_i = 15000000 #K
R_i = 1E-10
L_i = L(Rho_i,T_i,R_i)
M_i = M(Rho_i,R_i)

###tests with eulers method###

#Lists of Variables to Plot for Testing purposes
Ts = []
Rs = []
Rhos = []
kappas = []
Ls = []
dTau_dRs = []
dT_rads = []
dT_convs = []
dT_dRs = []
delta_Taus = []

#Range of R to iterate through
R_range = np.linspace(R_i,695700000,1E5)
delR = R_range[1] - R_range[0]

#set init conditions p1
T_j = T_i
Rho_j = Rho_i

for j in range(0,len(R_range)):
    #Append new T and Rho
    Ts.append(T_j)
    Rhos.append(Rho_j)
    #get R
    R_j = R_range[j]
    Rs.append(R_j)
    #Get Inits and append
    L_j = L(Rho_j,T_j,R_j)
    M_j = M(Rho_j,R_j)
    P_j = P(Rho_j,T_j)
    kappa_j = kappa(Rho_j,T_j)
    kappas.append(kappa_j)
    Ls.append(L_j)
    #get dtau
    #dTau_dR_J = kappa_j*Rho_j
    #dTau_dRs.append(dTau_dR_J)
    #get derivs
    dT_rad_j,dT_conv_j,dT_dR_j = dT_dR(Rho_j,T_j,R_j,L_j,M_j,P_j,kappa_j)
    dT_rads.append(dT_rad_j)
    dT_convs.append(dT_conv_j)
    dT_dRs.append(dT_dR_j)
    diP_diRho_j = diP_diRho(Rho_j,T_j)
    diP_diT_j = diP_diT(Rho_j,T_j)
    dRho_dR_j = dRho_dR(Rho_j,R_j,M_j,diP_diT_j,dT_dR_j,diP_diRho_j)
    delta_Tau = kappa_j*Rho_j**2./np.abs(dRho_dR_j)
    delta_Taus.append(delta_Tau)
    #step up one with Euler's Method
    T_jp1 = EM(T_j,dT_dR_j,delR)
    Rho_jp1 = EM(Rho_j,dRho_dR_j,delR)
    #redefine
    T_j = T_jp1
    Rho_j = Rho_jp1
    
#Plots
#Plot Temp
pplt.plot(Rs,Ts)
pplt.show()
pplt.title('Temperature as func of R')
pplt.xlabel('T (K)')
pplt.ylabel('R (m)')
#Plot Density
pplt.plot(Rs,Rhos)
pplt.title('Density as func of R')
pplt.xlabel('Density (kg/m^3)')
pplt.ylabel('R (m)')
pplt.show()

#scratch work
#Plot Tau Stuff
#print(np.sum(dTau_dRs)*delR)
#pplt.plot(Rs,delta_Taus)
#pplt.ylim(0,1E13)
#pplt.show()
#print(np.min(delta_Taus))

##Temp stops decreasing at R=1E8 why?????????

#Plot dT
pplt.plot(Rs,dT_rads,'b',label='Radiative dT')
pplt.plot(Rs,dT_convs,'r',label='Convective dT')
pplt.title('dT/dR from radiative diffusion or convection')
pplt.xlabel('R (m)')
pplt.ylabel('dT/dR (K/m)')
pplt.show()

##Plot variables in radiative diff to see why dT/dR goes to zero

#Plot Kappa
pplt.plot(Rs,kappas)
pplt.title('Opacity as a func of R')
pplt.xlabel('R (m)')
pplt.ylabel('Opacity (units)')
pplt.show()
#Plot Luminosity
pplt.plot(Rs,Ls)
pplt.title('Luminosity as a func of R')
pplt.xlabel('R (m)')
pplt.ylabel('L (W)')
pplt.show()

#Luminosity changes with epsilon
#plot epsilons
Rho_arr = np.array(Rhos)
T_arr = np.array(Ts)
Eps_arr = eps(Rho_arr,T_arr)

pplt.plot(Rs,Eps_arr)
pplt.title('Reaction Rate')
pplt.xlabel('R (m)')
pplt.ylabel('Reaction Rate')
pplt.show()

