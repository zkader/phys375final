"""
Rewrite of Shoshannah's code as a class

@Zarif
"""
from astropy import constants
import numpy as np
import matplotlib.pyplot as pplt
from scipy.integrate import odeint,ode

#def constants
G = constants.G.value
c = constants.c.value
hbar = constants.hbar.value
m_e = constants.m_e.value
m_p = constants.m_p.value
k_B = constants.k_B.value
sigma_sb = constants.sigma_sb.value
pi = np.pi

class MS_Star:
    def __init__(self,rho_c,T_c): # initial setup 
        self.rho_c = rho_c
        self.T_c = T_c
        self.a = 4.*sigma_sb/c
        self.gamma = 5./3.
        self.X = 0.55
        self.Y = 0.4
        self.Z = 0.05
        self.mu = (2*self.X + 0.75*self.Y + 0.5*self.Z)**(-1.)
        self.M_o = (4*pi/3.0)*(1e-16)**3*self.rho_c
        self.L_o = self.M_o*self.E_rate(rho_c,T_c)

    def Pressure(self,Rho,T): # (Eqn. 5) equation of state, total pressure from  non-relativistic degenerate, ideal gas and photon gas 
        P_deg = ((3.*pi**2.)**(2./3.)/5.)*(hbar**2./m_e)*(Rho/m_p)**(5./3.)
        P_IG = Rho*k_B*T/(self.mu*m_p)
        P_PG = self.a*T**4./3.
        return P_deg + P_IG + P_PG

    def E_rate(self,Rho,T):     # (Eqn. 9) total specific energy generation rate
        eps_pp = 1.07E-7*(Rho/1E5)*self.X**2.*(T/1E6)**4.
        eps_cno = 8.24E-26*(Rho/1E5)*0.03*self.X**2.*(T/1E6)**19.9
        return eps_pp + eps_cno

    def Opacity(self,Rho,T):   # (Eqn. 14) radiative opacity 
        kappa_Hminus = 2.5E-32*(self.Z/0.02)*(Rho/1E3)**0.5*T**9.
        kappa_ff = 1.0E24*(self.Z + 0.0001)*(Rho/1E3)**0.7*T**(-3.5)
        kappa_es = 0.02*(1 + self.X)
        max_kesff = np.maximum(kappa_es,kappa_ff)
        return (kappa_Hminus**-1 + max_kesff**-1)**(-1.)

    def diP_diRho(self,Rho,T):           # (Eqn. 7a) 
        return ((3.*pi**2.)**(2./3.)/3)*(hbar**2./(m_e*m_p))*(Rho/m_p)**(2./3.) + k_B*T/(self.mu*m_p)    
    
    def diP_diT(self,Rho,T):             # (Eqn. 7b) Eqn. 7a and b are both partial derivative of Eqn. 5, to be used in Eqn. 2a
        return Rho*k_B/(self.mu*m_p) + (4.*self.a*T**3./3.)

    ## main differentials ##
    def dRho_dR(self,Rho,T,R,L,M): # (Eqn. 2a)
        return -(G*M*Rho/R**2. + self.diP_diT(Rho,T)*self.dT_dR(Rho,T,R,L,M))/self.diP_diRho(Rho,T)

    def dT_dR(self,Rho,T,R,L,M): # (Eqn. 2b) energy transfer equation
        rad_transf = 3.*self.Opacity(Rho,T)*Rho*L/(16.*pi*self.a*c*T**3.*R**2.)
        conv = (1 - 1./self.gamma)*T*G*M*Rho/(self.Pressure(Rho,T)*R**2.)
        min_dT = min(rad_transf,conv)
        return -min_dT

    def dM_dR(self,Rho,R): # (Eqn. 2c)
        return 4*pi*R*R*Rho
        
    def dL_dR(self,Rho,T,R): # (Eqn. 2d)
        return self.dM_dR(Rho,R)*self.E_rate(Rho,T)
    
    def dtau_dR(self,Rho,T): # (Eqn. 2e) 
        return self.Opacity(Rho,T)*Rho

    def CoupledODEs(self,R,Rho,T,M,L): #odes vectorized
        #return np.array([1.0,self.dRho_dR(Rho,T,R,L,M),self.dT_dR(Rho,T,R,L,M),self.dM_dR(Rho,T),self.dL_dR(Rho,T,R)])
        return np.array([self.dRho_dR(Rho,T,R,L,M),self.dT_dR(Rho,T,R,L,M),self.dM_dR(Rho,T),self.dL_dR(Rho,T,R)])

    ## differential solvers ##
    def Set_R_Range(self,R_i,R_f,numpoints):
        self.R_range = np.linspace(R_i,R_f,numpoints)
        return
    """
    def Psi(self,Psi,R):
        Rho,T,M,L = Psi
        psiprime = [self.dRho_dR(Rho,T,R,L,M),self.dT_dR(Rho,T,R,L,M),self.dM_dR(Rho,T),self.dL_dR(Rho,T,R)]
        return np.array(psiprime)

    def PsiSolve(self,Rho_o,T_o,M_o,L_o):
        psi0 = Rho_o,T_o,M_o,L_o        
        return odeint(self.Psi,psi0,self.R_range).T
    """
    
    def RK4iter(self,R,init,h):
        k1 = self.CoupledODEs(R,*tuple(init))
        k2 = self.CoupledODEs(R,*tuple(init + k1*h/2.0))
        k3 = self.CoupledODEs(R,*tuple(init + k2*h/2.0))
        k4 = self.CoupledODEs(R,*tuple(init + k3*h))
        return R + h, init + (h/6.0)*(k1 + k2*2 + k3*2 + k4)


    def RK4solve(self,R,Rho,T,M,L,h=0.5):
        init = np.array([Rho,T,M,L])
        numiter = 10000
        self.Rs = np.ndarray(shape=(numiter,))
        self.Rhos = np.ndarray(shape=(numiter,))
        self.Ts = np.ndarray(shape=(numiter,))
        self.Ms = np.ndarray(shape=(numiter,))
        self.Ls = np.ndarray(shape=(numiter,))

        self.Rs[0] = R
        for i in range(numiter):
            self.Rhos[i] = init[0]
            self.Ts[i] = init[1]
            self.Ms[i] = init[2]
            self.Ls[i] = init[3]
            R,init = self.RK4iter(R,init,h)
            if i > 0:
                self.Rs[i] =  R
        return 

    def RKF45iter(self,R_o,init,h,tol):
        k1 = self.CoupledODEs(R_o,*tuple(init))
        k2 = self.CoupledODEs(R_o+h/4.0,*tuple(init + k1*h/4.0))
        k3 = self.CoupledODEs(R_o+h*3/8.0,*tuple(init + k1*h*3/32.0 + k2*h*9/32.0))
        k4 = self.CoupledODEs(R_o+h*12/13.0,*tuple(init + k1*h*1932/2197.0 - k2*h*7200/2197.0 + k3*h*7296/2197.0))
        k5 = self.CoupledODEs(R_o+h,*tuple(init + k1*h*439/216.0 - k2*h*8 + k3*h*3680/513.0 - k4*h*845/4104.0))
        k6 = self.CoupledODEs(R_o+h/2.0,*tuple(init - k1*h*8/27.0 + k2*h*2 - k3*h*3544/2565.0 + k4*h*1859/4104.0 - k5*h*11/40.0))

        yk1 = init + k1*25/216. + k3*1408/2565. + k4*2197/4101. - k5/5.
        zk1 = init + k1*16/135. + k3*6656/12825. + k4*28561/56430. - k5*9/50. + k6*2/55. 

        if 0 not in np.fabs(zk1-yk1):
            s = np.amin(tol*h*np.power(2*np.fabs(zk1-yk1),-1/4.))
        else:
            s = 1
        return R_o + s*h, yk1, s*h
    
    def RKF45solve(self,R,Rho,T,M,L,h=1e2,tol=1):
        init = np.array([Rho,T,M,L])
        numiter = 10000
        self.Rs = np.ndarray(shape=(numiter,))
        self.Rhos = np.ndarray(shape=(numiter,))
        self.Ts = np.ndarray(shape=(numiter,))
        self.Ms = np.ndarray(shape=(numiter,))
        self.Ls = np.ndarray(shape=(numiter,))

        self.Rs[0] = R    
        for i in range(numiter):
            self.Rhos[i] = init[0]
            self.Ts[i] = init[1]
            self.Ms[i] = init[2]
            self.Ls[i] = init[3]
            R,init,h = self.RKF45iter(R,init,h,tol=tol)
            if i > 0:
                self.Rs[i] = R
        return
    
    
    ## plot functions ##
    def plot(self,array1,array2,title='',xlabel='',ylabel=''):
        pplt.plot(array1,array2)
        pplt.title(title)
        pplt.xlabel(xlabel)
        pplt.ylabel(ylabel)
        #pplt.xscale('log')
        #pplt.yscale('log')
        pplt.grid()
        pplt.tight_layout()
        pplt.show()
        return
    

teststar = MS_Star(160000.0,15000000.0)
#teststar.Set_R_Range(1e-16,6e11,10000)
#arr = teststar.PsiSolve(teststar.rho_c,teststar.T_c,teststar.M_o,teststar.L_o)
#teststar.plot(teststar.R_range,arr[0])

#teststar.RKF45solve(1e-4,teststar.rho_c,teststar.T_c,teststar.M_o,teststar.L_o)

teststar.RK4solve(1.0,teststar.rho_c,teststar.T_c,teststar.M_o,teststar.L_o)
teststar.plot(teststar.Rs,teststar.Rhos,"Density")
teststar.plot(teststar.Rs,teststar.Ts,"Temp")
teststar.plot(teststar.Rs,teststar.Ms,"Mass")
teststar.plot(teststar.Rs,teststar.Ls,"Luminosity")
teststar.plot(teststar.Rs,teststar.Pressure(teststar.Rhos,teststar.Ts),"Pressure")
teststar.plot(teststar.Rs,teststar.dtau_dR(teststar.Rhos,teststar.Ts),"Dtau/dr")

