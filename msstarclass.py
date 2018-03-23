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
        self.R_sol = 695700000.0 #m
        self.T_sol = 5778.0 #K
        self.M_sol = 1.989e30 #kg
        self.L_sol = 3.828e26 #W
        self.c45 = np.array([0,1/5.0,3/10.0,4/5.0,8/9.0,1.0,1.0])
        self.a45 = np.array([[0,0,0,0,0,0],[1/5.,0,0,0,0,0],[3/40.,9/40.,0,0,0,0],[44/45.,-56/15.,32/9.,0,0,0],[19372/6561.,-25360/2187.,64448/6561.,-212/729.,0,0],[9017/3168.,-355/33.,46732/5247.,49/176.,-5103/18656.,0],[35/384.,0,500/1113.,125/192.,-2187/6784.,11/84.]])
        self.b45 = np.array([35/384.,0.0,500/1113.,125/192.,-2187/6784.,11/84.,0.0])
        self.bn45 = np.array([5179/57600.,0.0,7571/16695.,393/640.,-92097/339200.,187/2100.,1/40.])
        self.reject = False
        self.errold = 1
        self.addtoarray = True

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
        kappa_Hminus = 2.5e-32*(self.Z/2e-2)*np.sqrt(Rho/1e3)*np.power(T,9)
        kappa_ff = 1.0e24*(self.Z + 1e-4)*np.power(Rho/1e3,0.7)*np.power(T,-3.5)
        kappa_es = 0.02*(1.0 + self.X)
        max_kesff = np.maximum(kappa_es,kappa_ff)
        
        if type(T) == type(np.array([])):
            opacity_arr = np.ndarray(shape=T.shape)
            highT = np.greater(T, 1e4)
            lowT = np.logical_not(highT)
            opacity_arr[highT] = max_kesff[highT]
            opacity_arr[lowT] = np.power(np.power(kappa_Hminus,-1) + np.power(max_kesff,-1),-1)[lowT]
        else:
            if T > 1e4:
                return max_kesff
            else:
                return np.power(np.power(kappa_Hminus,-1) + np.power(max_kesff,-1),-1)

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
        return np.array([self.dRho_dR(Rho,T,R,L,M),self.dT_dR(Rho,T,R,L,M),self.dM_dR(Rho,T),self.dL_dR(Rho,T,R)])

    def TestCoupled(self,t,x,xprime,y,yprime,k=1.0,k2=2.0):
        return np.array([xprime,-k*x -k2*(x-y),yprime,-k2*(y-x)])

    def RKF45solvetest(self,x,x1,y,y1,k,t0,h=1e-2,atol=1e-6,rtol=1e-6):
        init = np.array([x,x1,y,y1])
        numiter = 10000
        self.t = np.ndarray(shape=(numiter,))
        self.x = np.ndarray(shape=(numiter,))
        self.x1 = np.ndarray(shape=(numiter,))
        self.y = np.ndarray(shape=(numiter,))
        self.y1 = np.ndarray(shape=(numiter,))

        for i in range(numiter):
            self.t[i] = t0
            self.x[i] = init[0]
            self.x1[i] = init[1]
            self.x[i] = init[2]
            self.x1[i] = init[3]
            t0,init,h = self.RKF45iter(self.TestCoupled,t0,init,h,atol=atol,rtol=rtol)
        return
        

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

    def RKF45iter(self,function,R_o,init,h,atol,rtol,maxscale=10.0,minscale=0.0001,beta=0.0):
        alpha = 0.2 - beta*0.75
        k1 = h*function(R_o,*tuple(init))
        k2 = h*function(R_o + h*self.c45[1],*tuple(init + k1*self.a45[1,0]))
        k3 = h*function(R_o + h*self.c45[2],*tuple(init + k1*self.a45[2,0] + k2*self.a45[2,1]))
        k4 = h*function(R_o + h*self.c45[3],*tuple(init + k1*self.a45[3,0] + k2*self.a45[3,1] + k3*self.a45[3,2]))
        k5 = h*function(R_o + h*self.c45[4],*tuple(init + k1*self.a45[4,0] + k2*self.a45[4,1] + k3*self.a45[4,2] + k4*self.a45[4,3]))
        k6 = h*function(R_o + h*self.c45[5],*tuple(init + k1*self.a45[5,0] + k2*self.a45[5,1] + k3*self.a45[5,2] + k4*self.a45[5,3] + k5*self.a45[5,4]))

        yk1 = init + self.b45[0]*k1 + self.b45[1]*k2 + self.b45[2]*k3 + self.b45[3]*k4 + self.b45[4]*k5 + self.b45[5]*k6
        ynk1 = init +  self.bn45[0]*k1 + self.bn45[1]*k2 + self.bn45[2]*k3 + self.bn45[3]*k4 + self.bn45[4]*k5 + self.bn45[5]*k6

        error = np.abs(yk1-ynk1)
        scale = atol + np.maximum(yk1,init)*rtol
        
        err = np.sqrt(np.sum(np.power(np.divide(error,scale),2))/error.size)

        if err <= 1:
            if err == 0:
                hscale = maxscale
            else:
                hscale = 0.9*err**(-alpha)*self.errold**(beta)
                if hscale < minscale:
                    hscale = minscale
                if hscale > maxscale:
                    hscale = maxscale
            if self.reject:
                hk1 = h*min(hscale,1.0)
            else:
                hk1 = h*hscale
            self.reject = False
            self.errold = max(err,1.0e-4)
            self.addtoarray = True
        else:
            hscale = max(0.9*err**(-alpha),minscale)
            hk1 = h*hscale
            self.reject = True
            self.addtoarray = False    
        
        return R_o + hk1, yk1, hk1
            
    def RKF45solve(self,R,Rho,T,h=1e1,atol=0,rtol=1e-6,maxscale=10.0,minscale=0.0001,beta=0.0):
        M_o = (4*pi/3.0)*(R)**3*Rho
        L_o = M_o*self.E_rate(Rho,T)
        init = np.array([Rho,T,M_o,L_o])
        numiter = 10000
        self.Rs = np.ndarray(shape=(numiter,))
        self.Rhos = np.ndarray(shape=(numiter,))
        self.Ts = np.ndarray(shape=(numiter,))
        self.Ms = np.ndarray(shape=(numiter,))
        self.Ls = np.ndarray(shape=(numiter,))

        for i in range(numiter):
            if init[2] > 1e3*self.M_sol:
                break
            dtau = self.Opacity(init[0],init[1])*init[0]*init[0]/(np.abs(self.dRho_dR(init[0],init[1],R,init[3],init[2])))
            if np.abs(dtau - 2/3.0) < rtol:
                print R
                print init[3]
            if self.addtoarray:
                self.Rs[i] = R
                self.Rhos[i] = init[0]
                self.Ts[i] = init[1]
                self.Ms[i] = init[2]
                self.Ls[i] = init[3]
                
            R,init,h = self.RKF45iter(self.CoupledODEs,R,init,h,atol=atol,rtol=rtol,maxscale=maxscale,minscale=minscale,beta=beta)
        return

    def coupledODE(self,R,y):
        return np.array([self.dRho_dR(y[0],y[1],R,y[2],y[3]),self.dT_dR(y[0],y[1],R,y[2],y[3]),self.dM_dR(y[0],y[1]),self.dL_dR(y[0],y[1],R)])

    def ODEsolve(self,R,Rho,T):
        M_o = (4*pi/3.0)*(R)**3*Rho
        L_o = M_o*self.E_rate(Rho,T)
        init = np.array([Rho,T,M_o,L_o])
        dydr = ode(self.coupledODE)
        #dydr.set_integrator('dop853',verbosity=1)
        dydr.set_initial_value(init,R)
        dr = 1e6
        self.Rs = np.array([])
        self.Rhos = np.array([])
        self.Ts = np.array([])
        self.Ls = np.array([])
        self.Ms = np.array([])

        self.Rhos = np.append(self.Rhos,Rho)
        self.Ts = np.append(self.Ts,T)
        self.Ls = np.append(self.Ls,L_o)
        self.Ms = np.append(self.Ms,M_o)
        self.Rs = np.append(self.Rs,R)
        while dydr.successful() and dydr.t < 1e11:
            temp = dydr.integrate(dydr.t+dr)
            self.Rhos = np.append(self.Rhos,temp[0])
            self.Ts = np.append(self.Ts,temp[1])
            self.Ls = np.append(self.Ls,temp[2])
            self.Ms = np.append(self.Ms,temp[3])
            self.Rs = np.append(self.Rs,dydr.t+dr)
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
    
rho_o = (5.0*G/(4*pi))*1.9891e30*1.9891e30/(695700000.)**4
print rho_o
teststar = MS_Star(rho_o,1.5e7)
#teststar.RKF45solvetest(0.,1.,0.,2.,1.0,0.0,h=1e-16,atol=1e-3,rtol=1e-3)
#teststar.plot(teststar.t,teststar.x,"Density")
#teststar.plot(teststar.t,teststar.x1,"Density")


#teststar.Set_R_Range(1e-16,6e11,10000)
#arr = teststar.PsiSolve(teststar.rho_c,teststar.T_c,teststar.M_o,teststar.L_o)
#teststar.plot(teststar.R_range,arr[0])

#teststar.ODEsolve(1e-16,teststar.rho_c,teststar.T_c)

teststar.RKF45solve(1e1,teststar.rho_c,teststar.T_c,maxscale=20.,minscale=0.1,atol=1e-6,rtol=1e-6,beta=0.04,h=1e-6)
print teststar.Rs
teststar.plot(teststar.Rs,teststar.Rhos,"Density")
teststar.plot(teststar.Rs,teststar.Ts,"Temp")
teststar.plot(teststar.Rs,teststar.Ms,"Mass")
teststar.plot(teststar.Rs,teststar.Ls,"Luminosity")
teststar.plot(teststar.Rs,teststar.Pressure(teststar.Rhos,teststar.Ts),"Pressure")



