"""
Rewrite of Shoshannah's code as a class

@Zarif
"""
from astropy import constants
import numpy as np
import matplotlib.pyplot as pplt
from scipy.integrate import odeint,ode,trapz,cumtrapz
from scipy.interpolate import interp1d
from scipy import optimize
from time import time

#def constants
G = constants.G.value
c = constants.c.value
hbar = constants.hbar.value
m_e = constants.m_e.value
m_p = constants.m_p.value
k_B = constants.k_B.value
sigma_sb = constants.sigma_sb.value
pi = np.pi

np.seterr(all='warn')
class MS_Star:
    def __init__(self,rho_c,T_c): # initial setup 
        self.rho_c = rho_c
        self.T_c = T_c
        self.a = 4.*sigma_sb/c
        self.gamma = 5./3.
        self.X = 0.7
        self.Y = 0.29
        self.Z = 0.01
        self.mu = (2*self.X + 0.75*self.Y + 0.5*self.Z)**(-1.)
        self.R_sol = 695700000.0 #m
        self.T_sol = 5778.0 #K
        self.M_sol = 1.989e30 #kg
        self.L_sol = 3.828e26 #W
        self.Rho_sol = self.M_sol/((4*pi/3.0)*self.R_sol**3)
        self.c45 = np.array([0,1/5.0,3/10.0,4/5.0,8/9.0,1.0,1.0])
        self.a45 = np.array([[0,0,0,0,0,0],[1/5.,0,0,0,0,0],[3/40.,9/40.,0,0,0,0],[44/45.,-56/15.,32/9.,0,0,0],[19372/6561.,-25360/2187.,64448/6561.,-212/729.,0,0],[9017/3168.,-355/33.,46732/5247.,49/176.,-5103/18656.,0],[35/384.,0,500/1113.,125/192.,-2187/6784.,11/84.]])
        self.b45 = np.array([35/384.,0.0,500/1113.,125/192.,-2187/6784.,11/84.,0.0])
        self.bn45 = np.array([5179/57600.,0.0,7571/16695.,393/640.,-92097/339200.,187/2100.,1/40.])
        self.reject = False
        self.errold = 1.0
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
        kappa_es = 0.02*(1.0 + self.X)
        kappa_Hminus = 2.5e-32*(self.Z/2.0e-2)*np.sqrt(Rho/1.0e3)*np.power(T,9)
        kappa_ff = 1.0e24*(self.Z + 1.0e-4)*np.power(Rho/1.0e3,0.7)*np.power(T,-3.5)
        max_kesff = np.maximum(kappa_es,kappa_ff)
        #if type(Rho) == type(np.array([])):
        #    opacity_arr = np.ndarray(shape=T.shape)
        #    highT = np.greater(T, 1e4)
        #    lowT = np.logical_not(highT)
        #    opacity_arr[highT] = max_kesff[highT]
        #    opacity_arr[lowT] = np.power(np.power(kappa_Hminus,-1) + np.power(max_kesff,-1),-1)[lowT]
               
        #if T > 1.0e4:
        #    return max_kesff
        #else:
        return np.power(np.power(kappa_Hminus,-1) + np.power(max_kesff,-1),-1)

    def diP_diRho(self,Rho,T):           # (Eqn. 7a)
        a = ((3.*pi**2.)**(2./3.)/3)*(hbar**2./(m_e*m_p))*(Rho/m_p)**(2./3.)
        return a + k_B*T/(self.mu*m_p)    
    
    def diP_diT(self,Rho,T):             # (Eqn. 7b) Eqn. 7a and b are both partial derivative of Eqn. 5, to be used in Eqn. 2a
        return Rho*k_B/(self.mu*m_p) + (4.*self.a*T**3./3.)

    ## main differentials ##
    def dRho_dR(self,Rho,T,R,L,M): # (Eqn. 2a)
        return -(G*M*Rho/R**2. + self.diP_diT(Rho,T)*self.dT_dR(Rho,T,R,L,M))/self.diP_diRho(Rho,T)

    def dT_dR(self,Rho,T,R,L,M): # (Eqn. 2b) energy transfer equation
        rad_transf = 3.*self.Opacity(Rho,T)*Rho*L/(16.*pi*self.a*c*T**3.*R**2.)
        conv = (1 - 1./self.gamma)*T*G*M*Rho/(self.Pressure(Rho,T)*R**2.)
        min_dT = np.minimum(rad_transf,conv)
        return -min_dT

    def dM_dR(self,Rho,R): # (Eqn. 2c)
        return 4*pi*R*R*Rho
        
    def dL_dR(self,Rho,T,R): # (Eqn. 2d)
        return 4*pi*R*R*Rho*self.E_rate(Rho,T)
    
    def dtau_dR(self,Rho,T): # (Eqn. 2e) 
        return self.Opacity(Rho,T)*Rho

    def delta_tau(self,R,Rho,T,M,L):
        return self.Opacity(Rho,T)*np.power(Rho,2)/np.abs(self.dRho_dR(Rho,T,R,L,M))
        
    def CoupledODEs(self,R,init): #odes vectorized
        Rho,T,M,L,tau = np.abs(init[0]),np.abs(init[1]),np.abs(init[2]),np.abs(init[3]),init[4]
        #Rho,T,M,L = init[0],init[1],init[2],init[3]
        

        P_deg_const = ((3.*pi**2.)**(2./3.)/5.)*(hbar**2./m_e)
        P_IG_const = k_B/(self.mu*m_p)
        
        P_deg = P_deg_const*(Rho/m_p)**(5./3.)
        P_IG = Rho*T*P_IG_const
        P_PG = (1/3.)*self.a*T**4
        P = P_deg + P_IG + P_PG

        
        dP_drho = (P_deg_const*(5/3.)/(m_p))*(Rho/m_p)**(2./3.) + P_IG/Rho
        dP_dT = P_IG/T +  P_PG*4/T
        
        SA = 4*pi*R*R
        Grav = G*M*Rho/(R*R)
        kappa = self.Opacity(Rho,T)
        eps = self.E_rate(Rho,T)
        
        dT_dr = -1.0*np.minimum(3.*kappa*Rho*L/(16.*pi*c*self.a*T*T*T*R*R),(1.0-1.0/self.gamma)*(T/P)*Grav)
        dM_dr = SA*Rho
        dL_dr = SA*Rho*eps
        drho_dr = -(Grav + dP_dT*dT_dr)/dP_drho
        dtau_dr = kappa*Rho
        ans = np.array([drho_dr,dT_dr,dM_dr,dL_dr,dtau_dr])
        #ans = np.array([drho_dr,dT_dr,dM_dr,dL_dr])
        
        return ans
    
    def TestCoupled(self,t,x,xprime,y,yprime,k=1.0,k2=2.0):
        return np.array([xprime,-k*x -k2*(x-y),yprime,-k2*(y-x)])

    def RKF45solvetest(self,x,x1,y,y1,k,t0,h=1e-2,atol=1e-6,rtol=1e-6):
        init = np.array([x,x1,y,y1])
        numiter = 2
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
        
        k1 = h*function(R_o,init)
        k2 = h*function(R_o + h*self.c45[1],init + k1*self.a45[1,0])
        k3 = h*function(R_o + h*self.c45[2],init + k1*self.a45[2,0] + k2*self.a45[2,1])
        k4 = h*function(R_o + h*self.c45[3],init + k1*self.a45[3,0] + k2*self.a45[3,1] + k3*self.a45[3,2])
        k5 = h*function(R_o + h*self.c45[4],init + k1*self.a45[4,0] + k2*self.a45[4,1] + k3*self.a45[4,2] + k4*self.a45[4,3])
        k6 = h*function(R_o + h*self.c45[5],init + k1*self.a45[5,0] + k2*self.a45[5,1] + k3*self.a45[5,2] + k4*self.a45[5,3] + k5*self.a45[5,4])
        
        
        yk1 = init + self.b45[0]*k1 + self.b45[1]*k2 + self.b45[2]*k3 + self.b45[3]*k4 + self.b45[4]*k5 + self.b45[5]*k6
        ynk1 = init +  self.bn45[0]*k1 + self.bn45[1]*k2 + self.bn45[2]*k3 + self.bn45[3]*k4 + self.bn45[4]*k5 + self.bn45[5]*k6

        #print yk1
        #print ynk1
        delta = ynk1-yk1
        error = np.abs(delta)
        #print error
        scale = np.ndarray(shape=error.shape)
        for i in range(error.shape[0]):
            if error[i] < 1e-8:
                scale[i] = atol + np.abs(np.maximum(yk1[i],init[i]))*rtol
            else:
                scale[i] = np.abs(k1[i])*1.0e-5
#        print scale


        err = np.sqrt(np.sum(np.power(np.divide(error,scale),2))/float(error.size))
        print err
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
        if not self.reject:
            return R_o + hk1, ynk1, hk1
        else:
            return R_o, ynk1, hk1
          
    def RKF45solve(self,R,Rho,T,h=1e-5,atol=0,rtol=1e-6,maxscale=10.0,minscale=0.0001,beta=0.0):
        M_o = (4*pi/3.0)*(R)**3*Rho
        L_o = M_o*self.E_rate(Rho,T)
        init = np.array([Rho,T,M_o,L_o])
        numiter = 100
        self.Rs = np.ndarray(shape=(numiter,))
        self.Rhos = np.ndarray(shape=(numiter,))
        self.Ts = np.ndarray(shape=(numiter,))
        self.Ms = np.ndarray(shape=(numiter,))
        self.Ls = np.ndarray(shape=(numiter,))

        for i in range(numiter):
            if init[2] > 1e3*self.M_sol or init[0] < 1e4 or R > 1.0e9:
                print "break it up"
                break
            dtau = self.Opacity(init[0],init[1])*init[0]*init[0]/(np.abs(self.dRho_dR(init[0],init[1],R,init[3],init[2])))
            
            if np.abs(dtau - 2/3.0) < 1.0e-2:
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
        return np.array([self.dRho_dR(y[0],y[1],R,y[3],y[2]),self.dT_dR(y[0],y[1],R,y[3],y[2]),self.dM_dR(y[0],y[1]),self.dL_dR(y[0],y[1],R)])
        
    def ODEsolve(self,R,Rho,T,blackhole=False):
        self.Rho_c = Rho
        if blackhole:
            self.L_c = 1.3e31*(self.M_bh/self.M_sol)
            self.R_bub = np.sqrt(self.L_c*self.mu*m_p/(4*pi*c*k_B*T*Rho))
            R = self.R_bub
            M_o = (4*pi/3.0)*(R)**3*Rho + self.M_bh
            L_o = M_o*self.E_rate(Rho,T) + self.L_c
            rs = np.linspace(1.0e2,2.0e11,4000)

        else:
            M_o = (4*pi/3.0)*(R)**3*Rho
            L_o = M_o*self.E_rate(Rho,T)
            rs = np.linspace(1.0e5,2.0e10,2000)

        init = np.array([Rho,T,M_o,L_o,0.0])
            
        dydr = ode(self.CoupledODEs)
        dydr.set_integrator('dop853',nsteps=10000)#,rtol=1e-6, nsteps=1000, first_step=1e-8,beta=0.08,verbosity=True)

        self.Rs = np.array([])
        self.Rhos = np.array([])
        self.Ts = np.array([])        
        self.Ls = np.array([])        
        self.Ms = np.array([])        
        self.taus = np.array([]) 

        def solout(t,y):
            self.Rs = np.append(self.Rs,t)
            self.Rhos = np.append(self.Rhos,y[0])
            self.Ts = np.append(self.Ts,y[1])
            self.Ms = np.append(self.Ms,y[2])
            self.Ls = np.append(self.Ls,y[3])
            self.taus = np.append(self.taus,y[4])

        dydr.set_solout(solout)
        dydr.set_initial_value(init,R)

        #good_rs = np.array([])      
        #good_sol = np.array([])
        #dydr.integrate(rs[-1])
        
        T_check = 0
        rad_check = False
        for i in range(rs.size):
            dydr.integrate(rs[i])
            if dydr.successful():
                tau_check = self.delta_tau(dydr.t,dydr.y[0],dydr.y[1],dydr.y[2],dydr.y[3]) 
                if tau_check < 1. and tau_check > 0:
                    break
                if dydr.y[2] > 1.0e3*self.M_sol:
                    break
                self.Rs = np.append(self.Rs,dydr.t)
                self.Rhos = np.append(self.Rhos,dydr.y[0])
                self.Ts = np.append(self.Ts,dydr.y[1])
                self.Ms = np.append(self.Ms,dydr.y[2])
                self.Ls = np.append(self.Ls,dydr.y[3])
                self.taus = np.append(self.taus,dydr.y[4])
        
        sortr = np.argsort(self.Rs)
        self.Rs = self.Rs[sortr]
        self.Rhos = self.Rhos[sortr]
        self.Ts = self.Ts[sortr]        
        self.Ls = self.Ls[sortr]        
        self.Ms = self.Ms[sortr]       
        self.taus = self.taus[sortr]       

        #rad_arr = self.dT_dR(self.Rhos,self.Ts,self.Rs,self.Ls,self.Ms)
        #radi = np.where(np.logical_and(np.less(rad_arr, 1.0e-4),np.greater(self.Rs,1.0e3)))
        
        maxi = np.where(self.taus==np.amax(self.taus))[0][0]
        mini = np.abs(np.amax(self.taus[:maxi]) - self.taus[:maxi] - 2/3.0).argmin()


             
        R_surf = self.Rs[:maxi][mini]
        self.R_surf = R_surf
        self.Rs = self.Rs[:maxi][:mini+1]
        self.Rhos = self.Rhos[:maxi][:mini+1]
        self.Ts = self.Ts[:maxi][:mini+1]        
        self.Ls = self.Ls[:maxi][:mini+1]        
        self.Ms = self.Ms[:maxi][:mini+1]       
        self.taus = self.taus[:maxi][:mini+1]

        self.Rho_surf = self.Rhos[-1]
        self.T_surf = self.Ts[-1]        
        self.L_surf = self.Ls[-1]        
        self.M_surf = self.Ms[-1]       
        self.tau_surf = self.taus[-1]
        
        return (self.L_surf - 4*pi*sigma_sb*self.R_surf**2*self.T_surf**4)/np.sqrt(4*pi*sigma_sb*self.R_surf**2*self.T_surf**4*self.L_surf)
        
        """
        if rad_check:
            print
        if maxi != mini:
             R_surf = self.Rs[:maxi][mini]
             self.R_surf = R_surf
             self.Rs = self.Rs[:maxi][:mini+1]
             self.Rhos = self.Rhos[:maxi][:mini+1]
             self.Ts = self.Ts[:maxi][:mini+1]        
             self.Ls = self.Ls[:maxi][:mini+1]        
             self.Ms = self.Ms[:maxi][:mini+1]       
             self.taus = self.taus[:maxi][:mini+1]
 
             self.Rho_surf = self.Rhos[-1]
             self.T_surf = self.Ts[-1]        
             self.L_surf = self.Ls[-1]        
             self.M_surf = self.Ms[-1]       
             self.tau_surf = self.taus[-1]
             
        else:
            temp1 = (self.Ls - 4*pi*sigma_sb*np.power(self.Rs,2)*np.power(self.Ts,4))/np.sqrt(4*pi*sigma_sb*np.power(self.Rs,2)*np.power(self.Ts,4)*self.Ls)
            
            est_ind = np.abs(temp1).argmin()
            self.R_surf = self.Rs[est_ind] 
            self.Rs = self.Rs[:est_ind+1]
            self.Rhos = self.Rhos[:est_ind+1]
            self.Ts = self.Ts[:est_ind+1]        
            self.Ls = self.Ls[:est_ind+1]        
            self.Ms = self.Ms[:est_ind+1]       
            self.taus = self.taus[:est_ind+1]

            self.Rho_surf = self.Rhos[-1]
            self.T_surf = self.Ts[-1]        
            self.L_surf = self.Ls[-1]        
            self.M_surf = self.Ms[-1]       
            self.tau_surf = self.taus[-1]
        """
        return (self.L_surf - 4*pi*sigma_sb*self.R_surf**2*self.T_surf**4)/np.sqrt(4*pi*sigma_sb*self.R_surf**2*self.T_surf**4*self.L_surf)

        
    def FindOptimal(self,R_i,rho_i,Mbh=0.0,blackhole=False):
        self.R_init = R_i
        self.minvals = np.array([])
        self.rhovals = np.array([])
        if blackhole:
            self.M_bh = Mbh
            try:
                self.Rho_c = optimize.brentq(self.OptimalFuncBH,300,5000000)        
                #self.Rho_c = optimize.bisect(self.OptimalFunc,300,5000000,maxiter=100)
                #self.Rho_c = optimize.brentq(self.OptimalFunc,300,5000000)
            except:
                besti = np.abs(self.minvals).argmin()
                self.Rho_c = self.rhovals[besti]
            self.ODEsolve(self.R_init,self.Rho_c,self.T_c)
        else:
            try:
                self.Rho_c = optimize.brentq(self.OptimalFunc,300,5000000)        
            #self.Rho_c = optimize.bisect(self.OptimalFunc,300,5000000,maxiter=100)
            #self.Rho_c = optimize.brentq(self.OptimalFunc,300,5000000)
            except:
                besti = np.abs(self.minvals).argmin()
                self.Rho_c = self.rhovals[besti]
            self.ODEsolve(self.R_init,self.Rho_c,self.T_c)
        return
    
    def OptimalFunc(self,rho):
        minfunc = self.ODEsolve(self.R_init,rho,self.T_c)
        self.minvals = np.append(self.minvals,minfunc)
        self.rhovals = np.append(self.rhovals,rho)
        return minfunc
    
    def OptimalFuncBH(self,rho):
        minfunc = self.ODEsolve(self.R_init,rho,self.T_c,blackhole=True)
        self.minvals = np.append(self.minvals,minfunc)
        self.rhovals = np.append(self.rhovals,rho)
        return minfunc
    
    ## plot functions ##
    def plot(self,array1,array2,title='',xlabel='',ylabel=''):
        #pplt.scatter(array1,array2)
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
    
    def GetSurf(self):
        # "Radius","central Rho","Temp","Mass","Luminosity"
        return np.array([self.R_surf,self.Rho_c,self.T_surf,self.M_surf,self.L_surf])
    
rho_o = (5.0*G/(4*pi))*1.9891e30*1.9891e30/(695700000.)**4

teststar = MS_Star(1.0e5,1.7e12)
#teststar.FindOptimal(1.0e-16,teststar.rho_c)
teststar.FindOptimal(1.0e-16,teststar.rho_c,1.0e-9*teststar.M_sol,blackhole=True)
print teststar.GetSurf()
teststar.plot(teststar.Rs,teststar.Ts,"Temp")

    
#T_cs = np.linspace(1.0e6,1.1e11,5)
T_cs = np.linspace(1.0e7,1.0e8,10)
val_arr = np.empty((T_cs.size,5))
t1 = time()
for i in range(T_cs.size):
    teststar = MS_Star(1.0e5,T_cs[i])
    teststar.FindOptimal(1.0e-16,teststar.rho_c)
    val_arr[i] = teststar.GetSurf()
t2 = time()
print T_cs.size,"stars took", t2-t1,"s"
pplt.scatter(val_arr[:,2],val_arr[:,4]/teststar.L_sol)
pplt.show()

#teststar.RKF45solvetest(0.,1.,0.,2.,1.0,0.0,h=1e-16,atol=1e-3,rtol=1e-3)
#teststar.plot(teststar.t,teststar.x,"x")
#teststar.plot(teststar.t,teststar.x1,"dx/dt")

#teststar.Set_R_Range(1e-16,6e11,100)
#arr = teststar.PsiSolve(teststar.rho_c,teststar.T_c,teststar.M_o,teststar.L_o)
#teststar.plot(teststar.R_range,arr[0])

#a = teststar.ODEsolve(1e-16,teststar.rho_c,teststar.T_c)
#teststar.plot(teststar.Rs,teststar.Rhos,"Density")
#teststar.plot(teststar.Rs,teststar.Ts,"Temp")
#teststar.plot(teststar.Rs,teststar.Ms,"Mass")
#teststar.plot(teststar.Rs,teststar.Ls,"Luminosity")
#teststar.plot(teststar.Rs,teststar.dtau_dR(teststar.Rhos,teststar.Ts),"$d \\tau / dr$")
#teststar.plot(teststar.Rs,teststar.taus,"$\\tau $")

#dtau = teststar.dtau_dR(teststar.Rhos,teststar.Ts)
#taus = np.empty(shape=dtau.shape)
#for i in range(taus.shape[0]):
#    taus[i] = trapz(dtau[0:i],x=teststar.Rs[0:i])
#taus = taus - np.amin(taus)
#f = interp1d(teststar.Rs[-1],taus,kind='cubic')


#teststar.plot(teststar.Rs,taus,"$\\tau$")
#teststar.plot(teststar.Rs,teststar.delta_tau(teststar.Rs,teststar.Rhos,teststar.Ts,teststar.Ms,teststar.Ls),"$\\delta \\tau$")


"""
teststar.RKF45solve(1e-5,teststar.rho_c,teststar.T_c,maxscale=1.0e4,minscale=1.0e-4,atol=0.0,rtol=1.0e-4,beta=0.4/5,h=1.0e-1)
print teststar.Rhos.shape
teststar.plot(teststar.Rs,teststar.Rhos,"Density")
teststar.plot(teststar.Rs,teststar.Ts,"Temp")
teststar.plot(teststar.Rs,teststar.Ms,"Mass")
teststar.plot(teststar.Rs,teststar.Ls,"Luminosity")
teststar.plot(teststar.Rs,teststar.Pressure(teststar.Rhos,teststar.Ts),"Pressure")
"""


