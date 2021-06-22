import numpy as np


#SPSA ALGORITHM 
#FOLLOWING SPALL (1998) AND SPALL(1998) - P1 AND P2.
def SPSA(fun,inits,tol,maxitr,th_min,th_max,Noise,limitbounds,magchange):
    #NUMBER OF SUCCESSIVE ITERATIONS REQUIRED TO BE BELOW THRESHOLD
    nsuc = 5

    #SOME INFO
    p = inits.shape[0]

    #APPLY INITIAL CONDITIONS
    cost    = np.zeros((maxitr))
    th      = np.zeros((maxitr,p))
    th[0,:] = inits

    #FOR HIGH NOISE SMALLER a AND LARGER c ARE RECOMMENDED BY 
    #SPALL (1998) P2 COMPARED WITH A LOW NOISE SETTING. 
    #CHOOSE "A" TO BE 10% OR LESS THAN THE EXPECTED/MAX NUMBER OF ITERATIONS
    #CHOOSE a SUCH THAT a/(A+1)^alpha*ghat(th_0) IS THE MAX MAGNITUDE CHANGE DESIRED
    #SET c EQUAL TO THE STANDARD DEVIATION OF THE MEASUREMENT NOISE.
    #SET alpha,gamma FOLLOWING SPALL (1998) P2
    #A = 'NOT SET' #7
    alpha = 0.602     #0.602 or 1
    gamma = 0.101     #0.101 or 1./6.
    
    #DETERMINE THE c PARAMETER
    #THIS IS MY METHOD AND MAY NEED MODIFICATIONS
    #HERE I AM MAKING SOME PERTURBATIONS AND CALCULATING 
    #THE STANDARD DEVIATION
    if (Noise):
        nsmpl  = 5
        y_smpl = np.zeros(nsmpl)
        for i in range(0,nsmpl):
            y_smpl[i] = fun(th[0,:])
        c       = np.std(y_smpl)
        cost[0] = y_smpl[0]
        ####FLAG: HERE I AM RE-SETTING THE TOLERANCE BASED ON THE STANDARD
        ####      DEVIATION OF THE NOISE. I MADE THIS UP SO BE CAREFUL. 
        #tol  = c/10.
    #NO NOISE CASE. SMALL POSITIVE NUMBER
    else:
        c = 10**(-1)
        cost[0] = fun(th[0,:])
   
    # #DETERMINE THE a PARAMETER
    # ck        = c
    # mag_ghat  = 0.
    # nloop     = 20
    # for i in range(0,nloop):
        # delta     = 2.*np.round(np.random.rand(p)) - 1.
        # th_plus   = th[0,:] + ck*delta
        # th_minus  = th[0,:] - ck*delta
        # y_plus    = fun(th_plus)
        # y_minus   = fun(th_minus)
        # #COMMENTED OUT BASED ON THE Kandala et al. METHOD
        # ghat_0    = (y_plus - y_minus) #/(2.*ck*delta)
        # mag_ghat  = mag_ghat + np.abs(ghat_0)
    # mag_ghat = mag_ghat/nloop
    # #FOLLOWING SPALL et al.
    # #FOLLOWING Kandala et al. 
    # a = 2*np.pi/5. * c/mag_ghat

    ####FLAG: TESTING
    a = magchange

    print('Parameters')
    print('a=',str(a),'c=',str(c))
    print('alpha=',str(alpha),'gamma=',str(gamma))


    #START ITERATIONS
    norm_err  = np.zeros(maxitr)
    condition = True
    k = -1
    while (condition):
        k += 1

        ####FLAG: REMOVING THE "A" PARAMETER
        #ak = a/(k+A+1)**alpha
        ak = a/(k+1)**alpha
        ck = c/(k+1)**gamma
        #print('ak=',str(ak),'ck=',str(ck))
        delta     = 2.*np.round(np.random.rand(p)) - 1.
        th_plus   = th[k,:] + ck*delta
        th_minus  = th[k,:] - ck*delta
        y_plus    = fun(th_plus)
        y_minus   = fun(th_minus)       
        ghat      = (y_plus - y_minus)/(2.*ck*delta)
        th[k+1,:] = th[k,:] - ak*ghat
        #print(th[k+1,:])
        cost[k+1] = (y_plus + y_minus)/2.
        norm_err[k] = np.abs( (cost[k+1] - cost[k]) )
        if (limitbounds):
            for i in range(0,p):
                if (th[k+1,i]<th_min[i]):
                    th[k+1,i] = th_min[i]   
                elif (th[k+1,i]>th_max[i]):
                    th[k+1,i] = th_max[i]

        #TERMINATION CONDITIONS
        if (k>=nsuc):
            if (all(norm_err[k-nsuc:k+1] <= tol)):
                print('Convergence!')
                print(str(k), ' iterations required')
                condition = False
                #th_final = th[k+1,:]
                th_final = np.average(th[k-nsuc:k+2,:],axis=0)    
        if (k+1 == maxitr-1):
            print('Max number of iterations reached')
            condition = False
            th_final = np.average(th[k-nsuc:k+2,:],axis=0)
            #th_final = th[k+1,:]
            
    #print('Final Parameters')
    #print(th_final)
    return th_final,norm_err[:k+1],cost[:k+2],th[:k+2,:]