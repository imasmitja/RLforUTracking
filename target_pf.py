#!/usr/bin/env python
# license removed for brevity
# -*- coding: utf-8 -*-
"""
Created on March 029 2020
@author: Ivan Masmitja Rusinol
Project: AIforUTracking
"""

import numpy as np
import random
import time
import sys

SOUND_SPEED = 1500.

#############################################################
## Particle Filter
############################################################
#For modeling the target we will use the TargetClass with the following attributes 
#and functions:
class ParticleFilter(object):
    """ Class for the Particle Filter """
 
    def __init__(self,std_range,init_velocity,dimx,particle_number = 6000, method = 'range', max_pf_range = 250):
 
        self.std_range = std_range
        self.init_velocity = init_velocity 
        self.x = np.zeros([particle_number,dimx])
        self.oldx = np.zeros([particle_number,dimx])
        self.particle_number = particle_number
        
        self._x = np.zeros([dimx])
       
        # target's noise
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0
        self.velocity_noise = 0.0
        
        # time interval
        self.dimx=dimx
        
        self._velocity = 0
        self._orientation = 0
        
        #Weights
        self.w = np.ones(particle_number)
        
        #Covariance of the result
        self.covariance_vals = [0.02,0.02]
        self.covariance_theta = 0.
        
        #Flag to initialize the particles
        self.initialized = False
        
        #save actual data as a old to be used on TDOA method
        self.measurement_old = 0
        self.dist_all_old = np.zeros(particle_number)
        self.w_old = self.w
        self.observer_old = np.array([0,0,0,0])
        
        self.method = method
        #covariance matrix of final estimation
        self.cov_matrix = np.ones([2,2])
        
        #maximum target range
        self.max_pf_range = max_pf_range
        
        
    def target_estimation(self):
        """ Calculate the mean error of the system
        :param r: current target object
        :param p: particle set
        :return mean error of the system
        """
        #8- Target prediction (we predict the best estimation for target's position = mean of all particles)
        sumx = 0.0
        sumy = 0.0
        sumvx = 0.0
        sumvy = 0.0

        method = 2
        if method == 1:
            for i in range(self.particle_number):
               sumx += self.x[i][0]
               sumy += self.x[i][2]
               sumvx += self.x[i][1]
               sumvy += self.x[i][3]
            self._x = np.array([sumx, sumvx, sumy, sumvy])/self.particle_number
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
        if method == 2:
            for i in range(self.particle_number):
               sumx += self.x[i][0]*self.w[i]
               sumy += self.x[i][2]*self.w[i]
               sumvx += self.x[i][1]*self.w[i]
               sumvy += self.x[i][3]*self.w[i]
            self._x = np.array([sumx, sumvx, sumy, sumvy])/np.sum(self.w)
            
            # #new approach to find the colosest particle to the mean
            # x_pos = np.where(abs(self.x.T[0]-self._x[0]) == np.amin(abs(self.x.T[0]-self._x[0])))[0][0]
            # y_pos = np.where(abs(self.x.T[2]-self._x[2]) == np.amin(abs(self.x.T[2]-self._x[2])))[0][0]
            # x_mean = (self.x.T[0][x_pos] + self.x.T[0][y_pos])/2.
            # y_mean = (self.x.T[2][x_pos] + self.x.T[2][y_pos])/2.
            # self._x[0] = x_mean
            # self._x[2] = y_mean
            
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
        #finally the covariance matrix is computed. 
        #http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
        xarray = self.x.T[0]
        yarray = self.x.T[2]
        self.cov_matrix = np.cov(xarray, yarray)
        return

    def init_particles(self,position,slantrange):
    	
        for i in range(self.particle_number):
            #Random distribution with circle shape
            t = 2*np.pi*np.random.rand()
            if self.method == 'area':
                r = np.random.rand()*self.max_pf_range*2 - self.max_pf_range
            else:
                r = np.random.rand()*self.std_range*2 - self.std_range + slantrange
            
            self.x[i][0] = r*np.cos(t)+position[0]
            self.x[i][2] = r*np.sin(t)+position[2]
            #target's orientation
            orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
            # target's velocity 
            v = random.gauss(self.init_velocity, self.init_velocity/2)  
            self.x[i][1] = np.cos(orientation)*v
            self.x[i][3] = np.sin(orientation)*v
        self.target_estimation()
        self.initialized = True
        # print('WARNING: Particles initialized')
        return
    
    #Noise parameters can be set by:
    def set_noise(self, forward_noise, turn_noise, sense_noise, velocity_noise):
        """ Set the noise parameters, changing them is often useful in particle filters
        :param new_forward_noise: new noise value for the forward movement
        :param new_turn_noise:    new noise value for the turn
        :param new_sense_noise:  new noise value for the sensing
        """
        # target's noise
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise
        self.velocity_noise = velocity_noise

    #Move particles acording to its motion
    def predict(self,dt):
        """ Perform target's turn and move
        :param turn:    turn command
        :param forward: forward command
        :return target's state after the move
        """
        gaussnoise = False
        for i in range(self.particle_number):
            # turn, and add randomness to the turning command
            turn = np.arctan2(self.x[i][3],self.x[i][1])
            if gaussnoise == True:
                orientation = turn + random.gauss(0.0, self.turn_noise)
            else:
                orientation = turn +  np.random.rand()*self.turn_noise*2 -self.turn_noise
            orientation %= 2 * np.pi
         
            # move, and add randomness to the motion command
            velocity = np.sqrt(self.x[i][1]**2+self.x[i][3]**2)
            forward = velocity*dt
            if gaussnoise == True:
                dist = float(forward) + random.gauss(0.0, self.forward_noise)
            else:
                dist = float(forward) + np.random.rand()*self.forward_noise*2 - self.forward_noise
            self.x[i][0] = self.x[i][0] + (np.cos(orientation) * dist)
            self.x[i][2] = self.x[i][2] + (np.sin(orientation) * dist)
            if gaussnoise == True:
                newvelocity = velocity + random.gauss(0.0, self.velocity_noise)
            else:
                newvelocity = velocity + np.random.rand()*self.velocity_noise*2 - self.velocity_noise
            if newvelocity < 0:
                newvelocity = 0
            self.x[i][1] = np.cos(orientation) * newvelocity
            self.x[i][3] = np.sin(orientation) * newvelocity
        return 

    #To calculate Gaussian probability:
    @staticmethod
    def gaussian(self,mu_old,mu, sigma, z_old,z,inc_observer):
        """ calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        :param mu:    distance to the landmark
        :param sigma: standard deviation
        :param x:     distance to the landmark measured by the target
        :return gaussian value
        """
        if self.method == 'area':
            sigma = 1. #was 5
            particlesRange = self.max_pf_range 
            # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma in a filled circle shape
            # We use the Cauchy distribution (https://en.wikipedia.org/wiki/Cauchy_distribution)
            if z != -1: #a new ping is received -> #all particles outside the tagrange have a small weight; #all particles inside the tagrange have a big weight
                return (1/2.)-(1/np.pi)*np.arctan((mu-particlesRange)/sigma)
            else: #no new ping is received -> #all particles outside the tagrange have a big weight; #all particles inside the tagrange have a small weight
                sigma = 40.
                return (1/2.)+(1/np.pi)*np.arctan((mu-particlesRange)/sigma)
        else:
            # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
            return np.exp(- ((mu - z) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))
    
    #The next function we will need to assign a weight to each particle according to 
    #the current measurement. See the text below for more details. It uses effectively a 
    #Gaussian that measures how far away the predicted measurements would be from the 
    #actual measurements. Note that for this function you should take care of measurement 
    #noise to prevent division by zero. Such checks are skipped here to keep the code 
    #as short and compact as possible.
    def measurement_prob(self, measurement,observer):
        """ Calculate the measurement probability: how likely a measurement should be
        :param measurement: current measurement
        :return probability
        """
        #The closer a particle to a correct position, the more likely will be the set of 
            #measurements given this position. The mismatch of the actual measurement and the 
            #predicted measurement leads to a so-called importance weight. It tells us how important 
            #that specific particle is. The larger the weight, the more important it is. According 
            #to this each of our particles in the list will have a different weight depending on 
            #a specific target measurement. Some will look very plausible, others might look 
            #very implausible.           
        dist_all = []
        equal = 0
        for i in range(self.particle_number):
            dist = np.sqrt((self.x[i][0] - observer[0])**2 + (self.x[i][2] - observer[2])**2)
            dist_old = np.sqrt((self.x[i][0] - self.observer_old[0])**2 + (self.x[i][2] - self.observer_old[2])**2)
            inc_observer = np.sqrt((observer[0] - self.observer_old[0])**2 + (observer[2] - self.observer_old[2])**2)
            self.w[i] = self.gaussian(self,dist_old,dist, self.sense_noise, self.measurement_old,measurement,inc_observer)
            inc_mu = (self.dist_all_old[i]-dist)
            inc_z = (self.measurement_old-measurement)
            if (inc_mu >= 0 and inc_z >= 0) or (inc_mu < 0 and inc_z < 0):
                equal +=1
            dist_all.append(dist)
            
        #save actual data as a old to be used on TDOA method
        self.measurement_old = measurement
        self.dist_all_old = np.array(dist_all)
        self.w_old=self.w
        self.observer_old = observer
        return 
    
    def resampling(self,z):
        #After that we let these particles survive randomly, but the probability of survival 
            #will be proportional to the weights.
            #The final step of the particle filter algorithm consists in sampling particles from 
            #the list p with a probability which is proportional to its corresponding w value. 
            #Particles in p having a large weight in w should be drawn more frequently than the 
            #ones with a small value
            #Here is a pseudo-code of the resampling step:
            #while w[index] < beta:
            #    beta = beta - w[index]
            #    index = index + 1
            #    select p[index]
                        
        #method = 2 #NO compound method
        #method = 3.2 #compound method
        
        if self._x[0] == 0 and self._x[2] == 0:
            method = 2
        else:
            method = 3 #compound method presented in OCEANS'18 Kobe
        
        if method == 1:   
            # 4- resampling with a sample probability proportional
            # to the importance weight
            p3 = np.zeros([self.particle_number,self.dimx])
            index = int(np.random.random() * self.particle_number)
            beta = 0.0
            mw = max(self.w)
            for i in range(self.particle_number):
                beta += np.random.random() * 2.0 * mw
                while beta > self.w[index]:
                    beta -= self.w[index]
                    index = (index + 1) % self.particle_number
                p3[i]=self.x[index]
            self.x = p3
            return
        if method == 2:
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            # Systematic Resampling
            p3 = np.zeros([self.particle_number,self.dimx])
            ci = np.zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = np.random.random()/self.particle_number
            i = 0
            for j in range(self.particle_number):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./self.particle_number
            self.x = p3
            return
        if method == 3: #this mehtod works ok and was presented in OCEANS Kobe 2018
            # Systematic Resampling + random resampling
            if self.particle_number == 10000:
                ratio = 640 #160 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 6000:
                ratio = 400 #100 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 3000:
                ratio = 200 #50 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 1000:
                ratio = 120 #15 works ok; ratio=10 is ok for statik targets
            else:
                ratio = 50 #50 works ok; ratio=10 is ok for statik targets
            radii = 0.2 #50 works ok
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            p3 = np.zeros([self.particle_number,self.dimx])
            ci = np.zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = random.random()/(self.particle_number-ratio)
            i = 0
            for j in range((self.particle_number-ratio)):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./(self.particle_number-ratio)
                
            for i in range(ratio):
                #Random distribution with circle shape
                aux=np.zeros(4)
                t = 2*np.pi*np.random.rand()
                r = np.random.rand()*radii
                aux[0] = r*np.cos(t)+self._x[0]
                aux[2] = r*np.sin(t)+self._x[2]
                #target's orientation
                orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
                # target's velocity 
                v = random.gauss(self.init_velocity, self.init_velocity/2.)  
                aux[1] = np.cos(orientation)*v
                aux[3] = np.sin(orientation)*v
                p3[j+i+1]= aux
                self.w[j+i+1] = 1./(self.particle_number/3.)
            self.x = p3
            return
        if method == 3.2: 
            #this mehtod is a modification used in TAG-Only tracking, is similar than the method presented in OCEANS Kobe 2018
            #the main difference is that the random resampling is centred over the WG position instead of the Target estimation
            # Systematic Resampling + random resampling
            ratio = 50 #50 works ok
            radii = self.max_pf_range #50 works ok
            
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            p3 = np.zeros([self.particle_number,self.dimx])
            ci = np.zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = np.random.random()/(self.particle_number-ratio)
            i = 0
            for j in range((self.particle_number-ratio)):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./(self.particle_number-ratio)
                
            for i in range(ratio):
                i += 1
                #Random distribution with circle shape
                aux=np.zeros(4)
                t = 2*np.pi*np.random.rand()
                r = np.random.rand()*radii
                aux[0] = r*np.cos(t)+self.observer_old[0]
                aux[2] = r*np.sin(t)+self.observer_old[2]
                #target's orientation
                orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
                # target's velocity 
                v = random.gauss(self.init_velocity, self.init_velocity/2.)  
                aux[1] = np.cos(orientation)*v
                aux[3] = np.sin(orientation)*v
                p3[j+i]= aux
                self.w[j+i] = 1/10000.
            self.x = p3
            return
    
    
    #6- It computes the average error of each particle relative to the target pose. We call 
            #this function at the end of each iteration:
            # here we get a set of co-located particles   
    #At every iteration we want to see the overall quality of the solution, for this 
    #we will use the following function:
    def evaluation(self,observer,z,max_error=50):
        """ Calculate the mean error of the system
        :param r: current target object
        :param p: particle set
        :return mean error of the system
        """
        if self.method != 'area':
        
            #Evaluate the distance error
            sum2 = 0.0
            for i in range(self.particle_number):
                # Calculate the mean error of the system between Landmark (WG) and particle set
                dx = (self.x[i][0] - observer[0])
                dy = (self.x[i][2] - observer[2])
                err = np.sqrt(dx**2 + dy**2)
                sum2 += err
            # print('Evaluation -> distance error: ',abs(sum2/self.particle_number - z))
            
            #Evaluate the covariance matrix
            err_x = self.x.T[0]-self._x[0]
            err_y = self.x.T[2]-self._x[2]
            cov = np.cov(err_x,err_y)
            # Compute eigenvalues and associated eigenvectors
            vals, vecs = np.linalg.eig(cov)
            confidence_int = 2.326**2
            self.covariance_vals = np.sqrt(vals) * confidence_int
            # Compute tilt of ellipse using first eigenvector
            vec_x, vec_y = vecs[:,0]
            self.covariance_theta = np.arctan2(vec_y,vec_x)
            # print('Evaluation -> covariance (CI of 98): %.2f m(x) %.2f m(y) %.2f deg'%(self.covariance_vals[0],self.covariance_vals[1],np.degrees(self.covariance_theta)))
            # print('Evaluation -> covariance (CI of 98): %.2f '%(np.sqrt(self.covariance_vals[0]**2+self.covariance_vals[1]**2)))
            # print('errorPF=',abs(sum2/self.particle_number - z))
            if abs(sum2/self.particle_number - z) > max_error and np.sqrt(self.covariance_vals[0]**2+self.covariance_vals[1]**2) < 5.:
            	self.initialized = False
        else:
            if np.max(self.w) < 0.1:
                self.initialized = False
            #Compute maximum particle dispersion:
            max_dispersion = np.sqrt((np.max(self.x.T[0])-np.min(self.x.T[0]))**2+(np.max(self.x.T[2])-np.min(self.x.T[2]))**2)     
        return 


##########################################################################################################
##############################                    TARGET CLASS   ##########################################
###########################################################################################################
class Target(object):
    
    def __init__(self,method='range',max_pf_range=250):
        #Target parameters
        self.method = method
        
        ############## PF initialization #######################################################################
        #Our particle filter will maintain a set of n random guesses (particles) where 
        #the target might be. Each guess (or particle) is a vector containing [x,vx,y,vy]
        # create a set of particles
        # sense_noise is not used in area-only
        # self.pf = ParticleFilter(std_range=.01,init_velocity=.001,dimx=4,particle_number=6000,method=method,max_pf_range=max_pf_range)
        # self.pf.set_noise(forward_noise = 0.0001, turn_noise = 0.1, sense_noise=.05, velocity_noise = 0.0001)
        
        # self.pf = ParticleFilter(std_range=.005,init_velocity=.001,dimx=4,particle_number=1000,method=method,max_pf_range=max_pf_range)
        # self.pf.set_noise(forward_noise = 0.01, turn_noise = 0.1, sense_noise=.09, velocity_noise = 0.0001)
        
        self.pf = ParticleFilter(std_range=.02,init_velocity=.2,dimx=4,particle_number=1000,method=method,max_pf_range=max_pf_range)
        self.pf.set_noise(forward_noise = 0.01, turn_noise = 0.1, sense_noise=.005, velocity_noise = 0.01)
            
            
        self.pfxs = [0.,0.,0.,0.]
        
        #############LS initialization###########################################################################
        self.lsxs=[]
        self.eastingpoints_LS=[]
        self.northingpoints_LS=[]
        self.Plsu=np.array([])
        self.allz=[]
    
    #############################################################################################
    ####            Particle Filter Algorithm  (PF)                                             ##         
    #############################################################################################                               
    def updatePF(self,dt,new_range,z,myobserver,update=True):
        max_error = 0.1
        if update == True:
                  
            # Initialize the particles if needed
            if self.pf.initialized == False:
                self.pf.init_particles(position=myobserver, slantrange=z)
                
            #we save the current particle positions to plot as the old ones
            self.pf.oldx = self.pf.x.copy() 
            
            # Predict step (move all particles)
            self.pf.predict(dt)
            
            # Update step (weight and resample)
            if new_range == True:     
                # Update the weiths according its probability
                self.pf.measurement_prob(measurement=z,observer=myobserver)      
                #Resampling        
                self.pf.resampling(z)
                # Calculate the avarage error. If it's too big the particle filter is initialized                    
                self.pf.evaluation(observer=myobserver,z=z,max_error=max_error)    
            # We compute the average of all particles to fint the target
            self.pf.target_estimation()
        #Save position
        self.pfxs = self.pf._x.copy()
        return True

    #############################################################################################
    ####             Least Squares Algorithm  (LS)                                             ##         
    #############################################################################################
    def updateLS(self,dt,new_range,z,myobserver):
        num_ls_points_used = 30
        #Propagate current target state estimate
        if new_range == True:
            self.allz.append(z)
            self.eastingpoints_LS.append(myobserver[0])
            self.northingpoints_LS.append(myobserver[2])
        numpoints = len(self.eastingpoints_LS)
        if numpoints > 3:
            #Unconstrained Least Squares (LS-U) algorithm 2D
            #/P_LS-U = N0* = N(A^T A)^-1 A^T b
            #where:
            P=np.matrix([self.eastingpoints_LS[-num_ls_points_used:],self.northingpoints_LS[-num_ls_points_used:]])
            # N is:
            N = np.concatenate((np.identity(2),np.matrix([np.zeros(2)]).T),axis=1)
            # A is:
            num = len(self.eastingpoints_LS[-num_ls_points_used:])
            A = np.concatenate((2*P.T,np.matrix([np.zeros(num)]).T-1),axis=1)
            # b is:
            b = np.matrix([np.diag(P.T*P)-np.array(self.allz[-num_ls_points_used:])*np.array(self.allz[-num_ls_points_used:])]).T
            # Then using the formula "/P_LS-U" the position of the target is:
            try:
                self.Plsu = N*(A.T*A).I*A.T*b
            except:
                print('WARNING: LS singular matrix')
                try:
                    self.Plsu = N*(A.T*A+1e-6).I*A.T*b
                except:
                    pass
            # Finally we calculate the depth as follows
#                r=np.matrix(np.power(allz,2)).T
#                a=np.matrix(np.power(Plsu[0]-eastingpoints_LS,2)).T
#                b=np.matrix(np.power(Plsu[1]-northingpoints_LS,2)).T
#                depth = np.sqrt(np.abs(r-a-b))
#                depth = np.mean(depth)
#                Plsu = np.concatenate((Plsu.T,np.matrix(depth)),axis=1).T
            #add offset
#                Plsu[0] = Plsu[0] + t_position.item(0)
#                Plsu[1] = Plsu[1] + t_position.item(1)
#                eastingpoints = eastingpoints + t_position.item(0)
#                northingpoints = northingpoints + t_position.item(1)
            #Error in 'm'
#                error = np.concatenate((t_position.T,np.matrix(simdepth)),axis=1).T - Plsu
#                allerror = np.append(allerror,error,axis=1)

        #Compute MAP orientation and save position
        try:
            ls_orientation = np.arctan2(self.Plsu[1]-self.lsxs[-1][2],self.Plsu[1]-self.lsxs[-1][0])
        except IndexError:
            ls_orientation = 0
        try:
            ls_velocity = np.array([(self.Plsu[0]-self.lsxs[-1][0])/dt,(self.Plsu[1]-self.lsxs[-1][1])/dt])
        except IndexError:
            ls_velocity = np.array([0,0])
        try:
            ls_position = np.array([self.Plsu.item(0),ls_velocity.item(0),self.Plsu.item(1),ls_velocity.item(1),ls_orientation.item(0)])
        except IndexError:
            ls_position = np.array([myobserver[0],ls_velocity[0],myobserver[2],ls_velocity[1],ls_orientation])
        self.lsxs.append(ls_position)
        return True
