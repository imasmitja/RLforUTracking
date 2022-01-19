import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from target_pf import Target
from utilities import random_levy


class Scenario(BaseScenario):
    
    def make_world(self, num_agents=3, num_landmarks=3, landmark_depth=15., landmark_movable = False, movement='linear', pf_method = False, rew_err_th=0.0003, rew_dis_th=0.3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.04
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks*2)]
        for i, landmark in enumerate(world.landmarks):
            if i < num_landmarks:
                landmark.name = 'landmark %d' % i
                landmark.collide = False
                landmark.movable = landmark_movable
            else:
                landmark.name = 'landmark_estimation %d' % (i-num_landmarks)
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.002
                
        # make initial conditions
        world.cov = np.ones(num_landmarks)/30.
        world.error = np.ones(num_landmarks)
        self.reset_world(world)
        
        #benchmark variables
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0
        #Scenario initial conditions
        self.landmark_depth = landmark_depth
        self.ra = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
        self.movement = movement
        self.pf_method = pf_method
        self.rew_err_th = rew_err_th
        self.rew_dis_th = rew_dis_th
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                landmark.color = np.array([0.25, 0.25, 0.25])
            else:
                landmark.color = np.array([0.55, 0.0, 0.0])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1., 1., world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_vel_old = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                dis = np.random.uniform(0.04, 1.)
                rad = np.random.uniform(0, np.pi*2)
                landmark.state.p_pos = agent.state.p_pos + np.array([np.cos(rad),np.sin(rad)])*dis
                landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.state.p_pos = world.agents[0].state.p_pos
                landmark.state.p_vel = np.zeros(world.dim_p)
        #Initailize the landmark estimated positions
        world.landmarks_estimated = [Target() for i in range(world.num_landmarks)]
        
        #benchmark variables
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0
        

    def benchmark_data(self, agent, world):
        landmarks_real_p = []
        for i in range(world.num_landmarks):
            landmarks_real_p.append(world.landmarks[i].state.p_pos)
        # return (rew, collisions, min_dists, occupied_landmarks,landmarks_real_p)
        return(world.error,landmarks_real_p, self.agent_outofworld, self.landmark_collision, self.agent_collision)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    done_state = False
    def reward(self, agent, world):
        global done_state
        done_state = False
        # Agents are rewarded based on landmarks_estimated covariance_vals, penalized for collisions
        rew = 0.
        
        for i,l in enumerate(world.landmarks_estimated):
            # reward as a function of covariance matrix (PF)
            if self.pf_method == True:
                world.cov[i] = np.sqrt((l.pf.covariance_vals[0])**2+(l.pf.covariance_vals[1])**2)/10.
                # rew -= world.cov[i]/100
            # reward as a function of the distance error between the landmark and its estimation (PF or LS)
            if self.pf_method == True:
                world.error[i] = np.sqrt((l.pfxs[0]-world.landmarks[i].state.p_pos[0])**2+(l.pfxs[2]-world.landmarks[i].state.p_pos[1])**2) #Error from PF
            else:
                world.error[i] = np.sqrt((l.lsxs[-1][0]-world.landmarks[i].state.p_pos[0])**2+(l.lsxs[-1][2]-world.landmarks[i].state.p_pos[1])**2) #Error from LS
            
            # rew -= 0.01*(world.error[i])**2
            # rew += 0.01*(0.004-world.error[i])
            
            if world.error[i]<self.rew_err_th:
                rew += 1.
            else:
                rew += 0.01*(0.004-world.error[i])
        
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks[:-world.num_landmarks]]
        
        #For Test 11
        
        for dist in dists:
            # rew += 10*np.exp(-1/2*(dist-0.1)**2/0.1)-5
            # rew += 0.01*(np.exp(-10*(0.2-dist)**2)-0.9) #test 123
            
            # rew -= 0.01*(0.1-dist)**2  #test 121 and 122
            
            if dist < self.rew_dis_th: #other tests
                rew += 1.
                # rew += 0.1*(0.5-0.1)
            else:
                rew += 0.01*(0.7-dist)
                
        if min(dists) > 1.5: #agent outside the world
            rew -= 100
            done_state = True
            self.agent_outofworld += 1
        if min(dists) < 0.02: #is collision
            rew -= 1.
            done_state = True
            self.landmark_collision += 1
            
            
        # reward based on increment of action (from paper ieeeAccess) done in test 25      
        
        #compute the angle between the old direction and the new direction 
        # rew -= 0.0001*abs(agent.state.p_vel.item(0))
        
        #old methods
        # inc_action = agent.state.p_vel_old - agent.state.p_vel
        # rew -= 0.001*np.sqrt(inc_action[0]**2+inc_action[1]**2)
        # agent.state.p_vel_old = agent.state.p_vel + 0.
        # if np.all(np.sign(agent.state.p_vel_old) == np.sign(agent.state.p_vel)) == True:
        #     rew = 0.01
            
            
        
            
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 10.
                    self.agent_collision += 1
                    done_state = True
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_range = []
        for i, entity in enumerate(world.landmarks):
            if i < world.num_landmarks: 
                #Update the landmarks_estiamted position using Particle Fileter
                #1:Compute radius between the agent and each landmark
                slant_range = np.sqrt(((entity.state.p_pos - agent.state.p_pos)[0])**2+((entity.state.p_pos - agent.state.p_pos)[1])**2)
                target_depth = self.landmark_depth/1000. #normalize the target depth
                slant_range = np.sqrt(slant_range**2+target_depth**2) #add target depth to the range measurement
                # Add some systematic error in the measured range
                slant_range *= 1.01 # where 0.99 = 1% of sound speed difference = 1495 m/s
                # Add some noise in the measured range
                slant_range += np.random.uniform(-0.001, +0.001)
                # Return to a planar range
                slant_range = np.sqrt(abs(slant_range**2-target_depth**2))
                #2:Update the PF
                if self.pf_method == True:
                    world.landmarks_estimated[i].updatePF(dt=0.04, new_range=True, z=slant_range, myobserver=[agent.state.p_pos[0],0.,agent.state.p_pos[1],0.], update=True)
                else:
                    #2b: Update the LS
                    world.landmarks_estimated[i].updateLS(dt=0.04, new_range=True, z=slant_range, myobserver=[agent.state.p_pos[0],0.,agent.state.p_pos[1],0.])
                # Traditional plot
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(5,5))
                # plt.plot(world.landmarks_estimated[i].pf._x[0],world.landmarks_estimated[i].pf._x[2], 'r^', ms=20)
                # plt.plot(world.landmarks_estimated[i].pf.x.T[0],world.landmarks_estimated[i].pf.x.T[2], 'ro', ms=5, alpha=0.3)
                # plt.plot(world.landmarks[0].state.p_pos[0],world.landmarks[0].state.p_pos[1], 'ko', ms=6, alpha = 0.5)
                # plt.plot(world.agents[0].state.p_pos[0],world.agents[0].state.p_pos[1], 'bo', ms=6, alpha = 0.5)
                # plt.xlim(-1,1)
                # plt.ylim(-1,1)
                # plt.show()
                #3:Publish the new estimated position
                if self.pf_method == True:
                    world.landmarks[i+world.num_landmarks].state.p_pos = [world.landmarks_estimated[i].pfxs[0],world.landmarks_estimated[i].pfxs[2]] #Using PF
                else:
                    world.landmarks[i+world.num_landmarks].state.p_pos = [world.landmarks_estimated[i].lsxs[-1][0],world.landmarks_estimated[i].lsxs[-1][2]] #Using LS
                
                #Append the position of the landmark to generate the observation state
                #Using the true landmark position
                # entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                #Using the estimated landmark position
                entity_pos.append(world.landmarks[i+world.num_landmarks].state.p_pos - agent.state.p_pos)
                #Using the estimated landmark position but without delating the agent position. so it has a global position.
                # entity_pos.append(world.landmarks[i+world.num_landmarks].state.p_pos)
                entity_range.append(slant_range)
                
                # Move the landmark if movable
                if entity.movable:
                    if self.movement == 'linear':
                        #linear movement
                        u_force = 0.05
                        entity.action.u = np.array([np.cos(self.ra)*u_force,np.sin(self.ra)*u_force])
                    
                    elif self.movement == 'random':
                        # random movement
                        entity.action.u = np.random.randn(2)/2.
                    
                    elif self.movement == 'levy':
                        #random walk Levy movement
                        beta = 1.9 #must be between 1 and 2
                        entity.action.u = random_levy(beta)
                        if entity.state.p_pos[0] > 0.8:
                            entity.action.u[0] = -abs(entity.action.u[0])
                        if entity.state.p_pos[0] < -0.8:
                            entity.action.u[0] = abs(entity.action.u[0])
                        if entity.state.p_pos[1] > 0.8:
                            entity.action.u[1] = -abs(entity.action.u[1])
                        if entity.state.p_pos[1] < -0.8:
                            entity.action.u[1] = abs(entity.action.u[1])
                    
                        
                        
                        
                
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
        
        #observations used until test 40
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + [entity_range])
        #observations used on test 41 and test 114
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + [entity_range])
    
    def done(self, agent, world):
        # episodes are done based on the agents minimum distance from a landmark.
        global done_state
        # dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks[:-world.num_landmarks]]
        # if min(dists) > 1.5 or min(dists) < 0.1:
        if done_state:
            done = True
        else:
            done = False
        return done
