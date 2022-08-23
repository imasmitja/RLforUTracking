import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        self.p_pos_origin = None
        # physical velocity
        self.p_vel = None
        self.p_vel_old = None
        self.a_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        self.max_a_speed = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        # action
        self.action = Action()
        # physical motor noise amount
        self.u_noise = None
        # velocity,direction,depth
        self.landmark_vel = 0.
        self.ra = 0.
        self.landmark_depth = 0.

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.landmarks_estimated = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep (in seconds)
        self.dt = 0.1
        self.dt = 30
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # if world is collaborative
        self.num_agents = 3
        self.num_landmarks = 3
        self.collaborative = True
        self.angle = []
        # sea currents
        self.vel_ocean_current = 0.
        self.angle_ocean_current = 0.

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,entity in enumerate(self.entities):
            if 'agent' in entity.name:
                if entity.movable:
                    noise = np.random.randn(*entity.action.u.shape) * entity.u_noise if entity.u_noise else 0.0
                    p_force[i] = entity.action.u + noise 
            if 'landmark' in entity.name:
                if entity.movable:
                    noise = np.random.randn(*entity.action.u.shape) * entity.u_noise if entity.u_noise else 0.0
                    p_force[i] = entity.action.u + noise 
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            
            if 'landmark' in entity.name:
                #if entity is a landmark (x-y force applyied independently)
                if (p_force[i] is not None):
                    entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                if entity.max_speed is not None:
                    speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                    if speed > entity.max_speed:
                        entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(entity.state.p_vel[1])) * entity.max_speed
                entity.state.p_pos += entity.state.p_vel * self.dt
            
            if 'agent' in entity.name:
                #if entity is an agnet (constant velocity, increment of angle)
                '''This is the new approach designed by Ivan'''
                #First position of p_vel is the angular velocity which is used to increase the angle of the agent
                if (p_force[i] is not None):
                    # entity.state.p_vel[0] = p_force[i] * 0.1 #multiply by 0.1 to set radius limit at 100m minimum (taken into consideration that the p_force are bounded between -1 and 1)
                    self.angle[i] += p_force[i].item(0)*0.3 #multiply by 0.1 to set radius limit at 100m minimum
                    if self.angle[i] > np.pi*2.:
                        self.angle[i] -= np.pi*2.
                    if self.angle[i] < -np.pi*2:
                        self.angle[i] += np.pi*2
                #The second position of p_vel is the liniar velocity of the agent
                # vel = entity.state.p_vel[1]+0.
                vel = 0.001 #seting this velocity (0.1) and considering that the dt is equal to 0.1, means that we have a new position each 10m.
                if vel < 0:
                    vel = 0
                #Finally, we increase the position (aka the agent) using the new angle and velocity.
                entity.state.p_pos += np.array([vel*np.cos(self.angle[i]),vel*np.sin(self.angle[i])]) * self.dt
                entity.state.p_vel = np.array([vel*np.cos(self.angle[i]),vel*np.sin(self.angle[i])])               
                ocean_current = True
                if ocean_current == True:
                    entity.state.p_pos += np.array([self.vel_ocean_current*np.cos(self.angle_ocean_current),self.vel_ocean_current*np.sin(self.angle_ocean_current)]) * self.dt
                    # we don't need to modify the vel of the agent, in that way, we conserve its real direction. And not the direction of the current,
                    # with can be extrapolated from the position of the agent, its next position, and its direction.
                    # entity.state.p_vel += np.array([self.vel_ocean_current*np.cos(self.angle_ocean_current),self.vel_ocean_current*np.sin(self.angle_ocean_current)]) 
                
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]