import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import math

def transpose_list(mylist):
    return list(map(list, zip(*mylist)))

def transpose_to_tensor(input_list):
    make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=0.5, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y

# modified by Ivan, to see an agent doing circles around the predicted landmark position. To compere with a MADDPG trained agent.
tracked = False
direction = -1.
old_distance = 1.
def circle_path(obs_all,radius,k):
    global tracked
    global direction
    global old_distance
    if k == 1:
        tracked = False
        direction = -1
    # Set the movement of the mywg
    # Get parameters
    actions = np.array([[[]]])
    for obs_env in obs_all:
        for obs in obs_env:
            agent_pos = np.matrix([obs[2],obs[3]]).T #Agent position [x,y]
            agent_ang = np.arctan2(obs.item(1),obs.item(0))
            if agent_ang < 0.:
                agent_ang += 2* np.pi
            landmark_pos = np.matrix([obs[4],obs[5]]).T #Predicted landmark position [x,y]
            angle_agent_landmark = np.arctan2((landmark_pos).item(1),(landmark_pos).item(0)) #Angle between WG and Target
            if angle_agent_landmark < 0.:
                angle_agent_landmark += 2* np.pi
            distance = np.sqrt((landmark_pos.item(0))**2+(landmark_pos.item(1))**2)

            alpha_radius = 64.181*radius**(-0.872) #extracted using a power regresion line with empirical values
            radius_circ = radius/1000.  
            # Option 2, whcih is designed to try to optimize the first option.
            if (distance > radius_circ*2) or k < 4:
                const = 0.5
                if agent_ang - angle_agent_landmark > np.pi and agent_ang > angle_agent_landmark:
                    direction = 1.
                    angle = direction
                elif agent_ang - angle_agent_landmark < np.pi and agent_ang > angle_agent_landmark:
                    direction = -1. 
                    if agent_ang - angle_agent_landmark < np.pi/2./2.:
                            angle = 0.
                    else:
                        angle = direction
                elif angle_agent_landmark - agent_ang > np.pi and agent_ang < angle_agent_landmark:
                    direction = -1.
                    angle = direction
                elif angle_agent_landmark - agent_ang < np.pi and agent_ang < angle_agent_landmark:
                    direction = 1.
                    if angle_agent_landmark - agent_ang < np.pi/2./2.:
                            angle = 0.
                    else:
                        angle = direction
                else:
                    angle = 0.
                
            else:
                aux = abs(agent_ang - angle_agent_landmark)
                vector_1 = np.array([obs.item(1),obs.item(0)])
                vector_2 = np.array([(landmark_pos).item(1),(landmark_pos).item(0)])
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                aux = np.arccos(dot_product)
                angle = 0.4*(aux+0.7)*direction
                angle -= (1-alpha_radius)
                
            if k < 2:
                angle = -1.

                
                
    return np.array([[[angle]]])


def random_levy(beta):
    # beta = 1. #must be between 1 and 2
    direction = np.random.uniform(0,2*np.pi)
    sigma = ((math.gamma(1+beta)*np.sin(np.pi*beta/2))/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0.,sigma)
    v = np.random.normal(0.,1.)
    step_length = np.clip(u / abs(v)**(1/beta),-5,5)
    return np.array([np.cos(direction)*step_length, np.sin(direction)*step_length])
    
    


"""def main():
    torch.Tensor()
    print(onehot_from_logits())

if __name__=='__main__':
    main()"""