from envs import gridworld
import numpy as np
from value_iteration import *
from maximumentropy_irl import *
import chainer
import chainer.links as Link
import chainer.functions as Func
from chainer import optimizers

def plot_res(V,gridworldvar):
    plot = np.zeros(gridworldvar.shape)
    for key, val in enumerate(V):
        plot[key // gridworldvar.shape[1], key % gridworldvar.shape[1]] = val #plot heat map , increased value means brighter colour
    return plot

class Class_Reward(chainer.Chain):
    def __init__(self, inputi, hiddeni):
        super(Class_Reward, self).__init__(
            ln1=Link.Linear(inputi, hiddeni),
            ln2=Link.Linear(hiddeni, hiddeni),# hidden layer
            ln3=Link.Linear(hiddeni, 1)
        )

    def __call__(self, x):
        h1 = Func.relu(self.ln1(x))#activation function 
        h2 = Func.tanh(self.ln2(h1))
        return self.ln3(h2)

def f_exp(features_arrayv,feature_expectationv,trajectoriesv):

    for trajectory in trajectoriesv:
        for step in trajectory:#for every state,action pair (ζ)
            #choose the step[0] row of the I(identity) matrix (basically count the frequency of all current states)
            feature_expectationv = feature_expectationv + features_arrayv[step[0], :]#φ(ζ) : sum of state features along the path
            #average path

    feature_expectationv = feature_expectationv/len(trajectoriesv)# Φ

    return feature_expectationv

def maximumentropy_deepirl(features_array, trajectories, transitions ,states,actions,
                g=0.8, epochs=50):
    """
    The nth row of the matrix represents the nth state. 
    Feature matrix is an identity matrix in this case(the place of 1 [0-15] means the No.state) , but I treat it
    as a shape (N, D) where N is the number of states and D is the
    dimensionality of that specific state. 
    """

    states_num, states_dim = features_array.shape
    feature_expectation = np.zeros(states_dim)#(1,states) ->size
    
    NN_reward = Class_Reward(states_dim,82)
    nnoptimizer = optimizers.Adam()
    nnoptimizer.setup(NN_reward)
    nnoptimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    feature_expectation = f_exp(features_array,feature_expectation,trajectories)
    
    f = chainer.Variable(features_array.astype(np.float32)) # this time there is no thita. I compute reward variable with a chainer variable passed to NN
    for _ in range(epochs):
        NN_reward.zerograds()#reset gradients
        reward_var = NN_reward(f)#compute reward
        vi = value_iteration(transitions, reward_var.array,states,actions, g)#same steps as max_ent 
        pi = optimal_policy(transitions, vi) #same steps as max_ent 
        svisit_fq = state_visitation_frequency(transitions, pi , trajectories,states_num) #same steps as max_ent 
        L_th_grad = feature_expectation - svisit_fq #compute gradient
        L_th_grad = L_th_grad.reshape((states_num, 1)).astype(np.float32)
        reward_var.grad = -L_th_grad #balance rewards
        reward_var.backward()
        nnoptimizer.update()

    return NN_reward(f).array#return reward


if __name__ == '__main__':
 
    gridworldvar = gridworld.GridworldEnv()
    
    alist = [gridworldvar.P[s] for s in range(gridworldvar.nS)]

    #parse gym instance
    blist = []
    for x in alist:
        clist = []
        for y in x:
            z = flatten(x[y])
            clist.append(flatten(np.eye(1,gridworldvar.nS,z[1]))) # No. of rows , nS :number of states , index(second value: 1)
        blist.append(clist)

    transitions = np.array(blist)

    print('\n\n\n\n')

    #get reward from the dict
    alist = [gridworldvar.P[s][0][0][2] for s in range(gridworldvar.nS)]
    reward = np.array(alist) 
    print(reward)

    states,actions,key = transitions.shape

    #transitions =  trans_mat(gridworldvar)
    #reward = reward.astype(np.float64)

    V = value_iteration(transitions, reward, states, actions)
    op = optimal_policy(transitions, V)

    #print(op)
    print(transitions)
    #print(trajectories)
    
    trajectories = generate_trajectories(op, gridworldvar)

    result = maximumentropy_deepirl(np.eye(gridworldvar.nS), trajectories, transitions ,states,actions)
    print(result)

    plot = plot_res(result,gridworldvar)
    plots.matshow(plot)
    plots.savefig("maximumentropy_deepirl.png")