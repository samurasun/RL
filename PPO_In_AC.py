import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
from collections import deque
import numpy as np

# hyper parameters
# learning rate of critic
Critic_LR = 0.0002
# learning rate of actor
Actor_LR = 0.0001
# the method of updating
# contains the PPO and PPO2
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method 'clip' for optimization
# the steps of training
BATCH_SIZE = 32
# the game's name
ENV_NAME = 'Pendulum-v0'
# the total episodes of training
EPISODES = 1000
# the total steps in one episode
STEPS = 300
# used to caculate the discount reward
GAMMA = 0.9
# the update times of actor and critic
ACTOR_UPDATE_TIMES = 10
CRITIC_UPDATE_TIMES = 10

# the output of PPO is continuous
class PPO():
    def __init__(self, observation_space, action_space):
        # the state is the input vector of network, in the game of 'Pendulum-v0', it has three dimensions
        self.state_dim = observation_space.shape[0]
        # the action is the output vector and  in the game of 'Pendulum-v0', it has one dimensions
        self.action_dim = action_space.shape[0]
        # it is the input, which come from the env
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
        # create the network to represent the state value
        self.Create_Critic()
        # create two networks to output the action, and update the networks
        self.Create_Actor_with_two_network()
        # Init session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # the critic network give the value of state
    def Create_Critic(self):
        # first, create the parameters of networks
        W1 = self.weight_variable([self.state_dim, 100])
        b1 = self.bias_variable([100])
        W2 = self.weight_variable([100, 50])
        b2 = self.bias_variable([50])
        W3 = self.weight_variable([50, self.action_dim])
        b3 = self.bias_variable([self.action_dim])
        # second, create the network with two hidden layers
        # hidden layer one
        h_layer_one = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        # hidden layer two
        h_layer_two = tf.nn.relu(tf.matmul(h_layer_one, W2) + b2)
        # the output of current_net
        self.v = tf.matmul(h_layer_two, W3) + b3
        # third, give the update method of critic network
        # the input of discounted reward
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        # the advantage value, use to update the critic network
        self.advantage = self.tfdc_r - self.v
        # the loss of the network
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        # the training method of critic
        self.ctrain_op = tf.train.AdamOptimizer(Critic_LR).minimize(self.closs)
        return

    # the actor network that give the action
    def Create_Actor_with_two_network(self):
        # create the actor that give the action distribution
        pi, pi_params = self.build_actor_net('pi', trainable=True)
        # create the actor that caculate the loss value
        oldpi, oldpi_params = self.build_actor_net('oldpi', trainable=False)
        # sample the action from the distribution
        with tf.variable_scope('sample_action'):
            #self.sample_from_pi = tf.squeeze(pi.sample(1), axis=0)
            self.sample_from_oldpi = tf.squeeze(oldpi.sample(1), axis=0)
        # update the oldpi by coping the parameters from pi
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_from_pi = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        # the actions in memory
        self.tfa = tf.placeholder(tf.float32, [None, self.action_dim], 'action')
        # the advantage value
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # the ration between the pi and oldpi, this is importance sampling part
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
                # the surrogate
                surr = ratio * self.tfadv
            # this is the method of PPO
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:  # this is the method of PPO2
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.tfadv))
        # define the method of training actor
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(Actor_LR).minimize(self.aloss)
        return

    # the function that create the actor network
    # it has two hidden layers
    # the method of creating actor is different  from the critic
    # the output of network is a distribution
    def build_actor_net(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.state_input, 100, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 50, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l2, self.action_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l2, self.action_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # output the action with state, the output is from oldpi
    def Choose_Action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_from_oldpi, {self.state_input: s})[0]
        return np.clip(a, -2, 2)

    # reset the memory in every episode
    def resetMemory(self):
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []

    # store the data of every steps
    def Store_Data(self, state, action, reward, next_state, done):
        self.buffer_s.append(state)
        self.buffer_a.append(action)
        self.buffer_r.append(reward)

    # get the state value from critic
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.state_input: s})[0, 0]

    # the function that update the actor and critic
    def update(self, s, a, r):
        adv = self.sess.run(self.advantage, {self.state_input: s, self.tfdc_r: r})
        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(ACTOR_UPDATE_TIMES):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.state_input: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)  # sometimes explode, this clipping is my solution
        else:  # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.state_input: s, self.tfa: a, self.tfadv: adv}) for _ in range(ACTOR_UPDATE_TIMES)]
        # update critic
        [self.sess.run(self.ctrain_op, {self.state_input: s, self.tfdc_r: r}) for _ in range(CRITIC_UPDATE_TIMES)]

    # the train function that update the network
    def Train(self, next_state):
        # caculate the discount reward
        v_s_ = self.get_v(next_state)
        discounted_r = []
        for r in self.buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        bs, ba, br = np.vstack(self.buffer_s), np.vstack(self.buffer_a), np.array(discounted_r)[:, np.newaxis]
        # this the main function of update
        self.update(bs, ba, br)

    # ths dunction the copy the pi's parameters to oldpi
    def UpdateActorParameters(self):
        self.sess.run(self.update_oldpi_from_pi)

    # the function that give the weight initial value
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    # the function that give the bias initial value
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

all_ep_r = []
def main():
    # first, create the envrioment of 'Pendulum-v0'
    # the game Pendulum-v0's observation_space is Box(3,), it means the observation has three var and each one is float
    # the action_space is Box(1,), it means the action has one var and is float
    # the output of the game is continuous, not discrete
    env = gym.make(ENV_NAME).unwrapped
    # second, create the PPO agent. it is based the AC arch, so it has two type of network
    # the critic which give the value of the state
    # the actor which give the action
    agent = PPO(env.observation_space, env.action_space)
    for episode in range(EPISODES):
        # every episode reset the memory
        agent.resetMemory()
        # get the initial state
        state = env.reset()
        # this is the total reward
        ep_r = 0
        for step in range(STEPS):
            # show the game window
            env.render()
            # output the action
            action = agent.Choose_Action(state)
            # process the action and get the info
            next_state, reward, done, info = env.step(action)
            # store the date to update the model
            agent.Store_Data(state, action, reward, next_state, done)
            # train the agent every BATCH_SIZE
            if (step + 1) % BATCH_SIZE == 0 or step == STEPS - 1:
                agent.Train(next_state)
            # set the new state
            state = next_state
            # record the reward
            ep_r += reward
            if step == STEPS - 1:
                agent.UpdateActorParameters()
        # caculate the total reward in every episode
        if episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        print(
            'Ep: %i' % episode,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )

    return

if __name__ == '__main__':
  main()