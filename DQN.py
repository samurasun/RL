import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
from collections import deque
import numpy as np

# experiences replay buffer size
REPLAY_SIZE = 10000
# size of minibatch
small_BATCH_SIZE = 16
big_BATCH_SIZE = 128
BATCH_SIZE_door = 1000

# these are the hyper Parameters for DQN
# discount factor for target Q to caculate the TD aim value
GAMMA = 0.9
# the start value of epsilon
INITIAL_EPSILON = 0.5
# the final value of epsilon
FINAL_EPSILON = 0.01

class DQN():
    def __init__(self, observation_space, action_space):
        # the state is the input vector of network, in this env, it has four dimensions
        self.state_dim = observation_space.shape[0]
        # the action is the output vector and it has two dimensions
        self.action_dim = action_space.n
        # init experience replay, the deque is a list that first-in & first-out
        self.replay_buffer = deque()
        # you can create the network by the two parameters
        self.create_Q_network()
        # after create the network, we can define the training methods
        self.create_updating_method()
        # set the value in choose_action
        self.epsilon = INITIAL_EPSILON
        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    # the function to create the network
    # we set the network with four layers
    # (self.state_dim[4]-->50-->20-->self.action_dim[1])
    # there are two networks, the one is action_value and the other is target_action_value
    # these two networks has same architecture
    def create_Q_network(self):
        # first, set the input of networks
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # second, create the current_net
        with tf.variable_scope('current_net'):
            # first, set the network's weights
            W1 = self.weight_variable([self.state_dim, 50])
            b1 = self.bias_variable([50])
            W2 = self.weight_variable([50, 20])
            b2 = self.bias_variable([20])
            W3 = self.weight_variable([20, self.action_dim])
            b3 = self.bias_variable([self.action_dim])
            # second, set the layers
            # hidden layer one
            h_layer_one = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
            # hidden layer two
            h_layer_two = tf.nn.relu(tf.matmul(h_layer_one, W2) + b2)
            # the output of current_net
            self.Q_value = tf.matmul(h_layer_two, W3) + b3
        # third, create the current_net
        with tf.variable_scope('target_net'):
            # first, set the network's weights
            t_W1 = self.weight_variable([self.state_dim, 50])
            t_b1 = self.bias_variable([50])
            t_W2 = self.weight_variable([50, 20])
            t_b2 = self.bias_variable([20])
            t_W3 = self.weight_variable([20, self.action_dim])
            t_b3 = self.bias_variable([self.action_dim])
            # second, set the layers
            # hidden layer one
            t_h_layer_one = tf.nn.relu(tf.matmul(self.state_input, t_W1) + t_b1)
            # hidden layer two
            t_h_layer_two = tf.nn.relu(tf.matmul(t_h_layer_one, t_W2) + t_b2)
            # the output of current_net
            self.target_Q_value = tf.matmul(t_h_layer_two, t_W3) + t_b3
        # at last, solve the parameters replace problem
        # the parameters of current_net
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')
        # the parameters of target_net
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        # define the operation that replace the target_net's parameters by current_net's parameters
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    # the function that give the weight initial value
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    # the function that give the bias initial value
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    # this the function that define the method to update the current_net's parameters
    def create_updating_method(self):
        # this the input action, use one hot presentation
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        # this the TD aim value
        self.y_input = tf.placeholder("float", [None])
        # this the action's Q_value
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        # this is the lost
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        # use the loss to optimize the network
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    # this is the function that use the network output the action
    def Choose_Action(self, state):
        # the output is a tensor, so the [0] is to get the output as a list
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        # use epsilon greedy to get the action
        if random.random() <= self.epsilon:
            # if lower than epsilon, give a random value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            # if bigger than epsilon, give the argmax value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    # this the function that store the data in replay memory
    def Store_Data(self, state, action, reward, next_state, done):
        # generate a list with all 0,and set the action is 1
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        # store all the elements
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        # if the length of replay_buffer is bigger than REPLAY_SIZE
        # delete the left value, make the len is stable
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

    # train the network, update the parameters of Q_value
    def Train_Network(self, BATCH_SIZE):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate TD aim value
        y_batch = []
        # give the next_state_batch flow to target_Q_value and caculate the next state's Q_value
        Q_value_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})
        # caculate the TD aim value by the formulate
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                # the Q value caculate use the max directly
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        # step 3: update the network
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def Update_Target_Network(self):
        # update target Q netowrk
        self.session.run(self.target_replace_op)

    # use for test
    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])


ENV_NAME = 'CartPole-v0'
EPISODES = 1000
STEPS = 300
# steps that copy the current_net's parameters to target_net
UPDATE_STEP = 50
# times that evaluate the network
TEST = 5

def main():
    # first, create the envrioment
    env = gym.make(ENV_NAME)
    # second, create the agent
    # the observation_space is Box(4,) object and the action_space Discrete(2)
    # it represent that the observation_space has four var and each one is float
    # and the action_space has two Discrete value， for example Discrete(3) has two values which are 0, 1, 2
    agent = DQN(env.observation_space, env.action_space)
    for episode in range(EPISODES):
        # get the initial state
        state = env.reset()
        for step in range(STEPS):
            # show the game window
            #env.render()
            # get the action by state
            action = agent.Choose_Action(state)
            # step the env forward and get the new state
            next_state, reward, done, info = env.step(action)
            # store the data in order to update the network in future
            agent.Store_Data(state, action, reward, next_state, done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                agent.Train_Network(big_BATCH_SIZE)
            # if len(agent.replay_buffer) > small_BATCH_SIZE:
            #     if len(agent.replay_buffer) < BATCH_SIZE_door:
            #         agent.Train_Network(small_BATCH_SIZE)
            #     else:
            #         agent.Train_Network(big_BATCH_SIZE)
            # update the target_network
            if step % UPDATE_STEP == 0:
                agent.Update_Target_Network()
            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEPS):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
  main()