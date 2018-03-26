from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.her.normalizer import Normalizer
from baselines.her.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.her.experiment import config


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, replay_k, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        # Create the actor critic networks. network_class is defined in actor_critic.py
        # This class is assigned to network_class when DDPG objest is created
        self.create_actor_critic = import_function(self.network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        # Next state (o_2) and goal at next state (g_2)
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Adding variable for correcting bias - Ameet
        self.stage_shapes_new = OrderedDict()
        self.stage_shapes_new['bias'] = (None,)
        ##############################################

        # Create network
        # Staging area is a datatype in tf to input data into GPUs
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            
            # Adding bias term from section 3.4 - Ameet
            self.staging_tf_new = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes_new.keys()],
                shapes=list(self.stage_shapes_new.values()))
            self.buffer_ph_tf_new = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes_new.values()]
            self.stage_op_new = self.staging_tf_new.put(self.buffer_ph_tf_new)
            ############################################

            self._create_network(reuse=reuse)

        # Configure the replay buffer
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size

        # conf represents the parameters required for initializing the priority_queue
        # Remember: The bias gets annealed only conf.total_steps number of times
        conf = {'size': self.buffer_size,
                'learn_start': self.batch_size,
                'batch_size': self.batch_size,
                # Using some heuristic to set the partition_num as it matters only when the buffer is not full (unlikely)
                'partition_size': (self.replay_k+1)*100}

        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions, conf, self.replay_k)

        # global_steps represents the number of batches used for updates
        self.global_step = 0
        self.debug = {}

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    # Preprocessing by clipping the goal and state variables
    # Not sure about the relative_goal part
    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    # target is the target policy network and main is the one which is updated
    # target is updated by moving the parameters towards that of the main
    # pi_tf is the output of the policy network, Q_pi_tf is the output of the Q network used for training pi_tf
    # i.e., Q_pi_tf uses the pi_tf's action to evaluate the value 
    # While just Q_tf uses the action which was actually taken
    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        # Updating stats

        ## Change this--------------
        update_stats = False
        ###--------------------------
        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    # Adam update for Q and pi networks
    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    # Sample a batch for mini batch gradient descent, already defined in replay_buffer.py
    def sample_batch(self):
        # Increment the global step
        self.global_step += 1

        transitions, w, rank_e_id = self.buffer.sample(self.batch_size, self.global_step, self.uniform_priority)
        priorities = self.get_priorities(transitions)

        # ##### Debug function
        # self.debug_td_error(transitions, priorities)
        # #####
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        # # Remove
        # print("Stage Shape keys in sample_batch are: "+str(self.stage_shapes.keys()))

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]

        # Updates the priorities of the sampled transitions in the priority queue
        self.buffer.update_priority(rank_e_id, priorities)

        return transitions_batch, [w]


    # This function is purely for debugging purposes
    def debug_td_error(self, transitions, priorities):
        f = open('td_error_debug.txt', 'a')
        self.debug['actual_goals'] = 0
        self.debug['alternate_goals'] = 0
        trans = transitions['is_actual_goal']
        for t in range(trans.shape[0]):
            if trans[t]:
                self.debug['actual_goals'] += 1
                # f.write('Actual goal transition: '+str(priorities[t])+'\n')
            else:
                self.debug['alternate_goals'] += 1
                # f.write('Alternate goal transition: '+str(priorities[t])+'\n')
        f.write('Ratio is: '+str(float(self.debug['alternate_goals'])/self.debug['actual_goals'])+'\n')
        del transitions['is_actual_goal']

    def get_priorities(self, transitions):
        pi_target = self.target.pi_tf
        Q_pi_target = self.target.Q_pi_tf
        Q_main = self.main.Q_tf


        o = transitions['o']
        o_2 = transitions['o_2']
        u = transitions['u']
        g = transitions['g']
        r = transitions['r']
        # Check this with Srikanth
        ag = transitions['ag']

        priorities = np.zeros(o.shape[0])

        # file_obj = open("priorities_print","a")
        for i in range(o.shape[0]):
            o_2_i = np.clip(o_2[i], -self.clip_obs, self.clip_obs)
            o_i, g_i = self._preprocess_og(o[i], ag[i], g[i])
            u_i = u[i]

            # Not sure about the o_2_i.size // self.dimo. I guess we need not pass one at a time
            feed_target = {
                self.target.o_tf: o_2_i.reshape(-1, self.dimo),
                self.target.g_tf: g_i.reshape(-1, self.dimg),
                self.target.u_tf: np.zeros((o_2_i.size // self.dimo, self.dimu), dtype=np.float32)
            }

            # u_tf for main network is just the action taken at that state
            feed_main = {
                self.main.o_tf: o_i.reshape(-1, self.dimo),
                self.main.g_tf: g_i.reshape(-1, self.dimg),
                self.main.u_tf: u_i.reshape(-1, self.dimu)
            }

            TD = r[i] + self.gamma*self.sess.run(Q_pi_target, feed_dict=feed_target) - self.sess.run(Q_main, feed_dict=feed_main)

            priorities[i] = abs(TD)

            text = str(TD)
            # file_obj.write(text)
        # file_obj.close()

        return priorities


    def stage_batch(self, batch=None):
        if batch is None:
            batch, bias = self.sample_batch()
            # print("Batch type is: "+str(type(batch)))
            # print("Batch Shape is: "+str(len(batch)))
            # print(str(type(batch[0])))
        assert len(self.buffer_ph_tf) == len(batch), "Expected: "+str(len(self.buffer_ph_tf))+" Got: "+str(len(batch))
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

        ##### Adding for bias - Ameet
        assert len(self.buffer_ph_tf_new) == len(bias), "Expected: "+str(len(self.buffer_ph_tf_new))+" Got: "+str(len(bias))
        self.sess.run(self.stage_op_new, feed_dict=dict(zip(self.buffer_ph_tf_new, bias)))
        #####
        
        # print("Completed stage batch")

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        # print("In ddpg priority:: The shapes of Q_grad and pi_grad are: "+str(Q_grad.shape)+"::"+str(pi_grad.shape))
        # print("Their types are::"+str(type(Q_grad)))
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        ########### Getting the bias terms - Ameet
        bias = self.staging_tf_new.get()
        bias_tf = OrderedDict([(key, bias[i])
                                for i, key in enumerate(self.stage_shapes_new.keys())])
        bias_tf['bias'] = tf.reshape(bias_tf['bias'], [-1, 1])
        #######################################

        # Create main and target networks, each will have a pi_tf, Q_tf and Q_pi_tf
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        ############## Added for bias - Ameet
        error = (tf.stop_gradient(target_tf) - self.main.Q_tf) * bias_tf['bias']
        self.Q_loss_tf = tf.reduce_mean(tf.square(error))
        # self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf * bias_tf['bias'])
        # Note that the following statement does not include bias because of the remark in the IEEE paper
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        ##############
        # Regularization - L2 - Check - Penalty for taking the best action
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        ################### Shape Info
        ####Shape of Q_grads_tf is: 8
        ####Shape of Q_grads_tf[0] is: (17, 256)
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        # 'main/Q' is a way of communicating the scope of the variables
        # _vars has a way to understand this
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        # Update the networks
        # target net is updated by using polyak averaging
        # target net is initialized by just copying the main net
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
