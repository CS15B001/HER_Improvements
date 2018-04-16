import threading

import numpy as np

from baselines.her.rank_based_new import Experience

import copy


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, conf, replay_k, n_reps, nonuniform_sampling):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        # Integer division
        self.size = size_in_transitions // T
        # T represents the number of timesteps in each episode
        self.T = T
        self.sample_transitions = sample_transitions
        self.replay_k = replay_k
        self.n_reps = n_reps
        self.nonuniform_sampling = nonuniform_sampling

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)} - OpenAI comment
        # The keys are 'o', 'g', 'ag' etc. dim_key represents the dimensionality of the 
        # vector corresponding to 'o', 'g' etc. * (starred expression) is used to unpack
        # the elements of the list

        # self.buffers now contains static memory to store all the transitions
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # self.queue_ratio contains the number of initial transitions in a batch which belong to the actual goals
        self.queue_ratio = int(1./(self.replay_k+1)*conf['batch_size'])
        conf = [copy.deepcopy(conf), copy.deepcopy(conf)]
        conf[0]['partition_size'], conf[0]['learn_start'], conf[0]['batch_size'] = 1*100-1, self.queue_ratio, self.queue_ratio 
        conf[1]['partition_size'], conf[1]['learn_start'], conf[1]['batch_size'] = self.replay_k*100-1, conf[1]['batch_size'] - self.queue_ratio, conf[1]['batch_size'] - self.queue_ratio
        # Unlike before, self.priority_queue is now a list
        # Alternate Buffer - Priority Queue
        self.priority_queue = [Experience(conf[i]) for i in range(2)]

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size, global_step, uniform_priority = False):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        # Doubt: 1: represents Time step 1 onwards?
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        # ToDo: Replace the last argument with global_step
        batch_size = [self.queue_ratio, batch_size-self.queue_ratio]
        transitions, w, rank_e_id = [[None, None] for _ in range(3)]
        transitions[0], w[0], rank_e_id[0] = self.sample_transitions(buffers, self.priority_queue[0], batch_size[0], global_step, uniform_priority)
        transitions[1], w[1], rank_e_id[1] = self.sample_transitions(buffers, self.priority_queue[1], batch_size[1], global_step, uniform_priority)    

        # Combine the two transitions
        transitions_ = {}
        w_, rank_e_id_ = [np.hstack((w[0], w[1])), rank_e_id[0] + rank_e_id[1]]
        for key in transitions[0].keys():
            if len(transitions[0][key].shape) > 1:
                transitions_[key] = np.vstack((transitions[0][key], transitions[1][key]))
            else:
                transitions_[key] = np.hstack((transitions[0][key], transitions[1][key]))

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions_, "key %s missing from transitions. In replay_buffer.py" % key

        return transitions_, w_, rank_e_id_

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]

        # Assert that sizes of all the keys are the same, i,e., a, g and o should
        # all have equal number of entries
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:

            # The code before transitions[] line can eventually be removed for PER
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            # The keys are 'o', 'a', 'ag' etc. self.buffers[key] is a numpy array and
            # idxs are a list of numbers. Thus only a few of the experiences will be replaced
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            # Total number of transitions includes each time step
            self.n_transitions_stored += batch_size * self.T*(self.replay_k+1)*self.n_reps

            # Store the transitions in the priority_queue
            # Slice the episodes to get the transitions

            # ###### Debug
            # debug_transitions = [[], []]
            # ###### Remove this

            # Remove this
            count = 0
            
            # These are the transitions with the actual goal
            for _ in range(self.n_reps):
                for t in range(batch_size):
                    for time_step in range(self.T):
                        transition = {key: episode_batch[key][t, time_step] for key in episode_batch.keys()}

                        # This is done at sample time in the original code, we are doing it here
                        transition['ag_2'] = episode_batch['ag'][t, time_step+1]
                        transition['o_2'] = episode_batch['o'][t, time_step+1]

                        ######### This is being done for debugging purposes
                        transition['is_actual_goal'] = True
                        ######### Remove this

                        # Store in the priority_queue - The one which stores the actual goals
                        self.priority_queue[0].store(transition)
                        count += 1

                        # ###### Debug
                        # debug_transitions[0].append(transition)
                        # ###### Remove this

            print("Total number of transitions added: "+str(count))

            # These are transitions with alternate goals
            for t in range(batch_size):
                for time_step in range(self.T):
                    if self.nonuniform_sampling:
                        future_t = self._sample_alt_goals(time_step)
                    else:
                        future_offset = np.random.uniform(size=self.replay_k*self.n_reps) * (self.T - time_step)
                        future_offset = [elem.astype(int) for elem in future_offset]
                        future_t = [(time_step + 1 + elem) for elem in future_offset]
                    
                    for future_time_step in future_t:
                        future_ag = episode_batch['ag'][t, future_time_step]
                        transition = {key: episode_batch[key][t, time_step] for key in episode_batch.keys()}
                        transition['g'] = future_ag
                        
                        # This is done at sample time in the original code, we are doing it here
                        transition['ag_2'] = episode_batch['ag'][t, time_step+1]
                        transition['o_2'] = episode_batch['o'][t, time_step+1]

                        ######### This is being done for debugging purposes
                        transition['is_actual_goal'] = False
                        ######### Remove this
                        
                        # Store in the priority_queue - The one which stores the actual goals
                        self.priority_queue[1].store(transition)
                        count += 1

                        # ###### Debug
                        # debug_transitions[1].append(transition)
                        # ###### Remove this

            print("Total number of transitions added: "+str(count))

            # ###### Debug
            # return debug_transitions
            # ###### Remove this

    def _sample_alt_goals(self, time_step):
        # Sample 2*k*(T - t)/(T+1) number of alt goals
        # Sample uniformly the above number of alt goals

        size_t = round(2*self.replay_k*self.n_reps*(self.T - time_step)/(self.T+1)) + 1

        # print(size_t)

        future_offset = np.random.uniform(size=size_t)*(self.T - time_step)
        future_offset = [elem.astype(int) for elem in future_offset]
        future_t = [(time_step + 1 + elem) for elem in future_offset]

        return future_t


    # Size in terms of number of episodes stored in the buffer
    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    # T represents the number of time steps in each episode
    # This represents the total number of time steps of
    # interation with the environment. (Previous function)*T
    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    # The number of transitions stored in the buffer
    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    # Empty the buffer
    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    # Get the indices to store the episodes
    # To remind, buffer is characterized by ['key_of_interest', episode_index, time_inside_episode, vector_of_interest]
    # If the replay buffer is full, randomly remove episodes till there is space
    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        # If there isn't enough space, evict random episodes
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        # If it is only one episode, return the index, else a list
        if inc == 1:
            idx = idx[0]
        return idx

    # Update the priorities of the sampled transitions
    def update_priority(self, rank_e_id, priorities):
        self.priority_queue[0].update_priority(rank_e_id[:self.queue_ratio], priorities[:self.queue_ratio])
        self.priority_queue[1].update_priority(rank_e_id[self.queue_ratio:], priorities[self.queue_ratio:])
        # f = open('checking_td_error.txt', 'a')
        # f.write('Transitions\n'+str(rank_e_id)+'Priorities\n'+str(priorities)+'\n')
        # f.write('Max priority is: '+str(self.priority_queue.priority_queue.get_max_priority())+'\n')
        # f.write('Buffer size is: '+str(self.priority_queue.record_size)+'\n')
        # f.close()
