import numpy as np
from rank_based import Experience

"""
Will serve as an alternative to her.py
Stores transitions in the priority queue and samples accordingly
"""

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay
       with prioritized sampling

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    # def _sample_her_transitions(episode_batch, batch_size_in_transitions):
    def _sample_her_transitions(episode_batch, priority_queue, batch_size_in_transitions, global_step):
        """priority_queue is an instance of type 'Experience' defined in rank_based.py
        """

        # Sample from the given priority_queue
        # global_step represents the step of the learning process
        # needed for annealing the bias
        sample_transitions, w, rank_e_id = priority_queue.sample(global_step)

        # Create a dictionary
        # key_to_index_map maps the keys of episode batch to the corresponding
        # index in sample_transitions. For example, if sample transitions returns 
        # (s,a,r,s,t), key_to_index_map['u'] = 1
        
        # Mapping from (s, a, r, s, t) to episode_batch keys
        # Only the required keys contain legitimate information for now
        required_keys = ['o', 'u', 'r', 'o_2', 'g']
        key_to_index_map = {'o':0, 'u':1, 'r':2, 'o_2':3, 'g':4}

        # Create a dictionary instead of a list of transitions
        for key in required_keys:
        	transitions[key] = []
        	for t in range(len(sample_transitions)):
        		transitions[key].append(sample_transitions[key_to_index_map[key]])
        	transitions[key] = np.array(transitions[key])

        # For keys other than required keys, fill in some random values for now
        for key in episode_batch.keys():
        	if key not in required_keys:
	        	transitions[key] = []
	        	for t in range(len(sample_transitions)):
	        		transitions[key].append(-1)
	        	transitions[key] = np.array(transitions[key])


        # transitions = {key: sample_transitions[:,key_to_index_map[key]]
        #                for key in episode_batch.keys()}   

        # Check if the batch_size returned is the one expected
        assert(transitions['u'].shape[0] == batch_size_in_transitions), "Unexpected batch size returned by rank_based.py"

        # return the transitions
        return transitions, w, rank_e_id

    return _sample_her_transitions



    #     T = episode_batch['u'].shape[1]
    #     rollout_batch_size = episode_batch['u'].shape[0]
    #     batch_size = batch_size_in_transitions

    #     # Select which episodes and time steps to use.
    #     episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    #     t_samples = np.random.randint(T, size=batch_size)
    #     transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
    #                    for key in episode_batch.keys()}

    #     # Select future time indexes proportional with probability future_p. These
    #     # will be used for HER replay by substituting in future goals.
    #     her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
    #     future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
    #     future_offset = future_offset.astype(int)
    #     future_t = (t_samples + 1 + future_offset)[her_indexes]

    #     # Replace goal with achieved goal but only for the previously-selected
    #     # HER transitions (as defined by her_indexes). For the other transitions,
    #     # keep the original goal.
    #     future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
    #     transitions['g'][her_indexes] = future_ag

    #     # Reconstruct info dictionary for reward  computation.
    #     info = {}
    #     for key, value in transitions.items():
    #         if key.startswith('info_'):
    #             info[key.replace('info_', '')] = value

    #     # Re-compute reward since we may have substituted the goal.
    #     reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
    #     reward_params['info'] = info
    #     transitions['r'] = reward_fun(**reward_params)

    #     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
    #                    for k in transitions.keys()}

    #     assert(transitions['u'].shape[0] == batch_size_in_transitions)

    #     return transitions

    # return _sample_her_transitions
