import numpy as np

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
        batch_size = batch_size_in_transitions
        # Sample from the given priority_queue
        # global_step represents the step of the learning process needed for annealing the bias
        sample_transitions, w, rank_e_id = priority_queue.sample(global_step)

        # sample_transitions is now a list of transitions, convert it to the usual {key: batch X dim_key}
        keys = sample_transitions[0].keys()
        # print("Keys in _sample_her_transitions are: "+str(keys))
        transitions = {}
        for key in keys:
            # Initialize for all the keys
            transitions[key] = []

            # Remove
            debug_count = 0

            # Add transitions one by one to the list
            for single_transition in range(len(sample_transitions)):
                ##################################
                if key not in sample_transitions[single_transition].keys():
                    print("Ran into problems in her_per. Keys are: "+str(sample_transitions[single_transition].keys()))
                    print("The transition is: "+str(sample_transitions[single_transition]))
                    print("Debug Count is: "+str(debug_count))
                else:
                    debug_count += 1
                ##################################
                transitions[key].append(sample_transitions[single_transition][key])
            transitions[key] = np.array(transitions[key])

        # transitions is now of the expected format, need to add rewards
        # Re-compute reward since we may have substituted the goal.

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # print("The keys in transitions are: "+str(transitions.keys()))
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}   

        # Check if the batch_size returned is the one expected
        assert(transitions['u'].shape[0] == batch_size_in_transitions), "Unexpected batch size returned by rank_based.py"

        # return the transitions
        return transitions, w, rank_e_id

    return _sample_her_transitions



        # # Create a dictionary
        # # key_to_index_map maps the keys of episode batch to the corresponding
        # # index in sample_transitions. For example, if sample transitions returns 
        # # (s,a,r,s,t), key_to_index_map['u'] = 1
        
        # # Mapping from (s, a, r, s, t) to episode_batch keys
        # # Only the required keys contain legitimate information for now
        # required_keys = ['o', 'u', 'r', 'o_2', 'g', 'ag']
        # key_to_index_map = {'o':0, 'u':1, 'r':2, 'o_2':3, 'g':4}

        # # Create a dictionary instead of a list of transitions
        # for key in required_keys:
        #   transitions[key] = []
        #   for t in range(len(sample_transitions)):
        #       transitions[key].append(sample_transitions[key_to_index_map[key]])
        #   transitions[key] = np.array(transitions[key])

        # # For keys other than required keys, fill in some random values for now
        # for key in episode_batch.keys():
        #   if key not in required_keys:
           #    transitions[key] = []
           #    for t in range(len(sample_transitions)):
           #        transitions[key].append(-1)
           #    transitions[key] = np.array(transitions[key])