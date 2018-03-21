import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        # future_p is the probability of choosing to perform 'future' HER as against just ER
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """

        # T refers to the number of timesteps the episode was played for
        T = episode_batch['u'].shape[1]

        f = open('reward_debug.txt', 'a')
        f.write("Shape of the state space: "+str(episode_batch['o'].shape)+"\n")
        # rollout_batch_size is the number of episodes in the batch
        rollout_batch_size = episode_batch['u'].shape[0]
        f.write("Rollout batch size is: "+str(rollout_batch_size)+"\n")
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use. - OpenAI

        # 'batch_size' number of elements, episode_batch[key][episode_idxs[i], t_samples[i]]
        # are sampled
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        # transitions[key] is now an array of samples
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.

        # future_p fraction of samples will be True and hence those many will be used for HER replay
        # np.where returns indices where the condition is true
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        # (t_samples + 1 + future_offset) will vary from t_samples to T
        # This statement randomly selects a future goal, wrt current time-step t_samples
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        ######Debug
        # f = open('reward_debug.txt', 'a')
        f.write("Keys of reward params are: "+str(reward_params.keys())+"\n")
        f.write("Shape of info is: "+str(reward_params['info']['is_success'].flatten().shape)+"\n")
        f.write("Shape of ag_2 is: "+str(reward_params['ag_2'].shape)+"\n")
        # f.write("The keys of info: "+str(reward_params['info']['is_success'].shape)+"\n")
        # f.write("New Observation\n")
        ######
        transitions['r'] = reward_fun(**reward_params)

        # f.write("\nThe reward is: "+str(transitions['r'])+"\n")
        # f.write("\nInfo Success is: "+str(reward_params['info']['is_success'].flatten())+"\n")

        # f.write("Are reward and info the same: "+str(np.array_equal(transitions['r'], reward_params['info']['is_success'].flatten()))+"\n")

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        f.write("The shape of the reward is: "+str(transitions['r'].shape)+"\n")
        f.write("The shape of goals is: "+str(transitions['g'].shape)+"\n")
        f.write("New Observation\n\n")

        f.close()

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
