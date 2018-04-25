import os
import time
import datetime


# Mention the experiments that need to be run here
time_of_run = str(datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d_%H_%M_%S'))
batch_size = [256, 512]
replay_k = [4,6,8]
env_name = ['FetchReach-v1', 'FetchSlide-v1']
num_cpus = 12
n_reps = 1

# Run all the experiments
for i in range(len(batch_size)):
	os.system('nohup python3 -m baselines.her.experiment.train --num_cpu '+str(num_cpus)+' --env_name '+str(env_name[i])+ ' --n_epochs '+str(n_epochs[i]) \
		' --batch_size '+str(batch_size[i])+' --n_reps '+str(n_reps)+' --replay_k '+str(replay_k[i])+\
		' >log_file_'+str(env_name[i])+'_batch_size_'+str(batch_size[i])+'_replay_k_'+str(replay_k[i])+'_'+time_of_run+'.txt 2>&1')