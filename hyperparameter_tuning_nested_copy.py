import os
import time
import datetime


# Mention the experiments that need to be run here
time_of_run = str(datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d_%H_%M_%S'))
batch_size = [512, 256]
replay_k = [4,6,8]
env_name = ['FetchPush-v1', 'FetchPickAndPlace-v1']
n_epochs = [3, 50]
n_reps = 1
num_cpus = 12

# Run all the experiments
for env_i in range(len(env_name)):
	print("Running ", env_name[env_i])
	for batch_i in range(len(batch_size)):
		print("Batch Size ", batch_size[batch_i])
		for replay_i in range(len(replay_k)):
			print("replay_k ", replay_k[replay_i])
			os.system('nohup python3 -m baselines.her.experiment.train --num_cpu '+str(num_cpus)+' --env_name '+str(env_name[env_i])+ ' --n_epochs '+str(n_epochs[env_i]) \
				+' --batch_size '+str(batch_size[batch_i])+' --n_reps '+str(n_reps)+' --replay_k '+str(replay_k[replay_i]) \
				+' >log_file_'+str(env_name[env_i])+'_n_cycles_50_batch_size_'+str(batch_size[batch_i])+'_n_reps_'+str(n_reps)+'_replay_k_'+str(replay_k[replay_i])+'_'+time_of_run+'.txt 2>&1')
