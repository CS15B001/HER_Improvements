import os
import time
import datetime


# Mention the experiments that need to be run here
time_of_run = str(datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d_%H_%M_%S'))
batch_size = [256, 256]
replay_k = [4,8]
env_name = ['FetchPush-v1', 'FetchPush-v1']
n_epochs = [50, 50]
num_cpus = 12

# Run all the experiments
for i in range(len(batch_size)):
	print('Running index', i)
	os.system('nohup python3 -m baselines.her.experiment.train --num_cpu '+str(num_cpus)+' --env_name '+str(env_name[i])+
		' --batch_size '+str(batch_size[i])+' --n_epochs '+str(n_epochs[i])+' --replay_k '+str(replay_k[i])+' >log_file_'+str(i)+'_'+time_of_run+'.txt 2>&1')
