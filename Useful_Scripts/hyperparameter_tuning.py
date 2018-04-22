import os
import time
import datetime


# Mention the experiments that need to be run here
time_of_run = str(datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d_%H_%M_%S'))
batch_size = [256, 512]
replay_k = [4,6,8]
env_name = ['FetchReach-v1', 'FetchSlide-v1']
num_cpus = 12

# Run all the experiments
for i in range(len(batch_size)):
	os.system('nohup python3 -m baselines.her.experiment.train --num_cpu '+str(num_cpus)+' --env_name '+str(env_name[i])+\
		' --batch_size '+str(batch_size[i])+' --replay_k '+str(replay_k[i])+' >log_file_'+str(i)+'_'+time_of_run+'.txt 2>&1')