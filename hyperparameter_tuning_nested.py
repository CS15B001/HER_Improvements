import os
import time
import datetime


# Mention the experiments that need to be run here
time_of_run = str(datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d_%H_%M_%S'))
batch_size = [256]
replay_k = [4,6,8]
env_name = ['FetchSlide-v1']
n_epochs = [50]
num_cpus = 12

# Run all the experiments
for env_i in range(len(env_name)):
	print('Running Environment: '+env_name[env_i])
	for batch_i in range(len(batch_size)):
		print('Batch Size: '+str(batch_size[batch_i]))
		for replay_i in range(len(replay_k)):
			print('repaly_k: '+str(replay_k[replay_i]))
			os.system('nohup python3 -m baselines.her.experiment.train --num_cpu '+str(num_cpus)+' --env_name '+str(env_name[env_i])+ ' --n_epochs '+str(n_epochs[env_i])+' --batch_size '+str(batch_size[batch_i])+' --replay_k '+str(replay_k[replay_i])+' >log_file_'+env_name[env_i]+'_b-'+str(batch_size[batch_i])+'_r-'+str(replay_k[replay_i])+'_'+time_of_run+'.txt 2>&1')
