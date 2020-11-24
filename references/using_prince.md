Using Prince
==============================

Instructions for setting up our working directory and running slurm jobs on New York University's Prince computer cluster.


Setup
------------

1. SSHing in
	* (Within NYU) Login Node:  `ssh <netid>@prince.hpc.nyu.edu`
	* (Outside NYU) Access Node: `ssh <netid>@gw.hpc.nyu.edu`
	* If you are within NYU network, you can SSH directly into the Login Node, otherwise you should first SSH into the Access Node, and then into the Login Node

2. Move to your scratch folder `cd /scratch/<netid>`

3. Create nlp_project directory `mkdir nlp_project`

4. Create outputs folder and saved_models folder within nlp_project directory
	* `mkdir saved_models`
	* `mkdir outputs`
	* `mkdir log`
	* `mkdir plots`

5. Clone repository within nlp_project directory `https://github.com/anhthyngo/nmt-vi-en.git`

6. Perform the following commands
	*  `module purge`
	* `module load anaconda3/5.3.1`
	* `module load cuda/10.0.130`
	* `module load gcc/6.3.0`

7. In cloned repository, create anaconda environment `nmt_env` from `environment.yml`

   `conda env create -f environment.yml`

8. When running sbatch jobs make sure you are within the sbatch directory (`/scratch/an3056/nlp_project/nmt-vi-en/sbatch`).

9. Create a copy of `lstm_attention_vi2en.s` and rename it as `{rnn_type}_{attention/no_attention}_vi2en.s` based on what configuration you are running. You will need to also change the hyperparameters and other arguments to your netid within the file. 

10. Make sure the `STUDY_NAME` argument is in the following format `{attn/no_attn}_{rnn_type}_{batch_size}_{embedding_size}_{num_RNN_layers}` so that the generated files are easier to maintain.

9.  Run job `sbatch {rnn_type}_{attention/no_attention}_vi2en.s`
	* You can check on the job by `squeue -u <netid>`
	* You can kill a running job by `scancel jobid`
	* You can cancel all of your jobs by `scancel -u <netid>`

10. If you need to rerun a configuration over again, you may need to delete generated files such as `lang_obj.pkl` and the encoder/decoder weight files in `/scratch/<netid>/nlp_project/saved_models/vi2en`.

