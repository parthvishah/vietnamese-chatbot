Using Prince
==============================

Instructions for running on New York University's Prince computer cluster.


Setup
------------

1. SSHing in
	* (Within NYU) Login Node:  `ssh netid@prince.hpc.nyu.edu`
	* (Outside NYU) Access Node: `ssh netid@gw.hpc.nyu.edu`
	* If you are within NYU network, you can SSH directly into the Login Node, otherwise you should first SSH into the Access Node, and then into the Login Node

2. Move to your scratch folder `cd /scratch/netid`

3. Create nlp_project directory `mkdir nlp_project`

4. Create outputs folder and saved_models folder within nlp_project directory
	* `mkdir saved_models`
	* `mkdir outputs`
	* `mkdir log`

5. Clone repository within nlp_project directory `git clone https://github.com/anhthyngo/vietnamese-chatbot.git`

6. Perform the following commands
	*  `module purge`
	* `module load anaconda3/5.3.1`
	* `module load cuda/10.0.130`
	* `module load gcc/6.3.0`

7. In cloned repository, create anaconda environment `nmt_env` from `environment.yml`

   `conda env create -f environment.yml`

8. Once the environment is created and you are within the root project directory, create copy of `seq2seq_train_1gpu.s` to include your netid in the file name and within the file.

9.  Run job `sbatch nmt_no_attention_1gpu_netid.s`
	* You can check on the job by `squeue -u netid`

