#!/bin/bash
#SBATCH --job-name=Zero_shot_7th_round
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            
#SBATCH --cpus-per-task=12              
#SBATCH --gres=gpu:h100:1            
#SBATCH --partition=gengpu               
#SBATCH --time=48:00:00   
#SBATCH --account=p32870        
#SBATCH --output=Zero_shot_7th_round.log            
#SBATCH --error=Zero_shot_7th_round.err    

python train_one_gpu_new_Step4_7th_round.py --use_clip True --clip_variant clip --prompt_file organ_prompts_shape_location.txt