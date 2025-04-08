#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=4000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python/3.10.9-fasrc01

# run code
python -m venv /tmp/myenv_$SLURM_JOB_ID
source /tmp/myenv_$SLURM_JOB_ID/bin/activate
pip install -r ../../requirements.txt

# python extract_cough_features.py --audio_directory ~/cough_mp3_full/Files_5_mp3 --output_filename file5_output

python yamnet_evaluation_cough_intervals.py --audio_directory ~/AC297r/CoughAnalyzer/data_w_noise_white0.03_nbn10_nbd0.5_w-beeping_w-talking/ --output_filename data_w_noise_white0.03_nbn10_nbd0.5_w-beeping_w-talking_yamnet_cough_intervals

# python extract_cough_features.py --audio_directory Data --output_filename online_data_output
# python extract_cough_features.py --audio_directory ./fake_file --output_filename fake_output
# python extract_cough_features.py --audio_filename ~/cough_mp3_full/Files_11_mp3/X320112\ \(Month\ 20\).mp3