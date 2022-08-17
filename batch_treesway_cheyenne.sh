#!/bin/bash
#PBS -N <job_name>
#PBS -A <account id>
#PBS -l walltime=<hh:mm:ss>
#PBS -q regular
#PBS -j oe
#PBS -k eod
#PBS -l select=2:ncpus=10:mpiprocs=10:ompthreads=1
#PBS -m abe
#PBS -M <email>

### Set TMPDIR as recommended
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

# set up conda environment
module load conda/latest
conda activate treesway

OUTPUT_DIR=<output_dir>
mkdir -p $OUTPUT_DIR

# Process each video in file segment
echo "Processing segments"
while read VIDEO; do
  echo "Processing $VIDEO"
  python3 treesway_vidproc.py $VIDEO $OUTPUT_DIR
done < $INPUT_FILE
