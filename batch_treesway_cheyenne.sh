#!/bin/bash
#PBS -N treesway_batch
#PBS -A TODO --> specify account
#PBS -l walltime=00:20:00
#PBS -q regular
#PBS -j oe
#PBS -k eod

### TODO: customize resources
#PBS -l select=2:ncpus=10:mpiprocs=10:ompthreads=1

### Send email on abort, begin and end
#PBS -m abe
### Specify mail recipient
#PBS -M TODO --> email address

### Set TMPDIR as recommended
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

# setup python environment
module load conda/latest
# TODO: activate custom conda environment

# load GNU parallel
module load parallel/20190622

export OMP_NUM_THREADS=1

OUTPUT_DIR=batch_frequency_output

parallel -a python3 treesway_vidproc.py {1} $OUTPUT_DIR :::: treesway_video_paths.txt