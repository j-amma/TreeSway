#!/bin/bash

# Specift directory to store file segments
SEGMENT_DIR=./input_segments/
mkdir -p $SEGMENT_DIR

# define prefix for segment names
SEGMENT_PREFIX=segment_

# Specift number of nodes (number of file segments to generate)
NUM_NODES=3

# Create new file segment for each node
NUM_LINES=$(wc -l < $1)
LINES_PER_FILE=$(($NUM_LINES / $NUM_NODES + 1))

split -l $LINES_PER_FILE $1 $SEGMENT_DIR$SEGMENT_PREFIX

# Submit job for each node
for FILE in $SEGMENT_DIR*; 
do 
    # submit NCAR Bash script
    qsub -v "INPUT_FILE=$FILE" treesway_submit_cheyenne.sh
done
