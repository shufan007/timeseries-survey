#!/usr/bin/env bash

source ~/.bashrc
export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`

export PATH="/home/luban/miniconda3/bin:$PATH"
conda activate base

python3 $@
