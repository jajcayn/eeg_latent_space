#!/bin/bash

target="praha1"

## for main
# pyscript="main_run_latent_per_subject.py"

# for datatype in "EC" "EO"
# do
#     for decomptype in "kmeans" "AAHC" "TAAHC" "PCA" "ICA" "hmm"
#     do
#         if [ $decomptype == "AAHC" ] || [ $decomptype == "TAAHC" ]; then
#             comp_time=48
#         else
#             comp_time=24
#         fi
#         qsub -l walltime=$comp_time:0:0 -q default@meta-pbs.metacentrum.cz -l select=1:ncpus=32:mem=120gb:scratch_local=10gb -v target=$target,pyscript=$pyscript,decomptype=$decomptype,datatype=$datatype meta_runner.sh
#     done
# done


# for running surrogates
pyscript="main_run_surrogates_per_subject.py"
for datatype in "EC" "EO"
do
    for decomptype in "kmeans" "PCA" "ICA" "hmm" #"AAHC" "TAAHC"
    do
        for surrtype in "FT" "AAFT" "IAAFT" "shuffle"
        do
            qsub -l walltime=24:0:0 -q default@meta-pbs.metacentrum.cz -l select=1:ncpus=32:mem=120gb:scratch_local=10gb -v target=$target,pyscript=$pyscript,decomptype=$decomptype,datatype=$datatype,surrtype=$surrtype meta_runner.sh
        done
    done
done
