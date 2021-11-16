#!/bin/bash
#PBS -m abe

# arguments:
#     - target
#     - pyscript


if [ $target == 'praha1' ]; then
    dir_prefix=/storage/praha1/home/jajcayn
elif [ $target == 'brno2' ]; then
    dir_prefix=/storage/brno2/home/jajcayn
# elif [ $target == 'plzen1' ]; then
#     dir_prefix=/storage/plzen1/home/jajcayn
# elif [ $target == 'brno3-cerit' ]; then
#     dir_prefix=/storage/brno3-cerit/home/jajcayn
else
    exit 1
fi

trap 'cp -r $SCRATCHDIR $RESULTSDIR && clean_scratch' TERM
module add python-3.6.2-gcc

export CODEBASEDIR=$dir_prefix/work-eeg-latent/code
export PYTHONUSERBASE=$dir_prefix/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$CODEBASEDIR:$PYTHONUSERBASE/lib/python3.6/site-packages:$PYTHONPATH

EXPERIMENTSDIR=$dir_prefix/work-eeg-latent/experiments
RESULTSDIR=$dir_prefix/work-eeg-latent/results
DATADIR=$dir_prefix/work-eeg-latent/data

if [ ! -d "$SCRATCHDIR" ] ; then echo "Adresar pro SCRATCH neni vytvoren!" 1>&2; exit 1; fi

if [ -d $DATADIR ]; then
  cp -r $DATADIR $SCRATCHDIR/
fi

cp -r $EXPERIMENTSDIR $SCRATCHDIR/

cd $SCRATCHDIR || exit 2

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> job_info.txt
# for main
python3 -u experiments/$pyscript $SCRATCHDIR/data/data_LEMON $decomptype --no_states 4 --filter 2.0 20.0 --data_type $datatype --use_gfp --workers 30

# for running surrogates
# python3 -u experiments/$pyscript $SCRATCHDIR/data/data_LEMON $decomptype $surrtype --no_states 4 --filter 2.0 20.0 --data_type $datatype --use_gfp --workers 30

# for EC/EO diff
# python3 -u experiments/$pyscript $SCRATCHDIR/data/data_LEMON --workers 30

cp -r $SCRATCHDIR $RESULTSDIR || export CLEAN_SCRATCH=false

clean_scratch
