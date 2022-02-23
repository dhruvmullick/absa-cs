#!/bin/sh
source $HOME/absa/bin/activate && module load python/3.9 StdEnv/2020 gcc/9.3.0 arrow/5.0.0
echo "CS = $CS"
python -u commongen_absa_multi.py $CS "WIKITEXT"
