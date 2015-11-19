rm -rf data/BSR/detected data/BSR/detected_eval
./run.sh data/BSR/images/test/*.jpg data/BSR/detected
matlab -nodisplay -nosplash -nodesktop -r "cd data/BSR/bench; try, run('bench_bsds500.m');, catch me, disp(['Error ', me.identifier, ': ', me.message]), end, exit"
