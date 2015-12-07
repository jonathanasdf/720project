rm -rf data/BSR/detected data/BSR/detected_eval
./run.sh CrispEdgeDetection data/BSR/images/test/*.jpg data/BSR/detected
matlab -r "cd data/BSR/bench; run('bench_bsds500.m');"
