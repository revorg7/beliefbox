#! /bin/bash

ulimit -Sv 2000000
parallel -j-4 --eta './bin/test_all {1} {2} {3} {4} > ./results/doubleloop-{1}-{2}-{3}-{4}-RTDP.txt' ::: 2 4 ::: 2 ::: 5 9 18 ::: 1 2
