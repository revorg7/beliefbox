#! /bin/bash

ulimit -Sv 2000000
parallel -j-4 --eta './bin/test_all {1} {2} {3} {4} {5} > ./correctPI/loop-{1}-{2}-{3}-{4}-{5}-.txt' ::: 4 ::: 4 ::: 5 9 18 ::: 1 0 ::: 1 2
