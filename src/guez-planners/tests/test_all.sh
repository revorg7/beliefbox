#! /bin/bash

ulimit -Sv 2000000
parallel -j-4 --eta './bin/test_all {1} {2} {3} {4} > ./singlePolicy/Chain-{1}-{2}-{3}-{4}-.txt' ::: 8 ::: 2 ::: 3 5 10 ::: 2
