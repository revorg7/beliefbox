#! /bin/bash

ulimit -Sv 2000000
parallel -j-4 --eta './bin/test_MCThomp {1} {2} {3} {4} > ./results/MCThomp-algo:{1}-rtdp:{2}-VI-.txt' ::: 0 1 ::: 0 1
