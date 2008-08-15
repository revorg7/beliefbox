#! /bin/bash
#


results=results/belief_uct/bandit/uct_complete.out
experiments=1000
##echo "#meth #n_iter #gamma #actions #total #discounted #o_total #o_discounted" > $results

for actions in 2 4 #8
do
    for iter in  0 1 2 3 4 5 6 7 8 10 12 14 18 22 28 32 64
    do
      for method in 0 1 2 3 4 5 6 7 8 9 10 11 12 13
	do
	qsub -v"method=$method","iter=$iter","actions=$actions","experiments=$experiments","results=$results"  ./test_bandit_uct_sub.sh
        done
    done
done