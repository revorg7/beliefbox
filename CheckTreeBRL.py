from wrap2 import derived
d = derived(5,2,0.7)
d.Act(0.3,2)

#Checking cloning works right
from wrap3 import sparseMDP
mdp = sparseMDP(5,2)
mdp.add(0,0,1,1)
mdp.show()

a = mdp.pclone()
a.add(0,1,.5,4)
a.show()


mdp.show()

