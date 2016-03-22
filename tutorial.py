#!/usr/bin/env python2
from pomegranate import *
import matplotlib.pyplot as plt
import argparse
import numpy as np


# In[2]:

d1 = DiscreteDistribution({'10' : 0.8, '20' : 0.1, '30' : 0.1})
d2 = DiscreteDistribution({'10' : 0.1, '20' : 0.8, '30' : 0.1})
d3 = DiscreteDistribution({'10' : 0.1, '20' : 0.1, '30' : 0.8})

s1 = State(d1, name="NotTied1" )
s2 = State(d2, name="NotTied2" )
s3 = State(d3, name="NotTied3" )

model = HiddenMarkovModel()
model.add_states( [s1, s2] )
model.add_transition( model.start, s1, 0.8 )
model.add_transition( model.start, s2, 0.1 )
model.add_transition( model.start, s3, 0.1 )

model.add_transition( s1, s1, 0.33 )
model.add_transition( s1, s2, 0.33 )
model.add_transition( s1, s3, 0.34 )

model.add_transition( s2, s1, 0.33 )
model.add_transition( s2, s2, 0.33 )
model.add_transition( s2, s3, 0.34 )

model.add_transition( s3, s1, 0.33 )
model.add_transition( s3, s2, 0.33 )
model.add_transition( s3, s3, 0.34 )

model.bake()
corpus_good = [['10','10','10']]

print sum([model.log_probability(x) for x in corpus_good])


print model.draw()

model.train(Vcorpus)
print sum([model.log_probability(x) for x in corpus_good])




# In[ ]:




# In[ ]:



