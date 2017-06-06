import random


a={1:[1,2,3],2:[2,3,4]}

for k,v in a.iteritems():
    b=v
    b.remove(3)

print a


c=[1,2,3,4]

print random.sample(c,1)