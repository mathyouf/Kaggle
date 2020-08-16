import sys

arguments = len(sys.argv)

for argI in range(0,arguments):
    print('Argument %s: %a' %(argI,sys.argv[argI]))