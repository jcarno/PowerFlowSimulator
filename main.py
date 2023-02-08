#import classes
from LineCode import LineCode
from LineGeomtry import LineGeometry
from YBus import YBus
#initialize partride conductors
partridge=LineCode('Partridge',0.642,0.0217,460,0.385)
geo1=LineGeometry(2,1.5,19.5,19.5,39)

#set up YBus Matrix and Add components
ybusM=YBus('System 1')
ybusM.add_Line('Line1',10,'2','4',partridge,geo1)


