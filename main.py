#import classes
from LineCode import LineCode
from LineGeometry import LineGeometry
from YBus import YBus
#initialize partride conductors
partridge=LineCode('Partridge',0.642,0.0217,460,0.385)
geo1=LineGeometry(2,1.5,19.5,19.5,39)

#set up YBus Matrix and Add components
ybus=YBus('System 1')
ybus.set_SBase(100)
ybus.add_Transformer('T1','1','2',20,230,125,0.085,10)
ybus.add_Line('L2',25,'2','3',partridge,geo1)
ybus.add_Line('L1',10,'2','4',partridge,geo1)
ybus.add_Line('L3',20,'3','5',partridge,geo1)
ybus.add_Line('L4',20,'4','6',partridge,geo1)
ybus.add_Line('L5',10,'5','6',partridge,geo1)
ybus.add_Line('L6',35,'4','5',partridge,geo1)
ybus.add_Transformer('T2','7','6',18,230,200,0.105,12)
ybus.set_Slack('1',20)

#solve system and print results
ybus.solve()
ybus.print_matrix()

