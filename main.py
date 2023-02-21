#import classes
from LineCode import LineCode
from LineGeometry import LineGeometry
from PowerSystem import PowerSystem
#initialize partride conductors
partridge=LineCode('Partridge',0.642,0.0217,460,0.385)
geo1=LineGeometry(2,1.5,19.5,19.5,39)

#set up YBus Matrix and Add components
system1=PowerSystem('System 1')
system1.set_SBase(100)
system1.add_Generator('G1','1',1,100)
system1.add_Transformer('T1','1','2',20,230,125,0.085,10)
system1.add_Line('L2',25,'2','3',partridge,geo1)
system1.add_Line('L1',10,'2','4',partridge,geo1)
system1.add_Line('L3',20,'3','5',partridge,geo1)
system1.add_Line('L4',20,'4','6',partridge,geo1)
system1.add_Line('L5',10,'5','6',partridge,geo1)
system1.add_Line('L6',35,'4','5',partridge,geo1)
system1.add_Transformer('T2','7','6',18,230,200,0.105,12)
system1.add_Generator('G2','7',1,200)
system1.add_Load('Z3','3',110,50)
system1.add_Load('Z4','4',100,70)
system1.add_Load('Z5','5',100,65)
system1.set_Slack('1',20)

#solve system and print results
system1.solve()
system1.print_Bus_Voltages()
