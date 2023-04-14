#import classes
from LineCode import LineCode
from LineGeometry import LineGeometry
from PowerSystem import PowerSystem
from Solution import Solution
from typing import Dict
import csv

#set up system
system1=PowerSystem('System 1')
system1.set_SBase(100)
lineCodes: Dict[str,LineCode]=dict()
lineGeometries: Dict[str,LineGeometry]=dict()

#
# #read in Data
#get line codes
with open('LineCodes.csv') as fp:
    reader=csv.reader(fp)
    for row in reader:
        lineCodes[row[0]]=LineCode(row[0],float(row[1]),float(row[2]),float(row[3]),float(row[4]))

#get Line Geometries
with open('LineGeometries.csv') as fp:
    reader = csv.reader(fp)
    for row in reader:
        lineGeometries[row[0]] = LineGeometry(float(row[1]), float(row[2]), float(row[3]), float(row[4]),float(row[5]))

#set up Buses
with open('buses.csv') as fp:
    reader = csv.reader(fp)
    for row in reader:
        system1.add_bus(row[0],(row[3]),float(row[1]),float(row[2]))
        if (row[3]=='S'):
            system1.set_Slack(row[0],float(row[4]))


#get Generators
with open('Generators.csv') as fp:
    reader = csv.reader(fp)
    for row in reader:
        system1.add_Generator(row[0],row[1], float(row[2]), float(row[3]), float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[9]),float(row[10]),float(row[11]),bool(row[12]),float(row[13]))

#get Lines
with open('Lines.csv') as fp:
    reader = csv.reader(fp)
    for row in reader:
        code=lineCodes[row[4]]
        geo=LineGeometry()
        if (lineGeometries.keys().__contains__(row[5])):
            geo=lineGeometries[row[5]]

        system1.add_Line(row[0],float(row[1]), (row[2]), (row[3]),code,geo)


#get Transformers
with open('Transformers.csv') as fp:
    reader = csv.reader(fp)
    for row in reader:
        system1.add_Transformer(row[0],row[1], (row[2]), float(row[3]), float(row[4]),float(row[5]),float(row[6]),float(row[7]),row[8],row[9],float(row[10]),float(row[11]))

# get Loads
with open('Loads.csv') as fp:
    reader = csv.reader(fp)
    for row in reader:
        system1.add_Load(row[0], row[1], float(row[2]), float(row[3]))

#configure system in solution class
soln1=Solution(system1)

# #solve system and print results
# system1.setFlatStart()
# print('Newton Raphson Solver:')
# soln1.solveNewtonRaphson()
# print('-----------------------------------------------------')
# system1.setFlatStart()
# print('Fast Decoupled Newton Raphson Solver:')
# soln1.solveFastDecoupled()
# print('-----------------------------------------------------')
# system1.setFlatStart()
# print('DC Power Flow Solver:')
# soln1.solveDCPowerFlow()
system1.get_ZBus012()
system1.print_ZBuses()
