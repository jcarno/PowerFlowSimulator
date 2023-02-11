from Bus import Bus
from LineCode import LineCode
from LineGeometry import LineGeometry
from Line import Line
from Transformer import Transformer
from typing import Dict, List
import numpy as np

class YBus:

    #Initialization Function
    def __init__(self, name):
        self.name=name

        #make list and dictionary for buses
        self.bus_order: List[str] = list()
        self.buses: Dict[str,Bus] = dict()
        self.connection_matrix=np.zeros((100,100),dtype=int)
        self.yBusM=np.zeros((100,100),dtype=complex)

        #Line and transformer classes
        self.lines: Dict[str, Line] = dict()
        self.transformers: Dict[str, Transformer] = dict()

        #add dictionary for bus voltage bases, in kV
        self.vBase:float=20
        self.slackBus=None

        #SBase in MVA
        self.SBase:float=100

#use dictionary to make sure that only new buses are added
    def add_bus(self,bus):
        if bus not in self.buses.keys():
            #if no buses set a default slack voltage of 20kV
            if self.bus_order.__len__()==0:
                self.set_Slack(bus,20)

            #add bus to list and dictionary, while updating connection and size matrices
            self.buses[bus] = Bus(bus)
            self.bus_order.append(bus)

    def add_Line(self,name,len,bus1,bus2,code:LineCode,geo:LineGeometry=LineGeometry()):
        #add line and buses to list
        self.lines[name] = Line(name,len,bus1,bus2,code,geo)
        self.add_bus(bus1)
        self.add_bus(bus2)

        #get indeces of connections and update connection matrix
        ind1=self.bus_order.index(bus1)
        ind2=self.bus_order.index(bus2)
        self.connection_matrix[ind1,ind2]=-1
        self.connection_matrix[ind2,ind1]=1

    def add_Transformer(self,name,bus1,bus2,v_1, v_2, s_rated, zt, x_r_ratio):
        #add transfomer and buses to list
        self.transformers[name]=Transformer(name,bus1,bus2,v_1,v_2,s_rated,zt,x_r_ratio)
        self.add_bus(bus1)
        self.add_bus(bus2)

        #get indeces and update connection matrix
        ind1 = self.bus_order.index(bus1)
        ind2 = self.bus_order.index(bus2)
        self.connection_matrix[ind1, ind2] = -2
        self.connection_matrix[ind2, ind1] = 2

    #set the sBase of the system - default 100 MVA
    def set_SBase(self,S):
        self.sBase=S

    #set slack bus, default is first bus added
    def set_Slack(self,slackBus,voltage):
        if slackBus in self.buses.keys():
            self.vBase=voltage
            self.slackBus=slackBus

    #function to get the zBase at a certain bus
    def findZBase(self,bus1):
        ind1 = self.bus_order.index(bus1)
        vMultiplier=self.getMultiplier(ind1)
        return (self.vBase*vMultiplier)**2/self.sBase

#get a scalar multiplier to adjust slack bus voltage to be the base voltage for that given bus
    def getMultiplier(self,indexGoal):
        #get total num of buses for looping
        length=Bus.numBuses

        #add to the list of multipliers as you go to string together a total change in bus voltage
        vMultipliers=[1]
        busPath=[]

        #make a list of buses to visit, a variable to hold the last index visited and the current index visited
        toVisit=[self.bus_order.index(self.slackBus)]
        visited=[]
        currIndex=self.bus_order.index(self.slackBus)

        #loop thorugh connection matrix as long as there are buses to visit and the current index isn't the goal index
        while toVisit.__len__() != 0:
            currIndex=toVisit.pop()
            busPath.append(currIndex)
            visited.append(currIndex)
            #break the loop if the index is found
            if currIndex==indexGoal:
                break

            #keep track of the number of additions to the toVisit, if none then step back in path
            numAdditions=0

            #cycle through the connection matrix, following the path
            for i in range(0,length):
                conMatValue=self.connection_matrix[currIndex,i]
                if currIndex != i and conMatValue != 0 and not hasVisited(visited,i) and not hasVisited(toVisit,i):
                    toVisit.append(i)
                    numAdditions+=1
                    if conMatValue == 1 or conMatValue==-1:
                        vMultipliers.append(1)

                    #if value in connection matrix is a 2 then transformer
                    #must adjust for voltage change
                    else:
                        bus1=self.bus_order[currIndex]
                        bus2=self.bus_order[i]
                        vMul=1

                        #find and add the turns ratio based on orientation/connection of transformer in system
                        for x in list(self.transformers.values()):
                            if (x.bus1==bus1 and x.bus2==bus2) or (x.bus1==bus2 and x.bus2==bus1):
                                vMul=float(x.v2)/x.v1
                        if conMatValue==-2:
                            vMultipliers.append(vMul)
                        else:
                            vMultipliers.append(1.0/vMul)

            #if no additions then take the last place visited out of the path matrix
            if numAdditions==0:
                busPath.pop()
                vMultipliers.pop()

        if vMultipliers.__len__()==0:
            print('Error: Path not Found')
            return 1
        else:
            multiplier =1
            for m in vMultipliers:
                multiplier*=m
            return multiplier


    #solve yBus matrix using algorithm from class
    def solve(self):
        #reset yBus Matrix
        self.yBusM = np.zeros((100, 100), dtype=complex)

        #take dictionary of lines and turn it into a list
        line_list= list(self.lines.values())

        #for each line, get the per unit impedance and admittance and add to the matrix
        for lin in line_list:
            Zbase = self.findZBase(lin.bus1)
            y_shunt_actual = lin.shuntY
            z_pu = lin.per_unit_Z(Zbase)
            y_shunt_pu = y_shunt_actual/(1/Zbase)
            index1 = self.bus_order.index(lin.bus1)
            index2 = self.bus_order.index(lin.bus2)


            #add elements to ybus matrix
            self.yBusM[index1, index1] += (y_shunt_pu/2)+(1/z_pu)
            self.yBusM[index2, index2] += (y_shunt_pu/2)+(1/z_pu)
            self.yBusM[index1, index2] -= 1 / z_pu
            self.yBusM[index2, index1] -= 1 / z_pu

        #repeat process for transformers
        xfmr_list = list(self.transformers.values())
        for xfmr in xfmr_list:
            zbase = self.findZBase(xfmr.bus1)
            z_pu = xfmr.per_unit_Z(zbase)
            index1 = self.bus_order.index(xfmr.bus1)
            index2 = self.bus_order.index(xfmr.bus2)
            #might be some redundancies in adding to matrix
            self.yBusM[index1, index1] += 1 / z_pu
            self.yBusM[index2, index2] += 1 / z_pu
            self.yBusM[index1, index2] -= 1 / z_pu
            self.yBusM[index2, index1] -= 1 / z_pu

    #funciton to print the matrix
    def print_matrix(self):
        print('YBus Matrix for ' + self.name)
        output:str=''
        for b in self.bus_order:
            output+=b + '\t\t'
        print(output)
        for r in range(0,Bus.numBuses):
            output=''
            for c in range(0,Bus.numBuses):
                output+=str(round(self.yBusM[r,c].real,3)+1j*round(self.yBusM[r,c].imag,3)) + '\t'
            print(output)
def hasVisited(visited:list(),ind):
    if ind in visited:
        return 1
    else:
        return 0
