from LineCode import LineCode
from LineGeometry import LineGeometry
from Line import Line
from typing import Dict, List
import numpy as np

class YBus:

    #Initialization Function
    def __init__(self, name):
        self.name=name

        #make list and dictionary for buses
        self.bus_order: List[str] = list()
        self.buses: Dict[str,Bus] = dict()
        self.connection_matrix=np.matrix([])
        self.yBusM=np.matrix([],complex)

        #Line and transformer classes
        self.lines: Dist[str, Line] = dict()
        self.transformers: Dist[str, Transformer] = dict()

        #add dictionary for bus voltage bases, in kV
        self.vBase:float=20
        self.slackBus=None

        #SBase in MVA
        self.SBase:float=100

#use dictionary to make sure that only new buses are added
    def add_bus(self,bus):
        if bus not in self.buses.keys():
            #if no buses set a default slack voltage of 20kV
            if bus_order.length()==0:
                set_Slack(bus,20)
            self.buses[bus] = Bus(bus)
            self.bus_order.append(bus)
            self.connection_matrix.reshape(Bus.numBuses,Bus.numBuses)
            self.yBusM.reshape(Bus.numBuses,Bus.numBuses)

    def add_Line(self,name,len,bus1,bus2,code:LineCode=LineCode(),geo:LineGeometry=LineGeometry()):
        self.lines[name] = Line(name,len,bus1,bus2,code,geo)
        self.add_bus(bus1)
        self.add_bus(bus2)
        ind1=self.bus_order.index(bus1)
        ind2=self.bus_order.index(bus2)
        self.connection_matrix[ind1,ind2]=-1
        self.connection_matrix[ind2,ind1]=1


###########################################################
    def add_Transformer(self,name):
        self.transformers[name]=1

    def set_SBase(self,S):
        self.sBase=S

    def set_Slack(self,slackBus,voltage):
        if slackBus in self.buses.keys():
            self.vBase=voltage
            self.slackBus=slackBus

    def findZBase(self,bus1):
        ind1 = self.bus_order.index(bus1)
        vMultiplier=1

        #Traverse connection matrix to find the equivalent Vbase

        return (self.vBase*vMultiplier)**2/self.sBase

Y = np.zeros((buses, buses))
#I'm not sure if this is how you step through the list properly, but just wanted to have something down
 line_list= list(self.lines)
    for lin in line_list:
        Zbase = line.bus1
        z_actual = line.Z
        y_shunt_actual = line.shuntY
        z_pu = z_actual/Zbase
        y_shunt_pu = y_shunt_actual/Zbase
        index1 = self.bus_order.index(line.bus1)
        index2 = self.bus_order.index(line.bus2)
        #might be some redundancies in adding to Y matrix
        Y[index1, index1] += (y_shunt_pu/2)+(1/z_pu)
        Y[index2, index2] += (y_shunt_pu/2)+(1/z_pu)
        Y[index1, index2] -= 1 / z_pu
        Y[index2, index1] -= 1 / z_pu

xfmr_list = list(self.transformers)
    for transformer in xfmr_list:
        zbase = findZbase(bus1)
        z_actual= (self.transformer.v_1**2)/self.transformer.s_rated
        X = z_actual*self.transformer.x_r_ratio
        R = z_actual/self.transformer.x_r_ratio
        #z_pu = ? not sure how to per unitize and combine the X and R
        index1 = self.bus_order.index(line.bus1)
        index2 = self.bus_order.index(line.bus2)
        #might be some redundancies in adding to matrix
        Y[index1, index1] += 1 / z_pu
        Y[index2, index2] += 1 / z_pu
        Y[index1, index2] -= 1 / z_pu
        Y[index2, index1] -= 1 / z_pu



