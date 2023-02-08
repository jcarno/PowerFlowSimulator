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
        self.yBusM=np.matrix([])

        #Line and transformer classes
        self.lines: Dist[str, Line] = dict()
        self.transformers: Dist[str, Transformer] = dict()

        #add dictionary for bus voltage bases, in kV
        self.vBase=13.8
        self.slackBus=None

        #SBase in MVA
        self.SBase=100

#use dictionary to make sure that only new buses are added
    def add_bus(self,bus):
        if bus not in self.buses.keys():
            #if no buses set a default slack voltage of 13.8kV
            if bus_order.length()==0:
                set_Slack(bus,13.8)
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

    def findZBase(self,bus1,bus2):
        ind1 = self.bus_order.index(bus1)
        ind2 = self.bus_order.index(bus2)

        #return a failure i buses are not connected
        if connection_matrix[ind1,ind2]==0 and not ind1==ind2:
            return 0

        #use other function to get vbase



