import math

from Bus import Bus
from LineCode import LineCode
from LineGeometry import LineGeometry
from Line import Line
from Transformer import Transformer
from Generator import Generator
from Load import Load
from typing import Dict, List
import numpy as np
import pandas as pd

class PowerSystem:

    #Initialization Function
    def __init__(self, name):
        self.name=name

        #make list and dictionary for buses
        self.bus_order: List[str] = list()
        self.buses: Dict[str,Bus] = dict()
        self.connection_matrix=None
        self.yBusM=None

        #Line and transformer classes
        self.lines: List[Line] = list()
        self.transformers: List[Transformer] = list()
        self.generators:List[Generator]=list()
        self.loads:List[Load]=list()

        #add dictionary for bus voltage bases, in kV
        self.vBase:float=20
        self.slackBus=None

        #SBase in MVA
        self.SBase:float=100

#use dictionary to make sure that only new buses are added
    def add_bus(self,bus,type='PQ',voltage=1.0,angle=0.0):
        if bus not in self.buses.keys():
            #if no buses set a default slack voltage of 20kV
            if self.bus_order.__len__()==0:
                self.set_Slack(bus,20)
                self.connection_matrix=np.zeros((1,1),dtype=int)
                self.yBusM=np.zeros((1,1),dtype=complex)
                type='S'
            else:
                a=np.zeros((1,Bus.numBuses))
                b=np.zeros((Bus.numBuses+1,1))
                self.connection_matrix=np.concatenate((self.connection_matrix,a),axis=0)
                self.connection_matrix=np.concatenate((self.connection_matrix,b),axis=1)
                self.yBusM = np.concatenate((self.yBusM, a), axis=0)
                self.yBusM = np.concatenate((self.yBusM, b), axis=1)

            #add bus to list and dictionary, while updating connection and size matrices
            self.buses[bus] = Bus(bus,voltage,angle,type)
            self.bus_order.append(bus)
        elif type=='PV' and self.buses[bus].type!='S':
                self.buses[bus].type=type

    def add_Line(self,name,len,bus1,bus2,code:LineCode,geo:LineGeometry=LineGeometry()):
        #add line and buses to list
        self.lines.append(Line(name,len,bus1,bus2,code,geo))
        self.add_bus(bus1)
        self.add_bus(bus2)

        #get indeces of connections and update connection matrix
        ind1=self.bus_order.index(bus1)
        ind2=self.bus_order.index(bus2)
        self.connection_matrix[ind1,ind2]=-1
        self.connection_matrix[ind2,ind1]=1

    def add_Transformer(self,name,bus1,bus2,v_1, v_2, s_rated, zt, x_r_ratio):
        #add transfomer and buses to list
        self.transformers.append(Transformer(name,bus1,bus2,v_1,v_2,s_rated,zt,x_r_ratio))
        self.add_bus(bus1)
        self.add_bus(bus2)

        #get indeces and update connection matrix
        ind1 = self.bus_order.index(bus1)
        ind2 = self.bus_order.index(bus2)
        self.connection_matrix[ind1, ind2] = -2
        self.connection_matrix[ind2, ind1] = 2

    #add a generator to the system
    def add_Generator(self,name,bus,voltage,P_setpoint,Q_setpoint=0,Q_min=-10000,Q_max=10000):
        # add generator and buses to list
        self.generators.append(Generator(name, bus, voltage, P_setpoint,Q_setpoint,Q_min,Q_max))
        self.add_bus(bus,'PV',voltage)

    #add load to the system
    def add_Load(self,name,bus, P,Q):
        # add generator and buses to list
        self.loads.append(Load(name,bus, P,Q))
        self.add_bus(bus)

    #set the sBase of the system - default 100 MVA
    def set_SBase(self,S):
        self.sBase=S

    def set_pu_Voltage(self,bus,v,angle=0):
        b=self.buses(bus)
        b.set_bus_voltage(v,angle)

    #set slack bus, default is first bus added
    def set_Slack(self,slackBus,voltage):
        wasGenBus=False
        for g in self.generators:
            if g.bus==self.slackBus:
                wasGenBus=True
        if slackBus in self.buses.keys():
            if (self.slackBus!=None):
                oldSlack=self.buses[self.slackBus]
                oldSlack.type='PV'
            newSlack=self.buses[slackBus]
            self.vBase=voltage
            self.slackBus=slackBus
            newSlack.type='S'

    #function to get the zBase at a certain bus
    def findZBase(self,bus1):
        ind1 = self.bus_order.index(bus1)
        vMultiplier=self.getVMultiplier(ind1)
        return (self.vBase*vMultiplier)**2/self.sBase

#get a scalar multiplier to adjust slack bus voltage to be the base voltage for that given bus
    def getVMultiplier(self,indexGoal):
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
                        for x in self.transformers:
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

    def make_YBus(self):
        # reset yBus Matrix
        self.yBusM = np.zeros((100, 100), dtype=complex)

        # for each line, get the per unit impedance and admittance and add to the matrix
        for lin in self.lines:
            Zbase = self.findZBase(lin.bus1)
            y_shunt_actual = lin.shuntY
            z_pu = lin.per_unit_Z(Zbase)
            y_shunt_pu = y_shunt_actual / (1 / Zbase)
            index1 = self.bus_order.index(lin.bus1)
            index2 = self.bus_order.index(lin.bus2)

            # add elements to ybus matrix
            self.yBusM[index1, index1] += (y_shunt_pu / 2) + (1 / z_pu)
            self.yBusM[index2, index2] += (y_shunt_pu / 2) + (1 / z_pu)
            self.yBusM[index1, index2] -= 1 / z_pu
            self.yBusM[index2, index1] -= 1 / z_pu

        # repeat process for transformers
        for xfmr in self.transformers:
            zbase = self.findZBase(xfmr.bus1)
            z_pu = xfmr.per_unit_Z(zbase)
            index1 = self.bus_order.index(xfmr.bus1)
            index2 = self.bus_order.index(xfmr.bus2)
            # might be some redundancies in adding to matrix
            self.yBusM[index1, index1] += 1 / z_pu
            self.yBusM[index2, index2] += 1 / z_pu
            self.yBusM[index1, index2] -= 1 / z_pu
            self.yBusM[index2, index1] -= 1 / z_pu
    #funciton to print the matrix
    def print_YBus(self):
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

    def print_Results(self):
        # output results
        # voltages
        print('System Parameters:')
        self.print_Bus_Voltages()
        # calculated powers
        powers = self.calculatedPower()*self.sBase
        print(' ')
        self.print_Bus_Power(powers)

        # Line currents
        lineCurrents = self.getLineCurrents()
        percentAmpacity=self.getAmpacityPercent(lineCurrents)
        self.print_Line_Currents(lineCurrents,percentAmpacity)

        # Total Power Losses
        lineLosses = self.getLineLosses()
        xfmrLosses=self.getXfmrLosses()
        totalLoss = np.sum(lineLosses[:][0], axis=0)+np.sum(xfmrLosses[:][0], axis=0)
        self.print_Losses(lineLosses,xfmrLosses,totalLoss)

    def getCurrentAngles(self, lineCurrents):
        currentangles = np.zeros((len(self.lines), 1))
        for k in range(0, len(lineCurrents)):
            iAngle = np.angle(lineCurrents[k]) * 180 / math.pi
            if (iAngle > 90):
                iAngle = iAngle - 180
            elif (iAngle < -90):
                iAngle = iAngle + 180
            currentangles[k] = (np.round(iAngle, 3))
        return currentangles

    def getCurrentMag(self, lineCurrents):
        currentMag = np.zeros((len(self.lines), 1))
        for k in range(0, len(lineCurrents)):
            currentMag[k] = (np.round(np.abs(lineCurrents[k]), 3))
        return currentMag

    def get_Bus_Voltages(self):

        bus_volts = np.zeros((len(self.bus_order), 1))
        for i in range(0, Bus.numBuses):
            bus_volts[i] = (round(np.absolute(self.buses[self.bus_order[i]].voltage), 3))
        return bus_volts

    def get_Volt_Angles(self):
        bus_angles = np.zeros((len(self.bus_order), 1))
        for i in range(0, Bus.numBuses):
            bus_angles[i] = (round(self.buses[self.bus_order[i]].angle * 180 / math.pi, 3))
        return bus_angles

    def print_Line_Currents(self,lineCurrents,percentAmpacity):
        print('Transmission Line Currents for ' + self.name)
        output: str = ''

        # print header for all lines
        for l in self.lines:
            output += l.name + '\t\t'
        print(output)
        output = ''

        # print all currents
        for k in range(0, len(lineCurrents)):
            output += str(np.round(np.abs(lineCurrents[k][0]), 3)) + '\t'
        print(output)
        output = ''
        # print all current angles
        for k in range(0, len(lineCurrents)):
            iAngle=np.angle(lineCurrents[k][0])*180/math.pi
            if (iAngle>90):
                iAngle=iAngle-180
            elif (iAngle<-90):
                iAngle=iAngle+180
            output += str(np.round(iAngle, 3)) + '\t'
        print(output)
        output = ''

        # print all ampacity
        print(' ')
        print('Percent Ampacity for Transmission Lines for ' + self.name)
        # print header for all lines
        for l in self.lines:
            output += l.name + '\t\t'
        print(output)
        output = ''
        for k in range(0, len(percentAmpacity)):
            output += str(np.round(percentAmpacity[k][0], 3)) + '\t'
        print(output)
        output = ''

    def print_Bus_Power(self, powers):
        print('Bus Real (MW) and Reactive Power (Mvar) for ' + self.name)
        output: str = ''

        # print header for all buses
        for b in self.bus_order:
            output += b + '\t\t'
        print(output)
        output = ''

        # print all power
        for k in range(0, int(len(powers)/2)):
            output += str(np.round(powers[k][0], 3)) + '\t'
        print(output)
        output = ''

        # print all reactive power
        for k in range(0, int(len(powers)/2)):
            output += str(np.round(powers[k+int(len(powers)/2)][0], 3)) + '\t'
        print(output)

    def print_Results(self):
        # output results
        # Print bus voltages and angles
        print('System Parameters:')
        bus_volts = self.get_Bus_Voltages()
        bus_angles = self.get_Volt_Angles()
        # self.print_Bus_Voltages()
        bf = pd.DataFrame()
        bf['Voltage (MV)'] = bus_volts.flatten()
        bf['Angle (deg)'] = bus_angles.flatten()
        bf.index = range(1, len(self.bus_order) + 1)
        bf.index.name = 'Bus'
        print(bf)
        print('')

        # Print Bus Powers
        powers = self.calculatedPower() * self.sBase
        print(' ')
        # self.print_Bus_Power(powers)
        print('Bus Powers:')
        half = len(powers) // 2
        real = powers[:half]
        reactive = powers[half:]
        df = pd.DataFrame()
        df['Real (MW)'] = real.flatten()
        df['Reactive (Mvar)'] = reactive.flatten()
        df.index = range(1, len(df) + 1)
        df.index.name = 'Bus'
        print(df)
        print('')

        # Print Line currents and angles
        lineCurrents = self.getLineCurrents()
        fromToBuses = self.getCurrentDirections()
        percentAmpacity = self.getAmpacityPercent(lineCurrents)
        currentMag = self.getCurrentMag(lineCurrents)
        currentangles = self.getCurrentAngles(lineCurrents)
        # col1_data = currentMag[1, :]
        # col2_data = currentangles[1, :]
        # Print Current and Angles
        print('Line Currents and Angles:')
        cf = pd.DataFrame()
        cf['Current (A)'] = currentMag.flatten()
        cf['Angle (deg)'] = currentangles.flatten()
        cf.index = range(1, len(self.lines) + 1)
        cf.index.name = 'Line'
        print(cf)
        print('')

        # Print Ampacity
        print('Transmission Line Ampacity:')
        af = pd.DataFrame(percentAmpacity)
        af.columns = ['% Ampacity']
        af.index = range(1, len(af) + 1)
        af.index.name = 'Line'
        print(af)
        print('')
        # df = pd.DataFrame(lineCurrents)
        # df.columns = ['Line', 'Current (A)']
        # print(df)

        # Total Power Losses
        print('Transmission Line Power Losses:')
        lineLosses = self.getLineLosses()
        lf = pd.DataFrame(lineLosses)
        lf.columns = ['Losses (kW)']
        lf.index = range(1, len(lf) + 1)
        lf.index.name = 'Line'
        print(lf)
        print('')

        # Transformer Power Losses
        xfmrLosses = self.getXfmrLosses()
        print('Transformer Power Losses:')
        xf = pd.DataFrame(xfmrLosses)
        xf.columns = ['Losses (kW)']
        xf.index = range(1, len(xf) + 1)
        xf.index.name = 'Transformer'
        print(xf)
        print('')
        xLoss=np.sum(xfmrLosses.flatten())
        lLoss=np.sum(lineLosses.flatten())
        totalLoss = lLoss + xLoss
        print('Total Line Losses: {:.3f} kW'.format(lLoss))
        print('Total Transformer Losses: {:.3f} kW'.format(xLoss))
        print('Total System Losses: {:.3f} kW'.format(totalLoss))

    def getCurrentDirections(self):
        fromToBuses = np.zeros((len(self.lines), 2), dtype=str)
        for k in range(0, len(self.lines)):
            bus1 = self.lines[k].bus1
            bus2 = self.lines[k].bus2
            v1 = self.buses[bus1].voltage
            v2 = self.buses[bus1].voltage
            if v1 > v2:
                fromToBuses[k][0] = bus1
                fromToBuses[k][1] = bus2
            else:
                fromToBuses[k][0] = bus2
                fromToBuses[k][1] = bus1

    def getLineLosses(self):
        losses = np.zeros((len(self.lines), 1))
        for k in range(0, len(self.lines)):
            bus1 = self.buses[self.lines[k].bus1]
            bus2 = self.buses[self.lines[k].bus2]
            Z = self.lines[k].per_unit_Z(self.findZBase(self.lines[k].bus1))
            V1 = bus1.voltage * np.exp(1j * bus1.angle)
            V2 = bus2.voltage * np.exp(1j * bus2.angle)
            I = np.abs((V1 - V2) / Z)
            losses[k] = (I ** 2) * np.real(Z) * self.sBase*1000
        return losses

    def getXfmrLosses(self):
        losses = np.zeros((len(self.transformers), 1))
        for k in range(0, len(self.transformers)):
            bus1 = self.buses[self.transformers[k].bus1]
            bus2 = self.buses[self.transformers[k].bus2]
            Z=self.transformers[k].per_unit_Z(self.findZBase(self.transformers[k].bus1))
            V1 = bus1.voltage * np.exp(1j * bus1.angle)
            V2 = bus2.voltage * np.exp(1j * bus2.angle)
            I = np.abs((V1 - V2) / Z)
            losses[k] = (I**2)*np.real(Z)*self.sBase*1000
        return losses

    def getLineCurrents(self):

        lineCurrents = np.zeros((len(self.lines), 1),complex)
        for l in self.lines:
            Zbase = self.findZBase(l.bus1)
            z_pu = l.per_unit_Z(Zbase)
            b1 = l.bus1
            b2 = l.bus2
            for b in list(self.buses.values()):
                if b.name is b1:
                    v1 = b.voltage*np.exp(1j*b.angle)
                if b.name is b2:
                    v2 = b.voltage*np.exp(1j*b.angle)
            iBase = (self.vBase * self.getVMultiplier(l.bus1) / np.sqrt(3) / Zbase)*1000
            ind = self.lines.index(l)
            lineCurrents[ind] = ((v1 - v2) / z_pu) * iBase

        return lineCurrents

    def getAmpacityPercent(self, lineCurrents):
        amp_percent = np.zeros((len(self.lines), 1))
        for l in self.lines:
            amp = l.ampacity
            ind = self.lines.index(l)
            amp_percent[ind] = abs(lineCurrents[ind] / amp)*100

        return amp_percent

    #print all bus voltages and angles
    def print_Bus_Voltages(self):
        print('Bus Voltages for ' + self.name)
        output: str = ''

        #print header for all buses
        for b in self.bus_order:
            output += b + '\t'
        print(output)
        output = ''

        #print all voltage magnitudes
        for i in range(0, Bus.numBuses):
            output += str(round(np.absolute(self.buses[self.bus_order[i]].voltage), 3)) + '\t'
        print(output)
        output=''

        #print all voltage angles
        for i in range(0, Bus.numBuses):
            output += str(round(self.buses[self.bus_order[i]].angle*180/math.pi, 3)) + '\t'
        print(output)

#get the current Power Injection based on V and angle
    def calculatedPower(self):
        P = np.zeros((Bus.numBuses,1))
        Q = np.zeros((Bus.numBuses,1))

        for k in range(0, Bus.numBuses):
            busK = self.buses[self.bus_order[k]]
            for n in range(0, Bus.numBuses):
                busN = self.buses[self.bus_order[n]]
                P[k] += busK.voltage*busN.voltage*np.abs(self.yBusM[k,n])*np.cos(busK.angle - busN.angle - np.angle(self.yBusM[k,n]))
                Q[k] += busK.voltage*busN.voltage*np.abs(self.yBusM[k,n])*np.sin(busK.angle - busN.angle - np.angle(self.yBusM[k,n]))


        yGuess = np.zeros((len(P)*2,1))
        for k in range(0,len(P)):
            yGuess[k][0] = P[k][0]
            yGuess[k+len(P)][0] = Q[k][0]
        return yGuess


def hasVisited(visited:list(),ind):
    if ind in visited:
        return 1
    else:
        return 0

def printMatrix(M):

    for r in range(0,M.shape[0]):
        output=''
        for c in range(0,M.shape[1]):
            output+=str(round(M[r][c],2)) + '\t'
        print(output)