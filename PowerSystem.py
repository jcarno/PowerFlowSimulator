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
        self.lines: Dict[str, Line] = dict()
        self.transformers: Dict[str, Transformer] = dict()
        self.generators:Dict[str,Generator]=dict()
        self.loads:Dict[str,Load]=dict()

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

    #add a generator to the system
    def add_Generator(self,name,bus,voltage,P_setpoint,Q_setpoint=0,Q_min=-10000,Q_max=10000):
        # add generator and buses to list
        self.generators[name] = Generator(name, bus, voltage, P_setpoint,Q_setpoint,Q_min,Q_max)
        self.add_bus(bus,'PV',voltage)

    #add load to the system
    def add_Load(self,name,bus, P,Q):
        # add generator and buses to list
        self.loads[name] = Load(name,bus, P,Q)
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
        gens=list(self.generators.values())
        for g in gens:
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

    def make_YBus(self):
        # reset yBus Matrix
        self.yBusM = np.zeros((100, 100), dtype=complex)

        # take dictionary of lines and turn it into a list
        line_list = list(self.lines.values())

        # for each line, get the per unit impedance and admittance and add to the matrix
        for lin in line_list:
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
        xfmr_list = list(self.transformers.values())
        for xfmr in xfmr_list:
            zbase = self.findZBase(xfmr.bus1)
            z_pu = xfmr.per_unit_Z(zbase)
            index1 = self.bus_order.index(xfmr.bus1)
            index2 = self.bus_order.index(xfmr.bus2)
            # might be some redundancies in adding to matrix
            self.yBusM[index1, index1] += 1 / z_pu
            self.yBusM[index2, index2] += 1 / z_pu
            self.yBusM[index1, index2] -= 1 / z_pu
            self.yBusM[index2, index1] -= 1 / z_pu

    #solve yBus matrix using algorithm from class
    def solve(self):
        self.make_YBus()

        #get the known power values at the buses
        yGoal=self.getGivenPower()

        #get the calculated power based on guess and determine change in y
        yGuess=self.currentPowerGuess()
        deltaY=yGoal-yGuess

        #find Jacobian for all buses
        jacobian=self.getJacobian()

        #set up matrix to show what indeces are useful after eliminating rows and columns
        usedBuses=np.zeros((Bus.numBuses*2,2))
        for k in range(0,Bus.numBuses*2-1):
            if k<Bus.numBuses:
                usedBuses[k][0]=k%Bus.numBuses
                usedBuses[k][1]=0
            else:
                usedBuses[k][0]=k%Bus.numBuses
                usedBuses[k][1]=1

        #reduce Jacobian
        #cycle through each bus, check for slack and PV buses
        for k in range(Bus.numBuses*2-1,0,-1):
            busK=None
            #remove all instances of slack bus
            if k<Bus.numBuses:
                busK=self.buses[self.bus_order[k]]
                if busK.type=='S':
                    jacobian = np.delete(jacobian, k, 0)
                    jacobian = np.delete(jacobian, k, 1)
                    deltaY = np.delete(deltaY, k, 0)
                    usedBuses = np.delete(usedBuses, k, 0)
            # remove all instances of Q for PV and V for PV
            else:
                busK=self.buses[self.bus_order[k-Bus.numBuses]]
                if busK.type=='S' or busK.type=='PV':
                    jacobian=np.delete(jacobian,k,0)
                    jacobian = np.delete(jacobian, k, 1)
                    deltaY = np.delete(deltaY,k,0)
                    usedBuses=np.delete(usedBuses,k,0)

        #determine delta X
        deltaX=np.linalg.inv(jacobian)*deltaY

        #update the bus voltages and angles
        self.updateBuses(deltaX,usedBuses)

    #get the current Power Injection based on V and angle
    def currentPowerGuess(self):
        P = np.zeros((Bus.numBuses,1))
        Q = np.zeros((Bus.numBuses,1))

        for k in range(0, Bus.numBuses):
            busR = self.buses[self.bus_order[k]]
            for n in range(0, Bus.numBuses):
                busP = self.buses[self.bus_order[n]]
                P[k] += busR.voltage*busP.voltage*np.real(self.yBusM[k,n])*np.cos(busR.angle - busP.angle - np.angle(self.yBusM[k,n]))
                Q[k] += busR.voltage*busP.voltage*np.real(self.yBusM[k,n])*np.sin(busR.angle - busP.angle - np.angle(self.yBusM[k,n]))


        yGuess = np.zeros((len(P)*2,1))
        for k in range(0,len(P)-1):
            yGuess[k][0] = P[k][0]
            yGuess[k+len(P)][0] = Q[k][0]
        return yGuess


    #function to get the jacobian for the entire system
    def getJacobian(self):
        #set up jacobian given size of the system
        n=Bus.numBuses
        jacobian=np.zeros((n*2,n*2))
        busK=None
        busN=None
        f=None

        #set jacobian
        for r in range(0,2*n-1):
            for c in range(0,2*n-1):
                #test for each possiblity for PQV and angle in Jacobian

                #nondiagonial P by angle
                if r<n and c<n and r!=c:
                    busK=self.buses[self.bus_order[r]]
                    busN=self.buses[self.bus_order[c]]
                    f=busK.voltage*busN.voltage*np.absolute(self.yBusM[r,c])
                    jacobian[r,c]=f*math.sin(busK.angle-busN.angle-np.angle(self.yBusM[r,c]))

                #nondiagonal P by voltage
                elif r<n and c>=n and r!=c-n:
                    busK = self.buses[self.bus_order[r]]
                    busN = self.buses[self.bus_order[c-n]]
                    f = busK.voltage * np.absolute(self.yBusM[r, c-n])
                    jacobian[r, c] = f * math.cos(busK.angle - busN.angle - np.angle(self.yBusM[r, c-n]))


                #nondiagonal Q by angle
                elif r >= n and c < n and r-n != c:
                    busK = self.buses[self.bus_order[r - n]]
                    busN = self.buses[self.bus_order[c]]
                    f = busK.voltage * busN.voltage * np.absolute(self.yBusM[r - n, c])
                    jacobian[r, c] = -f * math.cos(busK.angle - busN.angle - np.angle(self.yBusM[r- n, c]))

                #nondiagonal Q by Voltage
                elif r >= n and c >= n and r != c :
                    busK = self.buses[self.bus_order[r-n]]
                    busN = self.buses[self.bus_order[c - n]]
                    f = busK.voltage * np.absolute(self.yBusM[r-n, c - n])
                    jacobian[r, c] = f * math.sin(busK.angle - busN.angle - np.angle(self.yBusM[r-n, c - n]))

                #diagonal P by angle
                elif r<n and c<n and r==c:
                    busK=self.buses[self.bus_order[r]]
                    f=0

                    for p in range(0,n-1):
                        if c!=p:
                            busN = self.buses[self.bus_order[p]]
                            f+=self.yBusM[r,p]*busN.voltage*math.sin(busK.angle-busN.angle-np.angle(self.yBusM[r,p]))

                    jacobian[r,c]=-f*busK.voltage

                #diagonal P by voltage
                elif r<n and c>=n and r==c-n:
                    busK = self.buses[self.bus_order[r]]
                    f=0
                    for p in range(0,n-1):
                        busN = self.buses[self.bus_order[p]]
                        f+=busN.voltage*np.absolute(self.yBusM[r, p])* math.cos(busK.angle - busN.angle - np.angle(self.yBusM[r, p]))

                    jacobian[r, c] = f+ busK.voltage*np.absolute(self.yBusM[r, c-n])*math.cos(np.angle(self.yBusM[r, c-n]))


                #diagonal Q by angle
                elif r >= n and c < n and r-n == c:
                    busK = self.buses[self.bus_order[r-n]]
                    f = 0

                    for p in range(0, n - 1):
                        if c != p:
                            busN = self.buses[self.bus_order[p]]
                            f += self.yBusM[r-n, p] * busN.voltage * math.cos(busK.angle - busN.angle - np.angle(self.yBusM[r-n, p]))

                    jacobian[r, c] = f * busK.voltage

                #diagonal Q by Voltage
                elif r >= n and c >= n and r == c :
                    busK = self.buses[self.bus_order[r-n]]
                    f=0

                    for p in range(0,n-1):
                        busN=self.buses[self.bus_order[p]]
                        f+=self.yBusM[r-n, p] * busN.voltage * math.sin(busK.angle - busN.angle - np.angle(self.yBusM[r-n, p]))

                    jacobian[r, c] = f + -busK.voltage*np.absolute(self.yBusM[r-n, c - n])*math.sin(np.angle(self.yBusM[r-n, c - n]))
        return jacobian

    # get the power at each bus defined by loads and generators
    def getGivenPower(self):
        n = Bus.numBuses
        y = np.zeros((n * 2, 1))

        # get lists of loads and generators
        allGens = list(self.generators.values())
        allLoads = list(self.loads.values())

        # cycle through all generators
        for g in allGens:
            t=self.buses[g.bus].type
            if t!='S':
                ind=self.bus_order.index(g.bus)
                y[ind, 0] = y[ind]+ g.P

                #only add Q setpoint if the generator is considered PQ
                if t=='PQ':
                    y[ind+n,0]=y[ind+n,0]+ g.Q

        #cycle through all loads
        for l in allLoads:
            ind = self.bus_order.index(l.bus)
            y[ind, 0] = y[ind, 0] + l.P
            y[ind + n, 0] = y[ind + n, 0] + l.Q

        #return result
        return y

    #update all voltages and angles
    def updateBuses(self, deltaX, usedBuses):
        for k in range(0, len(deltaX) - 1):
            ind = int(usedBuses[k][0])
            busR = self.buses[self.bus_order[ind]]
            isVolt = int(usedBuses[k][1])
            if isVolt == 0:
                busR.angle = deltaX[k][0]
            else:
                busR.voltage = deltaX[k][0]

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
        for j in range(0, Bus.numBuses):
            output += str(round(np.angle(self.buses[self.bus_order[i]].voltage,True), 3)) + '\t'
        print(output)



def hasVisited(visited:list(),ind):
    if ind in visited:
        return 1
    else:
        return 0
