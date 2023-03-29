from PowerSystem import PowerSystem
from Bus import Bus
from Generator import Generator
from Load import Load
from Line import Line
from Transformer import Transformer
import math
import numpy as np

class Solution:

    def __init__(self, sys: PowerSystem):
        self.sys=sys
        self.tolerance=0.0001
        self.maxIterations=50

    def setTolerance(self,tolerance):
        self.tolerance=tolerance
    def setMaxIterations(self,maxIter):
        self.maxIterations=maxIter

    #solve system using newton Raphson
    def solveNewtonRaphson(self):
        maxError=self.tolerance+1
        a=0

        # make the YBus Matrix for the system
        self.sys.make_YBus()

        # get the known power values at the buses
        yGoal = self.sys.getGivenPower()

        while maxError>self.tolerance and a<self.maxIterations:

            #get the calculated power based on guess and determine change in y
            yGuess=self.sys.calculatedPower()

            # get the known power values at the buses
            deltaY=yGoal-yGuess

            #reduce deltaY
            #cycle through each bus, check for slack and PV buses
            for k in range(Bus.numBuses*2-1,-1,-1):
                busK=None
                #remove all instances of slack bus
                if k<Bus.numBuses:
                    busK=self.sys.buses[self.sys.bus_order[k]]
                    if busK.type=='S':
                        deltaY = np.delete(deltaY, k, 0)
                # remove all instances of Q for PV and V for PV
                else:
                    busK=self.sys.buses[self.sys.bus_order[k-Bus.numBuses]]
                    if busK.type=='S' or busK.type=='PV':
                        deltaY = np.delete(deltaY,k,0)

            #get max error
            maxError=self.getMaxError(deltaY)
            if maxError < self.tolerance:
                overLims = self.checkMvarLims(yGuess)
                if overLims:
                    self.sys.setFlatStart()
                    self.solveNewtonRaphson()
                else:
                    print(f"Newton Raphson Converged in {a} iterations!")
            else:
                #find Jacobian for all buses
                jacobian=self.getJacobian()

                # set up matrix to show what indeces are useful after eliminating rows and columns
                usedBuses = np.zeros((Bus.numBuses * 2, 2))
                for k in range(0, Bus.numBuses * 2):
                    if k < Bus.numBuses:
                        usedBuses[k][0] = k % Bus.numBuses
                        usedBuses[k][1] = 0
                    else:
                        usedBuses[k][0] = k % Bus.numBuses
                        usedBuses[k][1] = 1

                # reduce Jacobian
                # cycle through each bus, check for slack and PV buses
                for k in range(Bus.numBuses * 2 - 1, -1, -1):
                    busK = None
                    # remove all instances of slack bus
                    if k < Bus.numBuses:
                        busK = self.sys.buses[self.sys.bus_order[k]]
                        if busK.type == 'S':
                            jacobian = np.delete(jacobian, k, 0)
                            jacobian = np.delete(jacobian, k, 1)
                            usedBuses = np.delete(usedBuses, k, 0)
                    # remove all instances of Q for PV and V for PV
                    else:
                        busK = self.sys.buses[self.sys.bus_order[k - Bus.numBuses]]
                        if busK.type == 'S' or busK.type == 'PV':
                            jacobian = np.delete(jacobian, k, 0)
                            jacobian = np.delete(jacobian, k, 1)
                            usedBuses = np.delete(usedBuses, k, 0)


                deltaX=np.matmul(np.linalg.inv(jacobian),deltaY)
                #update the bus voltages and angles
                self.updateBuses(deltaX,usedBuses)
                a=a+1

        if (maxError>self.tolerance):
            print(f"Newton Raphson failed to converge in {self.maxIterations} iterations")
        elif (not overLims):
            self.sys.print_Results()

    def solveDCPowerFlow(self):
        self.sys.make_YBus()
        self.sys.setFlatStart()

        P = self.sys.getGivenPower()[0:Bus.numBuses]
        B_bus = np.imag(self.sys.yBusM)

        #reduce P and B
        for k in range(0,Bus.numBuses):
            if (self.sys.bus_order[k]==self.sys.slackBus):
                B_bus=np.delete(B_bus,k,0)
                B_bus = np.delete(B_bus, k, 1)
                P=np.delete(P,k,0)

        angle = np.matmul(-np.linalg.pinv(B_bus), P)

        allBuses=list(self.sys.buses.values())
        ind=self.sys.bus_order.index(self.sys.slackBus)
        a=0
        for k in range(0,Bus.numBuses):
            if k!=self.sys.bus_order.index(self.sys.slackBus):
                allBuses[k].angle = angle[a]
                a=a+1
        self.sys.printDCPwr()

    #do Fast Decoupled Newton Raphson
    def solveFastDecoupled(self):
        maxError=self.tolerance+1
        a=0

        # make the YBus Matrix for the system
        self.sys.make_YBus()

        # get the known power values at the buses
        yGoal = self.sys.getGivenPower()

        while maxError>self.tolerance and a<self.maxIterations:

            #get the calculated power based on guess and determine change in y
            yGuess=self.sys.calculatedPower()
            deltaY=yGoal-yGuess

            #reduce deltaY
            #cycle through each bus, check for slack and PV buses
            for k in range(Bus.numBuses*2-1,-1,-1):
                busK=None
                #remove all instances of slack bus
                if k<Bus.numBuses:
                    busK=self.sys.buses[self.sys.bus_order[k]]
                    if busK.type=='S':
                        deltaY = np.delete(deltaY, k, 0)

                # remove all instances of Q for PV and V for PV
                else:
                    busK=self.sys.buses[self.sys.bus_order[k-Bus.numBuses]]
                    if busK.type=='S' or busK.type=='PV':
                        deltaY = np.delete(deltaY,k,0)

            #get max error
            maxError=self.getMaxError(deltaY)
            if maxError<self.tolerance:
                overLims=self.checkMvarLims(yGuess)
                if overLims:
                    self.sys.setFlatStart()
                    self.solveFastDecoupled()
                else:
                    print(f"Fast Decoupled Newton Raphson Converged in {a} iterations!")
            else:
                #find Jacobian for all buses
                J1=self.getJ1()
                J4=self.getJ4()

                # set up matrix to show what indeces are useful after eliminating rows and columns
                usedBuses = np.zeros((Bus.numBuses * 2, 2))
                for k in range(0, Bus.numBuses * 2):
                    if k < Bus.numBuses:
                        usedBuses[k][0] = k % Bus.numBuses
                        usedBuses[k][1] = 0
                    else:
                        usedBuses[k][0] = k % Bus.numBuses
                        usedBuses[k][1] = 1

                # reduce Jacobian
                # cycle through each bus, check for slack and PV buses
                pDim=Bus.numBuses
                qDim=Bus.numBuses

                # reduce Jacobian
                # cycle through each bus, check for slack and PV buses
                for k in range(Bus.numBuses * 2 - 1, -1, -1):
                    busK = None
                    # remove all instances of slack bus
                    if k < Bus.numBuses:
                        busK = self.sys.buses[self.sys.bus_order[k]]
                        if busK.type == 'S':
                            J1 = np.delete(J1, k, 0)
                            J1 = np.delete(J1, k, 1)
                            usedBuses = np.delete(usedBuses, k, 0)
                    # remove all instances of Q for PV and V for PV
                    else:
                        nk=k-Bus.numBuses
                        busK = self.sys.buses[self.sys.bus_order[nk]]

                        if busK.type == 'S' or busK.type == 'PV':
                            J4 = np.delete(J4, nk, 0)
                            J4 = np.delete(J4, nk, 1)
                            usedBuses = np.delete(usedBuses, k, 0)

                pDim=np.size(J1,axis=0)
                qDim=np.size(J4,axis=0)
                #invert jacobians and find deltaX
                deltaP=deltaY[0:pDim]
                deltaQ=deltaY[pDim:pDim+qDim]
                deltaA=np.matmul(np.linalg.inv(J1),deltaP)
                deltaV = np.matmul(np.linalg.inv(J4), deltaQ)
                deltaX=np.vstack((deltaA,deltaV))

                #update the bus voltages and angles
                self.updateBuses(deltaX,usedBuses)
                a=a+1

        if (maxError>self.tolerance):
            print(f"Fast Decoupled Newton Raphson failed to converge in {a} iterations")
        elif (not overLims):
            self.sys.print_Results()

    #check if the mvar limits of a PV bus generator were passed, return true if passed
    def checkMvarLims(self,yGuess):
        yGuessActual=yGuess*self.sys.sBase
        for k in range(0,len(self.sys.generators)):
            gen=self.sys.generators[k]
            genBus=self.sys.buses[gen.bus]
            ind=self.sys.bus_order.index(gen.bus)
            if (yGuessActual[ind+Bus.numBuses][0]>gen.Q_max and genBus.type=='PV'):
                print(f'Positive Mvar Limit for Bus {genBus.name} exceeded. Switching to PQ bus and restarting')
                gen.Q=gen.Q_max
                genBus.type='PQ'
                return True
            elif (yGuessActual[ind+Bus.numBuses][0]<gen.Q_min and genBus.type=='PV'):
                print(f'Negative Mvar Limit for Bus {genBus.name} exceeded. Switching to PQ bus and restarting')
                gen.Q=gen.Q_min
                genBus.type='PQ'
                return True
        return False

    #function to get the jacobian for the entire system
    def getJacobian(self):
        #set up jacobian given size of the system
        n=Bus.numBuses
        jacobian=np.zeros((n*2,n*2))
        busK=None
        busN=None
        f=None

        #set jacobian
        for r in range(0,2*n):
            for c in range(0,2*n):
                #test for each possiblity for PQV and angle in Jacobian

                #nondiagonial P by angle
                if r<n and c<n and r!=c:
                    busK=self.sys.buses[self.sys.bus_order[r]]
                    busN=self.sys.buses[self.sys.bus_order[c]]
                    f=busK.voltage*busN.voltage*np.absolute(self.sys.yBusM[r,c])
                    jacobian[r,c]=f*math.sin(busK.angle-busN.angle-np.angle(self.sys.yBusM[r,c]))

                #nondiagonal P by voltage
                elif r<n and c>=n and r!=c-n:
                    busK = self.sys.buses[self.sys.bus_order[r]]
                    busN = self.sys.buses[self.sys.bus_order[c-n]]
                    f = busK.voltage * np.absolute(self.sys.yBusM[r, c-n])
                    jacobian[r, c] = f * math.cos(busK.angle - busN.angle - np.angle(self.sys.yBusM[r, c-n]))


                #nondiagonal Q by angle
                elif r >= n and c < n and r-n != c:
                    busK = self.sys.buses[self.sys.bus_order[r - n]]
                    busN = self.sys.buses[self.sys.bus_order[c]]
                    f = busK.voltage * busN.voltage * np.absolute(self.sys.yBusM[r - n, c])
                    jacobian[r, c] = -f * math.cos(busK.angle - busN.angle - np.angle(self.sys.yBusM[r- n, c]))

                #nondiagonal Q by Voltage
                elif r >= n and c >= n and r != c :
                    busK = self.sys.buses[self.sys.bus_order[r-n]]
                    busN = self.sys.buses[self.sys.bus_order[c - n]]
                    f = busK.voltage * np.absolute(self.sys.yBusM[r-n, c - n])
                    jacobian[r, c] = f * math.sin(busK.angle - busN.angle - np.angle(self.sys.yBusM[r-n, c - n]))

                #diagonal P by angle
                elif r<n and c<n and r==c:
                    busK=self.sys.buses[self.sys.bus_order[r]]
                    f=0

                    for p in range(0,n):
                        if r!=p:
                            busN = self.sys.buses[self.sys.bus_order[p]]
                            f+=np.absolute(self.sys.yBusM[r,p])*busN.voltage*math.sin(busK.angle-busN.angle-np.angle(self.sys.yBusM[r,p]))

                    jacobian[r,c]=-f*busK.voltage

                #diagonal P by voltage
                elif r<n and c>=n and r==c-n:
                    busK = self.sys.buses[self.sys.bus_order[r]]
                    f=0
                    for p in range(0,n):
                        busN = self.sys.buses[self.sys.bus_order[p]]
                        f+=busN.voltage*np.absolute(self.sys.yBusM[r, p])* math.cos(busK.angle - busN.angle - np.angle(self.sys.yBusM[r, p]))

                    jacobian[r, c] = f+ busK.voltage*np.absolute(self.sys.yBusM[r, c-n])*math.cos(np.angle(self.sys.yBusM[r, c-n]))


                #diagonal Q by angle
                elif r >= n and c < n and r-n == c:
                    busK = self.sys.buses[self.sys.bus_order[r-n]]
                    f = 0

                    for p in range(0, n):
                        if c != p:
                            busN = self.sys.buses[self.sys.bus_order[p]]
                            f += np.absolute(self.sys.yBusM[r-n, p]) * busN.voltage * math.cos(busK.angle - busN.angle - np.angle(self.sys.yBusM[r-n, p]))

                    jacobian[r, c] = f * busK.voltage

                #diagonal Q by Voltage
                elif r >= n and c >= n and r == c :
                    busK = self.sys.buses[self.sys.bus_order[r-n]]
                    f=0

                    for p in range(0,n):
                        busN=self.sys.buses[self.sys.bus_order[p]]
                        f+=np.absolute(self.sys.yBusM[r-n, p]) * busN.voltage * math.sin(busK.angle - busN.angle - np.angle(self.sys.yBusM[r-n, p]))

                    jacobian[r, c] = f + -busK.voltage*np.absolute(self.sys.yBusM[r-n, c - n])*math.sin(np.angle(self.sys.yBusM[r-n, c - n]))
        return jacobian

    #function to get the first quadrant of the jacobian
    def getJ1(self):
        #set up jacobian given size of the system
        n=Bus.numBuses
        jacobian=np.zeros((n,n))
        busK=None
        busN=None
        f=None

        #set jacobian
        for r in range(0,n):
            for c in range(0,n):
                #test for each possiblity for PQV and angle in Jacobian

                #nondiagonial P by angle
                if r!=c:
                    busK=self.sys.buses[self.sys.bus_order[r]]
                    busN=self.sys.buses[self.sys.bus_order[c]]
                    f=busK.voltage*busN.voltage*np.absolute(self.sys.yBusM[r,c])
                    jacobian[r,c]=f*math.sin(busK.angle-busN.angle-np.angle(self.sys.yBusM[r,c]))

                #diagonal P by angle
                else:
                    busK=self.sys.buses[self.sys.bus_order[r]]
                    f=0

                    for p in range(0,n):
                        if r!=p:
                            busN = self.sys.buses[self.sys.bus_order[p]]
                            f+=np.absolute(self.sys.yBusM[r,p])*busN.voltage*math.sin(busK.angle-busN.angle-np.angle(self.sys.yBusM[r,p]))

                    jacobian[r,c]=-f*busK.voltage


        return jacobian

    #function to get the fourth quadrant of the jacobian
    def getJ4(self):
        #set up jacobian given size of the system
        n=Bus.numBuses
        jacobian=np.zeros((n,n))
        busK=None
        busN=None
        f=None

        #set jacobian
        for r in range(0,n):
            for c in range(0,n):
                #get each value of Q by voltage in jacobian 4th quadrant

                #nondiagonal Q by Voltage
                if r != c :
                    busK = self.sys.buses[self.sys.bus_order[r]]
                    busN = self.sys.buses[self.sys.bus_order[c]]
                    f = busK.voltage * np.absolute(self.sys.yBusM[r, c])
                    jacobian[r, c] = f * math.sin(busK.angle - busN.angle - np.angle(self.sys.yBusM[r, c]))

                #diagonal Q by Voltage
                else:
                    busK = self.sys.buses[self.sys.bus_order[r]]
                    f=0

                    for p in range(0,n):
                        busN=self.sys.buses[self.sys.bus_order[p]]
                        f+=np.absolute(self.sys.yBusM[r, p]) * busN.voltage * math.sin(busK.angle - busN.angle - np.angle(self.sys.yBusM[r, p]))

                    jacobian[r, c] = f + -busK.voltage*np.absolute(self.sys.yBusM[r, c])*math.sin(np.angle(self.sys.yBusM[r, c]))
        return jacobian


    #update all voltages and angles
    def updateBuses(self, deltaX, usedBuses):
        for k in range(0, len(deltaX)):
            ind = int(usedBuses[k][0])
            busR = self.sys.buses[self.sys.bus_order[ind]]
            isVolt = int(usedBuses[k][1])
            if isVolt == 0:
                busR.angle = adjustAngleRange(busR.angle+deltaX[k][0])
            else:
                busR.voltage = busR.voltage+deltaX[k][0]


    def getMaxError(self,deltaY):
        return np.max(np.absolute(deltaY))



def adjustAngleRange(angle):
    if (angle<math.pi and angle>-math.pi):
        return angle
    elif (angle>math.pi):
        return adjustAngleRange(angle-2*math.pi)
    else:
        return adjustAngleRange(angle+2*math.pi)

def printMatrix(M):

    for r in range(0,M.shape[0]):
        output=''
        for c in range(0,M.shape[1]):
            output+=str(round(M[r][c],2)) + '\t'
        print(output)