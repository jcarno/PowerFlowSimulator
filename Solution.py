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

        while maxError>self.tolerance and a<self.maxIterations:

            #get the known power values at the buses
            yGoal=self.getGivenPower()

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
                print(f"Converged in {a} iterations!")
            else:
                #find Jacobian for all buses
                jacobian=self.getJacobian()

                # set up matrix to show what indeces are useful after eliminating rows and columns
                usedBuses = np.zeros((Bus.numBuses * 2, 2))
                for k in range(0, Bus.numBuses * 2 - 1):
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

    # get the power at each bus defined by loads and generators
    def getGivenPower(self):
        n = Bus.numBuses
        y = np.zeros((n * 2, 1))

        # cycle through all generators
        for g in self.sys.generators:
            t=self.sys.buses[g.bus].type
            if t!='S':
                ind=self.sys.bus_order.index(g.bus)
                y[ind, 0] = y[ind]+ g.P/self.sys.sBase

                #only add Q setpoint if the generator is considered PQ
                if t=='PQ':
                    y[ind+n,0]=y[ind+n,0]+ g.Q/self.sys.sBase

        #cycle through all loads
        for l in self.sys.loads:
            ind = self.sys.bus_order.index(l.bus)
            y[ind, 0] = y[ind, 0] - l.P/self.sys.sBase
            y[ind + n, 0] = y[ind + n, 0] - l.Q/self.sys.sBase

        #return result
        return y

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