#import classes
from LineCode import LineCode
from LineGeometry import LineGeometry
from PowerSystem import PowerSystem
from Solution import Solution
from typing import Dict
import math
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

print('Welcome to Power System Solver')
print('')

print('Power System Configured')


endProg=False

#loop and run system until user decides to end program
while True:
    print('--------------------------------------------------')
    print('')
    choice='Z'

    #choose a solver
    while choice!='0' and choice!='1' and choice!='2'and choice!='3':
        print('Solver Options:')
        print('(1) Power Flow')
        print('(2) Fault Study')
        print('(3) System Options')
        print('(0) Quit')
        choice=input('Choose a solver: ')

        if choice!='0' and choice!='1' and choice!='2'and choice!='3':
            print('Invalid Choice. Try again.')

    print('')

    #exit program
    if choice=='0':
        print('Thank you. Goodbye.')
        exit()
    # Power Flow
    elif choice=='1':
        print('Choose a Power Flow Solver:')
        sChoice = 'Z'

        # choose a solver
        while sChoice != '0' and sChoice != '1' and sChoice != '2' and sChoice!='3':
            print('Power Flow Solver Options:')
            print('(1) Newton-Raphson Power Flow')
            print('(2) Fast Decoupled Newton-Raphson Power Flow')
            print('(3) DC Power Flow')
            print('(0) Cancel')
            sChoice = input('Choose a Power Flow solver: ')

            if sChoice != '0' and sChoice != '1' and sChoice != '2' and sChoice!='3':
                print('Invalid Choice. Try again.')

        print('')
        #run the power flow solver
        if sChoice=='1':
            soln1.solveNewtonRaphson()
            soln1.resetBusesAndGenerators()
        elif sChoice=='2':
            soln1.solveFastDecoupled()
            soln1.resetBusesAndGenerators()
        elif sChoice=='3':
            soln1.solveDCPowerFlow()

    #Fault Study
    elif choice=='2':
        system1.get_ZBus012()

        print('Choose a Fault Study Option:')
        sChoice = 'Z'

        # choose a solver
        while sChoice != '0' and sChoice != '1' and sChoice != '2' and sChoice != '3' and sChoice!='4':
            print('Fault Options:')
            print('(1) Three Phase Symmetrical Fault')
            print('(2) Line to Line Fault')
            print('(3) Single Line to Ground Fault')
            print('(4) Double Line to Ground Fault')
            print('(0) Cancel')
            sChoice = input('Choose a Fault: ')

            if sChoice != '0' and sChoice != '1' and sChoice != '2' and sChoice != '3' and sChoice!='4':
                print('Invalid Choice. Try again.')

        print('')
        #get the faulted bus
        busName=input('Enter name of faulted bus: ')
        busFound=True
        index1=-1
        if not busName in system1.buses:
            print('Bus ' + busName + ' not found.')
            print('Returning to main menu')
            busFound=False
        else:
            # get index
            index1 = system1.bus_order.index(busName)


        # run the power flow solver
        if sChoice == '1' and busFound:
            Vf = float(input('Enter Prefault Voltage (per unit): '))
            soln1.solveSymmetricalFault(index1,Vf)
        elif sChoice == '2' and busFound:
            Vf = float(input('Enter Prefault Voltage (per unit): '))
            Rf = float(input('Enter Fault Resistance (ohms): '))
            Xf = float(input('Enter Fault Reactance (ohms): '))
            Zf=Rf+1j*Xf
            soln1.solveLineToLineFault(index1,Vf,Zf)
        elif sChoice == '3' and busFound:
            Vf = float(input('Enter Prefault Voltage (per unit): '))
            Rf = float(input('Enter Fault Resistance (ohms): '))
            Xf = float(input('Enter Fault Reactance (ohms): '))
            Zf = Rf + 1j * Xf
            soln1.solveLineToGroundFault(index1,Vf,Zf)
        elif sChoice=='4' and busFound:
            Vf = float(input('Enter Prefault Voltage (per unit): '))
            Rf = float(input('Enter Fault Resistance (ohms): '))
            Xf = float(input('Enter Fault Reactance (ohms): '))
            Zf = Rf + 1j * Xf
            soln1.solveDoubleLineToGroundFault(index1,Vf,Zf)
    #System Options
    else:
        print('Choose a System Option:')
        sChoice = 'Z'

        # choose a solver
        while sChoice != '0' and sChoice != '1' and sChoice != '2' and sChoice != '3' and sChoice != '4'and sChoice != '5' and sChoice != '6' and sChoice != '7' and sChoice != '8':
            print('Power System Options:')
            print('(1) Reset to Flat Start')
            print('(2) Change Bus Voltage (per unit)')
            print('(3) Change Slack Bus')
            print('(4) Change Load MW')
            print('(5) Change Load Mvar')
            print('(6) Change Generator Power Output')
            print('(7) Change Generator Mvar Limits')
            print('(8) Change System S Base')
            print('(0) Cancel')
            sChoice = input('Choose a system option: ')

            if sChoice != '0' and sChoice != '1' and sChoice != '2' and sChoice != '3'and sChoice != '4'and sChoice != '5' and sChoice != '6' and sChoice != '7' and sChoice != '8':
                print('Invalid Choice. Try again.')

        print('')
        # flat start
        if sChoice == '1':
            system1.setFlatStart()
        #bus voltage
        elif sChoice == '2':
            busName = input('Enter name of bus: ')
            if not busName in system1.buses:
                print('Bus ' + busName + ' not found.')
                print('Returning to main menu')
            else:
                busEdit=system1.buses[busName]
                print(f'Bus {busName} is operating at a voltage of {busEdit.voltage} pu and an angle of {busEdit.angle*180/math.pi} degrees')
                v=float(input('Enter the new bus voltage (per unit): '))
                if v>=0:
                    a=math.pi/180*float(input('Enter new bus angle (degrees): '))
                    busEdit.voltage=v
                    busEdit.angle=a
                    print('Success!')
                else:
                    print('Invalid voltage entry. No changes made')
        #slack bus
        elif sChoice == '3':
            print(f'Currently the slack bus is bus {system1.slackBus}')
            busName = input('Enter name of new slack bus: ')
            if not busName in system1.buses:
                print('Bus ' + busName + ' not found.')
                print('Returning to main menu')
            else:
                v=float(input('Enter new slack bus voltage (kV): '))
                if v>0:
                    system1.set_Slack(busName,v)
                    print('Success!')
                else:
                    print('Invalid voltage entry. No changes made')
        #load mw
        elif sChoice=='4':
            loadName = input('Enter name of load: ')
            loadExists=False
            load1=None
            for k in range(0,len(system1.loads)):
                if system1.loads[k].name==loadName:
                    load1=system1.loads[k]
                    loadExists=True
            if not loadExists:
                print('Load ' + loadName + ' not found.')
                print('Returning to main menu')
            else:
                print(f'Load {loadName} has a currently has a real power of {load1.P} MW')
                mw = float(input('Enter new load real power (MW): '))
                if mw >= 0:
                    load1.P=mw
                    print('Success!')
                else:
                    print('Invalid power entry. No changes made')
        #load mvar
        elif sChoice=='5':
            loadName = input('Enter name of load: ')
            loadExists = False
            load1 = None
            for k in range(0, len(system1.loads)):
                if system1.loads[k].name == loadName:
                    load1 = system1.loads[k]
                    loadExists = True
            if not loadExists:
                print('Load ' + loadName + ' not found.')
                print('Returning to main menu')
            else:
                print(f'Load {loadName} has a currently has a reactive power of {load1.Q} Mvar')
                mvar = float(input('Enter new load reactive power (Mvar): '))
                load1.Q=mvar
                print('Success!')
        #generator power output
        elif sChoice=='6':
            genName = input('Enter name of generator: ')
            genExists = False
            gen1 = None
            for k in range(0, len(system1.generators)):
                if system1.generators[k].name == genName:
                    gen1 = system1.generators[k]
                    genExists = True
            if not genExists:
                print('Generator ' + genName + ' not found.')
                print('Returning to main menu')
            else:
                print(f'Generator {genName} has real power set to {gen1.P} MW')
                mw = float(input('Enter new generator real power (MW): '))
                if mw >= 0:
                    gen1.P = mw
                    print('Success!')
                else:
                    print('Invalid power entry. No changes made')
        #generator mvar limits
        elif sChoice=='7':
            genName = input('Enter name of generator: ')
            genExists = False
            gen1 = None
            for k in range(0, len(system1.generators)):
                if system1.generators[k].name == genName:
                    gen1 = system1.generators[k]
                    genExists = True
            if not genExists:
                print('Generator ' + genName + ' not found.')
                print('Returning to main menu')
            else:
                print(f'Generator {genName} has reactive power limited to values between {gen1.Q_min} and {gen1.Q_max} Mvar')
                mvarMax = float(input('Enter new generator max reactive power limit (Mvar): '))
                mvarMin = float(input('Enter new generator min reactive power limit (Mvar): '))
                if mvarMax<mvarMin:
                    print('Invalid entry. Upper limit must be greater than lower limit')
                    print('No changes made')
                else:
                    gen1.Q_min=mvarMin
                    gen1.Q_max=mvarMax
                    print('Success!')
        #changing S base
        elif sChoice=='8':
            print(f'The system S base is currently {system1.sBase} MVA')
            newS=float(input('Enter new system S base (MVA): '))
            if newS<=0:
                print('Invalid Entry. No changes made.')
            else:
                system1.sBase=newS
                print('Success!')





