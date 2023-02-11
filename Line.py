from LineCode import LineCode
from LineGeometry import LineGeometry
import math
class Line:

    def __init__(self,name,len:float,bus1:str,bus2:str,code:LineCode,geo:LineGeometry):
        self.name=name
        self.len=len
        self.bus1=bus1
        self.bus2=bus2
        self.code=code
        self.geo=geo
        self.ampacity=code.ampacity*geo.nConductors
        self.Z=findZ(len,code,geo)
        self.shuntY=findShuntY(len,code,geo)

    def per_unit_Z(self,Zbase):
        return self.Z/Zbase

#Function to find the lines series impedance values
def findZ(len:float,code:LineCode,geo:LineGeometry):
    #calculate R based on length and number of conductors
    R=code.R1perMi*len/geo.nConductors

    #Find Deq and Dsl for  inductor
    Deq=(geo.Dab**2 + geo.Dbc**2 + geo.Dca**2)**(1.0/2.0)

    #break up calculations of Dsl based on number of conductors in bundle
    if (geo.nConductors==4):
        Dsl=1.091*(geo.bSpacingft**3*code.GMRft)**(1.0/4.0)**(1.0/4.0)
    elif (geo.nConductors==3):
        Dsl = (geo.bSpacingft ** 2 * code.GMRft) ** (1.0 / 4.0) ** (1.0 / 3.0)
    elif (geo.nConductors==2):
        Dsl = (geo.bSpacingft * code.GMRft) ** (1.0 / 4.0) ** (1.0 / 2.0)
    elif (geo.nConductors==1):
        Dsl = code.GMRft
    else:
        Dsl=1
        print('Number of conductors in bundle given outside of known range for calculation.')
        print('Check line parameters and try again')

    X=(2*math.pi*60)*2E-7*math.log(Deq/Dsl)*1609*len

    return R+1j*X




def findShuntY(len:float,code:LineCode,geo:LineGeometry):
    # Find Deq and Dsc for shunt capacitance
    Deq = (geo.Dab ** 2 + geo.Dbc ** 2 + geo.Dca ** 2) ** (1.0 / 2.0)

    # break up calculations of Dsl and Dsc based on number of conductors in bundle
    if (geo.nConductors == 4):
        Dsc = 1.091 * (geo.bSpacingft ** 3 * code.dInches / 2.0 / 12.0) ** (1.0 / 4.0)
    elif (geo.nConductors == 3):
        Dsc = (geo.bSpacingft ** 2 * code.dInches / 2.0 / 12.0) ** (1.0 / 3.0)
    elif (geo.nConductors == 2):
        Dsc = (geo.bSpacingft * code.dInches / 2.0 / 12.0) ** (1.0 / 2.0)
    elif (geo.nConductors == 1):
        Dsc = code.dInches / 2.0 / 12.0
    else:
        Dsc=1
        print('Number of conductors in bundle given outside of known range for calculation.')
        print('Check line parameters and try again')

    Y = 1j*(2*math.pi*60) * 2 * math.pi * 8.854E-12/(math.log(Deq/Dsc))*1609*len

    return Y
