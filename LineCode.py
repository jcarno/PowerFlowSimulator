#object to hold line code for a given conductor (Data from ASCR or copper conductor data tables)
class LineCode:

    def __init__(self,name:str,dInches:float,GMRft:float,ampacity:int,R1perMi:float):
        self.name=name
        self.dInches=dInches
        self.ampacity=ampacity
        self.GMRft=GMRft
        self.R1perMi=R1perMi
