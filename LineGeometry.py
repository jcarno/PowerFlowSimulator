class LineGeometry:
    #set up geometry to have a default of 1 conductor, and 1 foot spacing
    def __init__(self,nConductors=1,bSpacingft=1,Dab=1, Dbc=1, Dca=1):
        self.nConductors=nConductors
        self.bSpacingft=bSpacingft
        self.Dab=Dab
        self.Dbc=Dbc
        self.Dca=Dca