class Bus:
    #modeled after Paulo's example class
    numBuses=0

    #Initialization of a bus
    def __init__(self,name,v=1.0,ang=0,type="PQ"):
        self.name=name
        self.index=Bus.numBuses #keep track which bus this is in order of all buses

        self.voltage:float=v
        self.angle:float=ang
        self.type=type
        Bus.numBuses=Bus.numBuses+1

    #Function to set the voltage of a bus
    def set_bus_voltage(self,voltage,angle=0):
        self.voltage=voltage
        self.angle=0

    def set_type(self,type):
        self.type=type
