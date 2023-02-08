#Bus Class to store bus names and total number of buses
#Modeled after Paulo Radatz's example in class

class Bus:
    numBuses=0

    #Initialization of a bus
    def __int__(self, name):
        self.name=name
        self.index=Bus.numBuses #keep track which bus this is in order of all buses

        self.voltage=None
        Bus.numBuses=Bus.numBuses+1

    #Function to set the voltage of a bus
    def set_bus_voltage(self,voltage):
        self.voltage=voltage


