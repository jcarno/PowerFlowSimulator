#Generator Class
class Generator:

    def __init__(self,name,bus,voltage,P_setpoint,Q_setpoint=0,Q_min=-10000,Q_max=10000):
        self.name=name
        self.bus=bus
        self.voltage=voltage
        self.P=P_setpoint
        self.Q=Q_setpoint
        self.Q_min=Q_min
        self.Q_max=Q_max
        self.vControl=True

    def set_Power(self,P):
        self.P=P
    def set_Reactive(self,Q):
        self.Q=Q
