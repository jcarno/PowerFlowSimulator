#Generator Class
class Generator:

    def __init__(self,name,bus,voltage,voltagePU,P_setpoint,Q_setpoint,Q_min,Q_max,Xg1,Xg2,Xg0,grounded,groundingZ):
        self.name=name
        self.bus=bus
        self.voltagePU=voltagePU
        self.voltage=voltage
        self.P=P_setpoint
        self.Q=Q_setpoint
        self.Q_min=Q_min
        self.Q_max=Q_max
        self.vControl=True
        self.zBaseOld=voltage**2/P_setpoint
        self.Xg1=Xg1
        self.Xg2=Xg2
        self.Xg0=Xg0
        self.grounded=grounded
        self.groundingZ=groundingZ    #ohms

    def set_Power(self,P):
        self.P=P
    def set_Reactive(self,Q):
        self.Q=Q


