# import slack bus voltage?
import math

class Transformer:
    def __init__(self, name, bus1, bus2, v1, v2, s_rated, zt, x_r_ratio,bus1Conn,bus2Conn,groundingZ1,groundingZ2):
        self.name = name
        self.bus1 = bus1
        self.bus2 = bus2
        self.v1 = v1
        self.v2 = v2
        self.s_rated = s_rated
        self.zt = zt
        self.x_r_ratio = x_r_ratio
        self.bus1Conn=bus1Conn
        self.bus2Conn=bus2Conn
        self.groundingZ1=groundingZ1
        self.groundingZ2=groundingZ2

    def per_unit_Z(self, z_base):
        return (self.zt*((self.v1**2)/self.s_rated))/z_base*(math.cos(math.atan(self.x_r_ratio))+1j*math.sin(math.atan(self.x_r_ratio)))
