# import slack bus voltage?
class Transformer:
    def __init__(self, name, bus_1, bus_2, v_1, v_2, s_rated, zt, x_r_ratio):
        self.name = name
        self.bus_1 = bus_1
        self.bus_2 = bus_2
        self.v_1 = v_1
        self.v_2 = v_2
        self.s_rated = s_rated
        self.zt = zt
        self.x_r_ratio = x_r_ratio

    def per_unit_impedance(self, z_base):
        return (self.zt*((self.v_1**2)/self.s_rated))/z_base
