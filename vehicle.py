import numpy as np


class Vehicle:
    def __init__(
        self,
        radius,
        cd,
        stage_1_dry_mass,
        stage_2_dry_mass,
        stage_1_fuel_mass,
        stage_2_fuel_mass,
        payload_mass,
        mass_flow_rate_stage_1,
        mass_flow_rate_stage_2,
        exit_velocity_stage_1,
        exit_velocity_stage_2,
    ):
        self.radius = float(radius)
        self.area = float(np.pi * radius**2)
        self.cd = float(cd)
        self.stage_1_dry_mass = float(stage_1_dry_mass)
        self.stage_2_dry_mass = float(stage_2_dry_mass)
        self.stage_1_fuel_mass = float(stage_1_fuel_mass)
        self.stage_2_fuel_mass = float(stage_2_fuel_mass)
        self.payload_mass = float(payload_mass)
        self.mass_flow_rate_stage_1 = float(mass_flow_rate_stage_1)
        self.mass_flow_rate_stage_2 = float(mass_flow_rate_stage_2)
        self.exit_velocity_stage_1 = float(exit_velocity_stage_1)
        self.exit_velocity_stage_2 = float(exit_velocity_stage_2)
