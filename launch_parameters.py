import numpy as np

R_EARTH = 6.371e6  # m
OMEGA_EARTH = 7.2921e-5  # rad/s


class LaunchParameters:
    def __init__(
        self,
        vehicle,
        initial_lat,
        initial_long,
        initial_altitude,
        initial_azimuth,
        coe,
    ):
        self.vehicle = vehicle
        self.initial_lat = float(np.deg2rad(initial_lat))
        self.initial_long = float(np.deg2rad(initial_long))
        self.initial_altitude = float(initial_altitude)
        self.initial_azimuth = float(np.deg2rad(initial_azimuth))
        self.a = float(coe[0])  # length of semi-major axis
        self.ecc = float(coe[1])  # eccentricity
        self.inc = float(np.deg2rad(coe[2]))  # inclination (rad)
        # self.omega = float(np.deg2rad(coe[3])) # RAAN (rad)
        # self.w = float(coe[4]) # argument of periapsis
        # self.nu = float(coe[5]) # true anomaly

    def initialize_launch(self):
        # INITIAL POSITION (ECI) m
        r0 = R_EARTH + self.initial_altitude
        x_initial = r0 * np.cos(self.initial_lat) * np.cos(self.initial_long)
        y_initial = r0 * np.cos(self.initial_lat) * np.sin(self.initial_long)
        z_initial = r0 * np.sin(self.initial_lat)
        r_initial = np.array([x_initial, y_initial, z_initial])

        # INITIAL VELOCITY (ECI) m/s
        omega_Earth = np.array([0, 0, OMEGA_EARTH])
        v_initial = np.cross(omega_Earth, r_initial)

        net_mass_initial = (
            self.vehicle.stage_1_dry_mass
            + self.vehicle.stage_1_fuel_mass
            + self.vehicle.stage_2_dry_mass
            + self.vehicle.stage_2_fuel_mass
            + self.vehicle.payload_mass
        )
        initial_state = np.concatenate((r_initial, v_initial, [net_mass_initial]))
        return initial_state
