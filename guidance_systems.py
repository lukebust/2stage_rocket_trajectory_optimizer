import numpy as np


class GuidanceSystems:
    def __init__(
        self,
        vehicle,
        coordinate_systems,
        pitch_times,
        pitch_angles_deg,
    ):
        self.vehicle = vehicle
        self.coordinate_systems = coordinate_systems
        self.pitch_times = np.asarray(pitch_times, dtype=float)
        self.pitch_angles_deg = np.asarray(pitch_angles_deg, dtype=float)

    def compute_angle_above_horizontal(self, state, t):
        state_ECEF = self.coordinate_systems.convert_frame(
            state,
            t,
            "ECI",
            "ECEF",
        )
        velocity_ECEF = state_ECEF[3:6]

        R_ECEF_ENU = self.coordinate_systems.get_R_ECEF2ENU(state_ECEF)
        velocity_ENU = R_ECEF_ENU @ velocity_ECEF

        horizontal_speed = np.hypot(velocity_ENU[0], velocity_ENU[1])
        return np.arctan2(velocity_ENU[2], horizontal_speed)

    def compute_azimuth(self, state, t):
        state_ENU = self.coordinate_systems.convert_frame(state, t, "ECI", "ENU")
        east_velocity = state_ENU[3]
        north_velocity = state_ENU[4]

        if np.hypot(east_velocity, north_velocity) < 1.0:
            return None

        return np.arctan2(east_velocity, north_velocity)

    def compute_downrange_ENU(self, state, t):
        azimuth = self.compute_azimuth(state, t)
        d_hat = np.array([np.sin(azimuth), np.cos(azimuth), 0])
        return d_hat

    def set_pitch_angles(self, pitch_angles_deg):
        self.pitch_angles_deg = np.asarray(pitch_angles_deg, dtype=float)

    def pitch_at_time(self, t):
        pitch_deg = np.interp(t, self.pitch_times, self.pitch_angles_deg)
        return np.deg2rad(pitch_deg)

    def compute_staging_times(self):
        first_stage_burn_time = float(
            self.vehicle.stage_1_fuel_mass / self.vehicle.mass_flow_rate_stage_1
        )
        second_stage_burn_time = (
            self.vehicle.stage_2_fuel_mass / self.vehicle.mass_flow_rate_stage_2
        )
        return np.array([first_stage_burn_time, second_stage_burn_time])
