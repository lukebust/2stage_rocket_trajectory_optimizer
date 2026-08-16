import numpy as np

from environment import compute_atm_data

MU_EARTH = 3.986004418e14
R_EARTH = 6.371e6  # m


class Physics:
    def __init__(
        self,
        vehicle,
        coordinate_systems,
        guidance_systems,
        launch_parameters,
    ):
        self.vehicle = vehicle
        self.coordinate_systems = coordinate_systems
        self.guidance_systems = guidance_systems
        self.launch_parameters = launch_parameters

    def compute_g(self, state):
        r = state[0:3]  # meters
        g = -MU_EARTH / np.linalg.norm(r) ** 3 * r  # m/s^2
        return g

    def compute_drag_force(self, state, t):
        r_ECI = state[0:3]  # m
        alt = np.linalg.norm(r_ECI) - R_EARTH  # m
        atm_data = compute_atm_data(alt)
        density = atm_data[2]  # kg/m^3

        state_ECEF = self.coordinate_systems.convert_frame(state, t, "ECI", "ECEF")
        # assuming no wind
        v_ECEF = state_ECEF[3:6]  # m/s
        speed = np.linalg.norm(v_ECEF)  # m/s
        force_drag_simple_magnitude = (
            0.5 * density * speed**2 * self.vehicle.cd * self.vehicle.area
        )

        if speed == 0:
            return np.zeros(3)  # N

        drag_force_ECEF = -force_drag_simple_magnitude * v_ECEF / speed  # N
        drag_force_ECI = self.coordinate_systems.convert_vector(
            state,
            t,
            drag_force_ECEF,
            "ECEF",
            "ECI",
        )  # N
        return drag_force_ECI

    def compute_thrust_force(self, state, t, stage):
        if stage == "stage_1":
            mass_flow_rate = self.vehicle.mass_flow_rate_stage_1
            exit_velocity = self.vehicle.exit_velocity_stage_1
        elif stage == "stage_2":
            mass_flow_rate = self.vehicle.mass_flow_rate_stage_2
            exit_velocity = self.vehicle.exit_velocity_stage_2
        else:
            raise NotImplementedError(
                "This stage is not implemented. Use stage_1 or stage_2"
            )

        thrust_magnitude = mass_flow_rate * exit_velocity
        pitch = self.guidance_systems.pitch_at_time(t)
        azimuth = self.guidance_systems.compute_azimuth(state, t)

        # Hold the launch azimuth until horizontal motion is established.
        if azimuth is None:
            azimuth = self.launch_parameters.initial_azimuth

        thrust_hat_ENU = np.array(
            [
                np.cos(pitch) * np.sin(azimuth),  # east
                np.cos(pitch) * np.cos(azimuth),  # north
                np.sin(pitch),  # up
            ]
        )
        thrust_ENU = thrust_hat_ENU * thrust_magnitude
        thrust_ECI = self.coordinate_systems.convert_vector(
            state,
            t,
            thrust_ENU,
            "ENU",
            "ECI",
        )
        return thrust_ECI
