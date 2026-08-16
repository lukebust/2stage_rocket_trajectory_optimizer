import numpy as np
from scipy.integrate import solve_ivp

R_EARTH = 6.371e6  # m


def ground_impact(t, state):
    r = state[0:3]
    altitude = np.linalg.norm(r) - R_EARTH
    return altitude


ground_impact.terminal = True
ground_impact.direction = -1


class Simulation:
    def __init__(
        self,
        vehicle,
        physics,
        launch_parameters,
        guidance_systems,
        t_f,
    ):
        self.vehicle = vehicle
        self.physics = physics
        self.launch_parameters = launch_parameters
        self.guidance_systems = guidance_systems
        self.t_f = t_f

    def eq_motion_1(self, t, state):
        dydt = np.zeros(7)
        v = state[3:6]
        mass = state[6]

        thrust_force = self.physics.compute_thrust_force(state, t, "stage_1")
        drag_force = self.physics.compute_drag_force(state, t)
        gravity_acceleration = self.physics.compute_g(state)
        dv = thrust_force / mass + drag_force / mass + gravity_acceleration

        dydt[0:3] = v
        dydt[3:6] = dv

        if np.linalg.norm(thrust_force) == 0:
            dydt[6] = 0
        else:
            dydt[6] = -self.vehicle.mass_flow_rate_stage_1
        return dydt

    def eq_motion_2(self, t, state):
        dydt = np.zeros(7)
        v = state[3:6]
        mass = state[6]

        thrust_force = self.physics.compute_thrust_force(state, t, "stage_2")
        drag_force = self.physics.compute_drag_force(state, t)
        gravity_acceleration = self.physics.compute_g(state)
        dv = thrust_force / mass + drag_force / mass + gravity_acceleration

        dydt[0:3] = v
        dydt[3:6] = dv

        if np.linalg.norm(thrust_force) == 0:
            dydt[6] = 0
        else:
            dydt[6] = -self.vehicle.mass_flow_rate_stage_2
        return dydt

    def eq_motion_coast(self, t, state):
        dydt = np.zeros(7)
        v = state[3:6]
        mass = state[6]

        drag_force = self.physics.compute_drag_force(state, t)
        gravity_acceleration = self.physics.compute_g(state)
        dv = drag_force / mass + gravity_acceleration

        dydt[0:3] = v
        dydt[3:6] = dv

        return dydt

    def propagate_simulation(
        self,
        include_coast=True,
        stage_2_cutoff_time=None,
        coast_duration=None,
    ):
        staging_times = self.guidance_systems.compute_staging_times()
        stage_1_burn_time = staging_times[
            0
        ]  # time it takes for stage 1 to use all fuel
        stage_2_burn_time = staging_times[
            1
        ]  # time it takes for stage 2 to use all fuel

        if stage_2_cutoff_time is None:
            stage_2_cutoff_time = stage_2_burn_time

        # STAGE 1: Initial Burn from all 9 Merlin engines, from launch with stage 1 dry mass
        solution_stage_1 = solve_ivp(
            fun=self.eq_motion_1,
            t_span=(0, stage_1_burn_time),
            y0=self.launch_parameters.initialize_launch(),
            rtol=1e-8,
            atol=1e-6,
            max_step=1.0,
            events=ground_impact,
        )

        # STAGE 2: Merlin Vacuum Engine burn

        # Stores the final state of stage 1 as the initial state of stage 2
        # Removes stage 1 dry mass
        state2 = solution_stage_1.y[:, -1].copy()
        state2[6] -= self.vehicle.stage_1_dry_mass

        solution_stage_2 = solve_ivp(
            fun=self.eq_motion_2,
            t_span=(
                stage_1_burn_time,
                stage_1_burn_time + stage_2_cutoff_time,
            ),
            y0=state2,
            rtol=1e-8,
            atol=1e-6,
            max_step=1.0,
            events=ground_impact,
        )

        # STAGE 3: Free Orbit/Coast
        if not include_coast:
            return solution_stage_1, solution_stage_2, None

        if coast_duration is None:
            coast_duration = self.t_f - (stage_1_burn_time + stage_2_cutoff_time)

        if coast_duration <= 0:
            raise ValueError("coast_duration must be positive")

        state3 = solution_stage_2.y[:, -1].copy()
        coast_start_time = stage_1_burn_time + stage_2_cutoff_time

        solution_coast = solve_ivp(
            fun=self.eq_motion_coast,
            t_span=(coast_start_time, coast_start_time + coast_duration),
            y0=state3,
            rtol=1e-8,
            atol=1e-6,
            max_step=1.0,
            events=ground_impact,
        )

        return solution_stage_1, solution_stage_2, solution_coast
