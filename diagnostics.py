import numpy as np
from environment import compute_atm_data

R_EARTH = 6.371e6  #
GAMMA = 1.4
R = 287.05


class Diagnostics:
    def __init__(self, vehicle, simulation, coordinate_systems):
        self.vehicle = vehicle
        self.simulation = simulation
        self.coordinate_systems = coordinate_systems

        self.solution_1, self.solution_2, self.solution_3 = (
            self.simulation.propagate_simulation()
        )

    def get_altitude(self):

        # STAGE 1
        x1 = self.solution_1.y[0]
        y1 = self.solution_1.y[1]
        z1 = self.solution_1.y[2]
        r_1 = np.array([x1, y1, z1])  # each entry is a row
        altitude_1 = np.linalg.norm(r_1, axis=0) - R_EARTH  # norm of each column

        # STAGE 2
        x2 = self.solution_2.y[0]
        y2 = self.solution_2.y[1]
        z2 = self.solution_2.y[2]
        r_2 = np.array([x2, y2, z2])  # each entry is a row
        altitude_2 = np.linalg.norm(r_2, axis=0) - R_EARTH  # norm of each column

        # STAGE COAST
        x3 = self.solution_3.y[0]
        y3 = self.solution_3.y[1]
        z3 = self.solution_3.y[2]
        r_3 = np.array([x3, y3, z3])  # each entry is a row
        altitude_3 = np.linalg.norm(r_3, axis=0) - R_EARTH  # norm of each column

        return altitude_1, altitude_2, altitude_3

    def get_speed(self):

        # STAGE 1
        vx1 = self.solution_1.y[3]
        vy1 = self.solution_1.y[4]
        vz1 = self.solution_1.y[5]
        v_1 = np.array([vx1, vy1, vz1])  # each entry is a row
        speed_1 = np.linalg.norm(v_1, axis=0)  # norm of each column

        # STAGE 2
        vx2 = self.solution_2.y[3]
        vy2 = self.solution_2.y[4]
        vz2 = self.solution_2.y[5]
        v_2 = np.array([vx2, vy2, vz2])  # each entry is a row
        speed_2 = np.linalg.norm(v_2, axis=0)  # norm of each column

        # STAGE COAST
        vx3 = self.solution_3.y[3]
        vy3 = self.solution_3.y[4]
        vz3 = self.solution_3.y[5]
        v_3 = np.array([vx3, vy3, vz3])  # each entry is a row
        speed_3 = np.linalg.norm(v_3, axis=0)  # norm of each column

        return speed_1, speed_2, speed_3

    def get_fuel_masses(self):
        vehicle = self.vehicle

        # STAGE 1
        total_fuel_1 = (
            self.solution_1.y[6]
            - vehicle.stage_1_dry_mass
            - vehicle.stage_2_dry_mass
            - vehicle.payload_mass
        )

        stage_1_fuel_1 = total_fuel_1 - vehicle.stage_2_fuel_mass
        stage_2_fuel_1 = np.full_like(stage_1_fuel_1, vehicle.stage_2_fuel_mass)

        # STAGE 2
        stage_2_fuel_2 = (
            self.solution_2.y[6] - vehicle.stage_2_dry_mass - vehicle.payload_mass
        )
        stage_1_fuel_2 = np.zeros_like(stage_2_fuel_2)
        total_fuel_2 = stage_2_fuel_2

        # COAST
        stage_1_fuel_3 = np.zeros_like(self.solution_3.t)
        stage_2_fuel_3 = np.zeros_like(self.solution_3.t)
        total_fuel_3 = np.zeros_like(self.solution_3.t)

        return (
            (total_fuel_1, total_fuel_2, total_fuel_3),
            (stage_1_fuel_1, stage_1_fuel_2, stage_1_fuel_3),
            (stage_2_fuel_1, stage_2_fuel_2, stage_2_fuel_3),
        )

    def compute_acceleration_magnitude(self):
        phase_data = [
            (self.solution_1, self.simulation.eq_motion_1),
            (self.solution_2, self.simulation.eq_motion_2),
            (self.solution_3, self.simulation.eq_motion_coast),
        ]

        acceleration_magnitudes = []

        for solution, equation_of_motion in phase_data:
            derivatives = []

            for t, state in zip(solution.t, solution.y.T):
                derivative = equation_of_motion(t, state)
                derivatives.append(derivative)

            # This line is required.
            derivatives = np.array(derivatives)

            acceleration_magnitude = np.linalg.norm(
                derivatives[:, 3:6],
                axis=1,
            )

            acceleration_magnitudes.append(acceleration_magnitude)

        return tuple(acceleration_magnitudes)

    def get_times(self):
        return (
            self.solution_1.t,
            self.solution_2.t,
            self.solution_3.t,
        )

    def get_q(self):
        coordinate_systems = self.simulation.physics.coordinate_systems

        phase_solutions = [
            self.solution_1,
            self.solution_2,
            self.solution_3,
        ]

        dynamic_pressures = []

        for solution in phase_solutions:
            q_values = []

            # Get one matching time/state pair at a time.
            for t, state in zip(solution.t, solution.y.T):

                # Altitude from the ECI position.
                position_ECI = state[0:3]
                altitude = np.linalg.norm(position_ECI) - R_EARTH

                # Convert the whole state to ECEF.
                # Its velocity is relative to Earth's rotating surface/atmosphere.
                state_ECEF = coordinate_systems.convert_frame(
                    state,
                    t,
                    "ECI",
                    "ECEF",
                )
                velocity_ECEF = state_ECEF[3:6]
                speed_ECEF = np.linalg.norm(velocity_ECEF)

                # Get atmospheric density at this altitude.
                density = compute_atm_data(altitude)[2]

                q = 0.5 * density * speed_ECEF**2
                q_values.append(q)

            # Convert the normal Python list to a NumPy array.
            dynamic_pressures.append(np.array(q_values))

        # Returns:
        # q1 = stage-1 dynamic pressure array
        # q2 = stage-2 dynamic pressure array
        # q3 = coast dynamic pressure array
        return tuple(dynamic_pressures)

    def get_mach(self):
        coordinate_systems = self.simulation.physics.coordinate_systems

        phase_solutions = [
            self.solution_1,
            self.solution_2,
            self.solution_3,
        ]

        mach_numbers = []

        for solution in phase_solutions:
            mach_values = []

            # Get one matching time/state pair at a time.
            for t, state in zip(solution.t, solution.y.T):

                # Altitude from the ECI position.
                position_ECI = state[0:3]
                altitude = np.linalg.norm(position_ECI) - R_EARTH

                # Convert the whole state to ECEF.
                # Its velocity is relative to Earth's rotating surface/atmosphere.
                state_ECEF = coordinate_systems.convert_frame(
                    state,
                    t,
                    "ECI",
                    "ECEF",
                )

                velocity_ECEF = state_ECEF[3:6]
                speed_ECEF = np.linalg.norm(velocity_ECEF)

                # Get atmospheric temperature at this altitude.
                temperature = compute_atm_data(altitude)[0]
                density = compute_atm_data(altitude)[2]

                # Mach number
                if temperature <= 0 or density <= 0:
                    mach = np.nan
                else:
                    alpha = np.sqrt(GAMMA * R * temperature)
                    mach = speed_ECEF / alpha

                mach_values.append(mach)

            # Convert the normal Python list to a NumPy array.
            mach_numbers.append(np.array(mach_values))

        # Returns:
        # q1 = stage-1 dynamic pressure array
        # q2 = stage-2 dynamic pressure array
        # q3 = coast dynamic pressure array
        return tuple(mach_numbers)

    def get_reynolds(self):
        mu_0 = 1.716e-5
        T0 = 273.15
        S = 111
        coordinate_systems = self.simulation.physics.coordinate_systems

        phase_solutions = [
            self.solution_1,
            self.solution_2,
            self.solution_3,
        ]

        reynolds_numbers = []

        for solution in phase_solutions:
            reynolds_values = []

            # Get one matching time/state pair at a time.
            for t, state in zip(solution.t, solution.y.T):

                # Altitude from the ECI position.
                position_ECI = state[0:3]
                altitude = np.linalg.norm(position_ECI) - R_EARTH

                # Convert the whole state to ECEF.
                # Its velocity is relative to Earth's rotating surface/atmosphere.
                state_ECEF = coordinate_systems.convert_frame(
                    state,
                    t,
                    "ECI",
                    "ECEF",
                )

                velocity_ECEF = state_ECEF[3:6]
                speed_ECEF = np.linalg.norm(velocity_ECEF)

                # Get atmospheric data at this altitude.
                temperature = compute_atm_data(altitude)[0]
                density = compute_atm_data(altitude)[2]

                char_length = 2 * self.vehicle.radius

                if temperature <= 0 or density <= 0:
                    reynolds = np.nan
                else:
                    mu = (
                        mu_0
                        * (temperature / T0) ** (3 / 2)
                        * (T0 + S)
                        / (temperature + S)
                    )
                    reynolds = density * speed_ECEF * char_length / mu

                reynolds_values.append(reynolds)

            # Convert the normal Python list to a NumPy array.
            reynolds_numbers.append(np.array(reynolds_values))

        # Returns:
        # q1 = stage-1 dynamic pressure array
        # q2 = stage-2 dynamic pressure array
        # q3 = coast dynamic pressure array
        return tuple(reynolds_numbers)
