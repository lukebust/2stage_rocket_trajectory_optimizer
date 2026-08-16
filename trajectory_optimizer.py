import numpy as np
from scipy.optimize import differential_evolution

MU_EARTH = 3.986004418e14
R_EARTH = 6.371e6


class TargetOrbit:
    def __init__(self, altitude_m, eccentricity, inclination_deg):
        self.a = R_EARTH + float(altitude_m)
        self.e = float(eccentricity)
        self.i = np.deg2rad(inclination_deg)


def get_orbit_from_state(state):
    r_vec = state[0:3]
    v_vec = state[3:6]

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    e_vec = np.cross(v_vec, h_vec) / MU_EARTH - r_vec / r
    eccentricity = np.linalg.norm(e_vec)

    specific_energy = v**2 / 2 - MU_EARTH / r
    semi_major_axis = -MU_EARTH / (2 * specific_energy)

    inclination = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    return {
        "a": semi_major_axis,
        "e": eccentricity,
        "i": inclination,
    }


def get_orbital_period(semi_major_axis):
    return 2 * np.pi * np.sqrt(semi_major_axis**3 / MU_EARTH)


class TrajectoryOptimizer:
    def __init__(self, simulation, target_orbit):
        self.simulation = simulation
        self.target_orbit = target_orbit

        # p2 through p12; the initial two 90-degree values stay fixed.
        self.initial_parameters = np.array(
            [
                85,
                75,
                60,
                45,
                35,
                25,
                15,
                5,
                0,
                -5,
                -15,
                317.25,
            ]
        )

        self.bounds = [
            (80, 90),  # p2: 30 s
            (65, 88),  # p3: 60 s
            (45, 80),  # p4: 100 s
            (25, 65),  # p5: 130 s
            (15, 50),  # p6: staging
            (5, 40),  # p7
            (0, 30),  # p8
            (-5, 20),  # p9
            (-12, 12),  # p10
            (-15, 8),  # p11
            (-20, 5),  # p12
            (100, 322.4347826086956),  # stage-2 cutoff time (s)
        ]

    def set_candidate_guidance(self, pitch_parameters):
        pitch_angles_deg = np.array(
            [
                90,
                90,
                *pitch_parameters,
            ]
        )

        self.simulation.guidance_systems.set_pitch_angles(pitch_angles_deg)

    def objective(self, parameters):
        # Reject pitch-up profiles for this simple gravity-turn optimizer.
        pitch_parameters = parameters[:-1]
        stage_2_cutoff_time = parameters[-1]

        if np.any(np.diff(pitch_parameters) > 0):
            return 1e12

        self.set_candidate_guidance(pitch_parameters)

        _, solution_stage_2, _ = self.simulation.propagate_simulation(
            include_coast=False,
            stage_2_cutoff_time=stage_2_cutoff_time,
        )

        if solution_stage_2.status == 1:
            return 1e14

        # This is the state at second-stage burnout / orbital insertion.
        final_state = solution_stage_2.y[:, -1]
        achieved = get_orbit_from_state(final_state)

        a = achieved["a"]
        e = achieved["e"]
        i = achieved["i"]

        if not np.all(np.isfinite([a, e, i])):
            return 1e15

        # Must be a bound elliptical orbit.
        if a <= 0 or e >= 1:
            return 1e12

        perigee = a * (1 - e)
        minimum_perigee = R_EARTH + 100_000

        if perigee < minimum_perigee:
            return 1e10 + (minimum_perigee - perigee)

        # Errors are normalized so their different units are comparable.
        a_error = (a - self.target_orbit.a) / self.target_orbit.a
        e_error = (e - self.target_orbit.e) / 0.01
        i_error = (i - self.target_orbit.i) / np.deg2rad(1)

        return a_error**2 + e_error**2 + i_error**2

    def optimize(self):
        result = differential_evolution(
            self.objective,
            bounds=self.bounds,
            seed=42,
            x0=self.initial_parameters,
            popsize=3,
            maxiter=6,
            polish=False,
        )

        # Leave the guidance system with the best discovered profile.
        self.set_candidate_guidance(result.x[:-1])

        return result
