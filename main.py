import numpy as np

from vehicle import Vehicle
from simulation import Simulation
from launch_parameters import LaunchParameters
from guidance_systems import GuidanceSystems
from physics import Physics
from coordinate_systems import CoordinateSystems
from plotting import Plotting
from diagnostics import Diagnostics
from trajectory_optimizer import (
    TargetOrbit,
    TrajectoryOptimizer,
    get_orbit_from_state,
    get_orbital_period,
)

R_EARTH = 6.371e6

if __name__ == "__main__":
    # region VEHICLE PARAMETERS
    stage_1_dry_mass = 26100  # kg
    stage_1_fuel_mass = 411300  # kg
    stage_2_dry_mass = 3900  # kg
    stage_2_fuel_mass = 92700  # kg
    payload_mass = 15000  # kg
    cd = 0.4  # dimensionless
    radius = 2.6  # m
    mass_flow_rate_stage_1 = 2750  # kg/s
    exit_velocity_stage_1 = 2735  # m/s
    mass_flow_rate_stage_2 = 287.5
    exit_velocity_stage_2 = 3500
    # endregion

    # region LAUNCH/ORBIT PARAMETERS
    initial_lat = 0  # deg
    initial_long = 0  # deg
    initial_altitude = 0  # m
    initial_azimuth = 80  # deg
    # 6 CLASSICAL ORBITAL ELEMENTS
    altitude = 7000  # km above Earth for the target semi-major axis
    a = altitude * 1000 + R_EARTH
    e = 0.5  # dimensionless
    i = 90-initial_azimuth  # deg
    # this program doesnt worry about RAAN, AoP, or true anomaly yet
    coe = np.array([a, e, i, 0, 0, 0])
    # endregion

    # region SIMULATION PARAMETERS
    t_f = 5000
    pitch_times = np.array(
        [
            0,
            10,
            30,
            60,
            100,
            130,
            149.56,  # stage 1 burn ends
            180,
            240,
            320,
            400,
            450,
            471.99,  # stage 2 burnout
        ]
    )
    pitch_angles_initial = np.array(
        [
            90,
            90,
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
        ]
    )
    # endregion

    vehicle = Vehicle(
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
    )
    coordinate_systems = CoordinateSystems()
    guidance_systems = GuidanceSystems(
        vehicle,
        coordinate_systems,
        pitch_times,
        pitch_angles_initial,
    )
    launch_parameters = LaunchParameters(
        vehicle,
        initial_lat,
        initial_long,
        initial_altitude,
        initial_azimuth,
        coe,
    )
    physics = Physics(
        vehicle,
        coordinate_systems,
        guidance_systems,
        launch_parameters,
    )
    simulation = Simulation(
        vehicle,
        physics,
        launch_parameters,
        guidance_systems,
        t_f,
    )
    target_orbit = TargetOrbit(
        altitude_m=altitude*1000,
        eccentricity=e,
        inclination_deg=i,
    )

    optimizer = TrajectoryOptimizer(
        simulation,
        target_orbit
    )

    result = optimizer.optimize()

    diagnostics = Diagnostics(
        vehicle,
        simulation,
        coordinate_systems)
    plotting = Plotting(simulation, diagnostics)

    print("Best cost:", result.fun)
    print("Best p2 through p12 and cutoff time:", result.x)

    stage_2_cutoff_time = result.x[-1]
    _, insertion_solution, _ = simulation.propagate_simulation(
        include_coast=False,
        stage_2_cutoff_time=stage_2_cutoff_time,
    )
    insertion_orbit = get_orbit_from_state(insertion_solution.y[:, -1])
    orbital_period = get_orbital_period(insertion_orbit["a"])

    print("Insertion semi-major axis (km):", insertion_orbit["a"] / 1000)
    print("Insertion eccentricity:", insertion_orbit["e"])
    print("Insertion inclination (deg):", np.rad2deg(insertion_orbit["i"]))
    print("Orbital period (s):", orbital_period)
    print("The target insertion semi-major axis (km) was ", a)
    print("The target insertion eccentricity was ", e)
    print("The target insertion inclination (degrees) was ", i)


    solution_stage_1, solution_stage_2, solution_coast = (
        simulation.propagate_simulation(
            stage_2_cutoff_time=stage_2_cutoff_time,
            coast_duration=orbital_period,
        )
    )


    
    """plotting.plot_3d_trajectory(
        solution_stage_1,
        solution_stage_2,
        solution_coast,
    )"""


    plotting.plot_speed_and_acceleration()
    plotting.plot_dynamic_pressure_vs_altitude()
    plotting.plot_dynamic_pressure_vs_time()
    plotting.plot_reynolds_numbers_vs_time()
    plotting.plot_reynolds_numbers_vs_altitude()
    plotting.plot_mach_numbers_vs_time()
    plotting.plot_mach_numbers_vs_altitude()
