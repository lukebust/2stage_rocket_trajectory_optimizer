import numpy as np


def compute_atm_data(altitude, launch_altitude=0, atm_data_seaLevel=None):
    # input: altitude, in meters
    # output: atm_data=[temp, pressure, density]

    # sea level constants if not provided
    MU_EARTH = 3.986004418e14
    R_EARTH = 6.371e6
    R = 287.05  # J/(kg*K)
    if atm_data_seaLevel is None:
        temp_seaLevel = 288.16  # K
        pressure_seaLevel = 101325  # Pa
        density_seaLevel = 1.2250  # kg/m^3
    else:
        temp_seaLevel = atm_data_seaLevel[0]  # K
        pressure_seaLevel = atm_data_seaLevel[1]  # Pa
        density_seaLevel = atm_data_seaLevel[2]  # kg/m^3

    # lapse rates up to edge of mesosphere
    a_troposphere = -0.0065  # K/m , until 11km
    # isothermal from 11km-25 km (tropopause)
    a_stratosphere = 0.0030  # K/m, from 25km-47km
    # isothermal from 47km-53km (stratopause)
    a_mesosphere = -0.0045  # K/m, from 53km-79km
    # isothermal from 79km-90km (mesopause)
    a_thermosphere = 0.0040  # K/m, from 90-100km
    # karman line at 100km (negligible air resistance)

    # compute gravitational acceleration
    def compute_grav(altitude):
        r = R_EARTH + altitude
        g = MU_EARTH / (r**2)  # m/s^2
        return g

    # Temperature Calculations
    temp_troposphere_top = temp_seaLevel + a_troposphere * (11000 - launch_altitude)
    temp_stratosphere_top = temp_troposphere_top + a_stratosphere * (47000 - 25000)
    temp_mesosphere_top = temp_stratosphere_top + a_mesosphere * (79000 - 53000)
    temp_thermosphere_top = temp_mesosphere_top + a_thermosphere * (100000 - 90000)
    # Pressure Calculations
    p_troposphere_top = pressure_seaLevel * (temp_troposphere_top / temp_seaLevel) ** (
        -compute_grav(11000) / (a_troposphere * R)
    )  # gradient
    p_tropopause_top = p_troposphere_top * np.exp(
        -compute_grav(25000) * (25000 - 11000) / (R * temp_troposphere_top)
    )  # isothermal
    p_stratosphere_top = p_tropopause_top * (
        temp_stratosphere_top / temp_troposphere_top
    ) ** (
        -compute_grav(47000) / (a_stratosphere * R)
    )  # gradient
    p_stratopause_top = p_stratosphere_top * np.exp(
        -compute_grav(53000) * (53000 - 47000) / (R * temp_stratosphere_top)
    )  # isothermal
    p_mesosphere_top = p_stratopause_top * (
        temp_mesosphere_top / temp_stratosphere_top
    ) ** (
        -compute_grav(79000) / (a_mesosphere * R)
    )  # gradient
    p_mesopause_top = p_mesosphere_top * np.exp(
        -compute_grav(90000) * (90000 - 79000) / (R * temp_mesosphere_top)
    )  # isothermal
    p_thermosphere_top = p_mesopause_top * (
        temp_thermosphere_top / temp_mesosphere_top
    ) ** (
        -compute_grav(100000) / (a_thermosphere * R)
    )  # gradient
    # Density Calculations
    rho_troposphere_top = density_seaLevel * (temp_troposphere_top / temp_seaLevel) ** (
        -compute_grav(11000) / (a_troposphere * R) - 1
    )  # gradient
    rho_tropopause_top = rho_troposphere_top * np.exp(
        -compute_grav(25000) * (25000 - 11000) / (R * temp_troposphere_top)
    )  # isothermal
    rho_stratosphere_top = rho_tropopause_top * (
        temp_stratosphere_top / temp_troposphere_top
    ) ** (
        -compute_grav(47000) / (a_stratosphere * R) - 1
    )  # gradient
    rho_stratopause_top = rho_stratosphere_top * np.exp(
        -compute_grav(53000) * (53000 - 47000) / (R * temp_stratosphere_top)
    )  # isothermal
    rho_mesosphere_top = rho_stratopause_top * (
        temp_mesosphere_top / temp_stratosphere_top
    ) ** (
        -compute_grav(79000) / (a_mesosphere * R) - 1
    )  # gradient
    rho_mesopause_top = rho_mesosphere_top * np.exp(
        -compute_grav(90000) * (90000 - 79000) / (R * temp_mesosphere_top)
    )  # isothermal
    rho_thermosphere_top = rho_mesopause_top * (
        temp_thermosphere_top / temp_mesosphere_top
    ) ** (
        -compute_grav(100000) / (a_thermosphere * R) - 1
    )  # gradient

    if altitude < 11000:  # troposphere
        temperature = temp_seaLevel + a_troposphere * (altitude - launch_altitude)
        pressure = pressure_seaLevel * (temperature / temp_seaLevel) ** (
            -compute_grav(altitude) / (a_troposphere * R)
        )
        density = density_seaLevel * (temperature / temp_seaLevel) ** (
            -compute_grav(altitude) / (a_troposphere * R) - 1
        )
    elif altitude < 25000:  # tropopause
        temperature = temp_troposphere_top
        pressure = p_troposphere_top * np.exp(
            -compute_grav(altitude) * (altitude - 11000) / (R * temperature)
        )
        density = rho_troposphere_top * np.exp(
            -compute_grav(altitude) * (altitude - 11000) / (R * temperature)
        )
    elif altitude < 47000:  # stratosphere
        temperature = temp_troposphere_top + a_stratosphere * (altitude - 25000)
        pressure = p_tropopause_top * (temperature / temp_troposphere_top) ** (
            -compute_grav(altitude) / (a_stratosphere * R)
        )
        density = rho_tropopause_top * (temperature / temp_troposphere_top) ** (
            -compute_grav(altitude) / (a_stratosphere * R) - 1
        )
    elif altitude < 53000:  # stratopause
        temperature = temp_stratosphere_top
        pressure = p_stratosphere_top * np.exp(
            -compute_grav(altitude) * (altitude - 47000) / (R * temperature)
        )
        density = rho_stratosphere_top * np.exp(
            -compute_grav(altitude) * (altitude - 47000) / (R * temperature)
        )
    elif altitude < 79000:  # mesosphere
        temperature = temp_stratosphere_top + a_mesosphere * (altitude - 53000)
        pressure = p_stratopause_top * (temperature / temp_stratosphere_top) ** (
            -compute_grav(altitude) / (a_mesosphere * R)
        )
        density = rho_stratopause_top * (temperature / temp_stratosphere_top) ** (
            -compute_grav(altitude) / (a_mesosphere * R) - 1
        )
    elif altitude < 90000:  # mesopause
        temperature = temp_mesosphere_top
        pressure = p_mesosphere_top * np.exp(
            -compute_grav(altitude) * (altitude - 79000) / (R * temperature)
        )
        density = rho_mesosphere_top * np.exp(
            -compute_grav(altitude) * (altitude - 79000) / (R * temperature)
        )
    elif altitude < 100000:  # thermosphere
        temperature = temp_mesosphere_top + a_thermosphere * (altitude - 90000)
        pressure = p_mesopause_top * (temperature / temp_mesosphere_top) ** (
            -compute_grav(altitude) / (a_thermosphere * R)
        )
        density = rho_mesopause_top * (temperature / temp_mesosphere_top) ** (
            -compute_grav(altitude) / (a_thermosphere * R) - 1
        )
    else:
        # for when spacecraft is too high to experience drag
        temperature = 0
        pressure = 0
        density = 0
    atm_data = np.array([temperature, pressure, density])
    return atm_data
