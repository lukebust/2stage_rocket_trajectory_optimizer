import numpy as np

OMEGA_EARTH = 7.2921e-5  # rad/s


class CoordinateSystems:

    def get_coord(self, state):
        long = np.atan2(state[1], state[0])
        lat = np.atan2(state[2], np.sqrt(state[0] ** 2 + state[1] ** 2))
        coord = np.array([lat, long])
        return coord

    def get_R_ECI2ECEF(self, t):
        theta = OMEGA_EARTH * t
        R_ECI2ECEF = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        return R_ECI2ECEF

    def get_R_ECEF2ENU(self, state):
        coord = self.get_coord(state)
        lat = coord[0]
        long = coord[1]
        R_ECEF2ENU = np.array(
            [
                [-np.sin(long), np.cos(long), 0],
                [-np.sin(lat) * np.cos(long), -np.sin(lat) * np.sin(long), np.cos(lat)],
                [np.cos(lat) * np.cos(long), np.cos(lat) * np.sin(long), np.sin(lat)],
            ]
        )
        return R_ECEF2ENU

    def convert_frame(self, state, t, initial_frame, final_frame):
        if initial_frame == "ECI":
            if final_frame == "ECEF":
                r_ECI = state[0:3]  # m
                v_ECI = state[3:6]  # m/s
                state_ECEF = np.zeros(7)
                R = self.get_R_ECI2ECEF(t)
                state_ECEF[0:3] = R @ r_ECI
                state_ECEF[3:6] = R @ (v_ECI - np.cross([0, 0, OMEGA_EARTH], r_ECI))
                state_ECEF[6] = state[6]  # kg
                return state_ECEF

            elif (
                final_frame == "ENU"
            ):  # the position vector is in ECEF, but the velocity uses the ENU basis #confusing!
                state_ECEF = self.convert_frame(state, t, initial_frame, "ECEF")
                state_ENU = self.convert_frame(state_ECEF, t, "ECEF", "ENU")
                return state_ENU

            else:
                raise NotImplementedError(
                    "This coordinate frame conversion is not supported. Use ECI, ECEF, or ENU w/ position in terms of ECEF, velocity as a ENU basis."
                )

        elif initial_frame == "ECEF":
            if (
                final_frame == "ENU"
            ):  # the position vector is in ECEF, but the velocity uses ENU basis!
                R = self.get_R_ECEF2ENU(state)
                state_ENU = np.zeros(7)
                state_ENU[0:3] = state[0:3]
                state_ENU[3:6] = R @ state[3:6]
                state_ENU[6] = state[6]
                return state_ENU

            elif final_frame == "ECI":
                R = self.get_R_ECI2ECEF(t).T
                r_ECEF = state[0:3]  # m
                v_ECEF = state[3:6]  # m/s
                state_ECI = np.zeros(7)
                state_ECI[0:3] = R @ r_ECEF
                state_ECI[3:6] = R @ (v_ECEF + np.cross([0, 0, OMEGA_EARTH], r_ECEF))
                state_ECI[6] = state[6]  # kg
                return state_ECI

            else:
                raise NotImplementedError(
                    "This coordinate frame conversion is not supported. Use ECI, ECEF, or ENU w/ position in terms of ECEF, velocity as a ENU basis."
                )

        elif initial_frame == "ENU":
            if final_frame == "ECEF":
                R = self.get_R_ECEF2ENU(state)
                state_ECEF = np.zeros(7)
                state_ECEF[0:3] = state[0:3]
                state_ECEF[3:6] = R.T @ state[3:6]
                state_ECEF[6] = state[6]
                return state_ECEF

            elif final_frame == "ECI":
                state_ECEF = self.convert_frame(state, t, initial_frame, "ECEF")
                state_ECI = self.convert_frame(state_ECEF, t, "ECEF", "ECI")
                return state_ECI

            else:
                raise NotImplementedError(
                    "This coordinate frame conversion is not supported. Use ECI, ECEF, or ENU w/ position in terms of ECEF, velocity as a ENU basis."
                )

        else:
            raise NotImplementedError(
                "This coordinate frame conversion is not supported. Use ECI, ECEF, or ENU w/ position in terms of ECEF, velocity as a ENU basis."
            )

    def convert_vector(self, state, t, vec, initial_frame, final_frame):
        if initial_frame == "ECEF":
            if final_frame == "ECI":
                R = self.get_R_ECI2ECEF(t).T
                vec_ECI = R @ vec
                return vec_ECI

            elif final_frame == "ENU":
                R = self.get_R_ECEF2ENU(state)
                vec_ENU = R @ vec
                return vec_ENU

            else:
                raise NotImplementedError(
                    "This coordinate frame conversion is not supported. Use ECI, ECEF, or ENU w/ position in terms of ECEF, velocity as a ENU basis."
                )

        elif initial_frame == "ECI":
            if final_frame == "ECEF":
                R = self.get_R_ECI2ECEF(t)
                vec_ECEF = R @ vec
                return vec_ECEF

            elif final_frame == "ENU":
                state_ECEF = self.convert_frame(state, t, "ECI", "ECEF")
                vec_ECEF = self.convert_vector(state, t, vec, initial_frame, "ECEF")
                R = self.get_R_ECEF2ENU(state_ECEF)
                vec_ENU = R @ vec_ECEF
                return vec_ENU

            else:
                raise NotImplementedError(
                    "This coordinate frame conversion is not supported. Use ECI, ECEF, or ENU w/ position in terms of ECEF, velocity as a ENU basis."
                )

        elif initial_frame == "ENU":
            if final_frame == "ECEF":
                state_ECEF = self.convert_frame(state, t, "ECI", "ECEF")
                R = self.get_R_ECEF2ENU(state_ECEF).T
                vec_ECEF = R @ vec
                return vec_ECEF

            elif final_frame == "ECI":
                state_ECEF = self.convert_frame(state, t, "ECI", "ECEF")
                R_ECEF2ENU = self.get_R_ECEF2ENU(state_ECEF)
                vec_ECEF = R_ECEF2ENU.T @ vec

                R = self.get_R_ECI2ECEF(t).T
                vec_ECI = R @ vec_ECEF
                return vec_ECI

            else:
                raise NotImplementedError(
                    "This coordinate frame conversion is not supported. Use ECI, ECEF, or ENU w/ position in terms of ECEF, velocity as a ENU basis."
                )

        else:
            raise NotImplementedError(
                "This coordinate frame conversion is not supported. Use ECI, ECEF, or ENU w/ position in terms of ECEF, velocity as a ENU basis."
            )
