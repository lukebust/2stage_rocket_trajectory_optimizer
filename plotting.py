import numpy as np
import matplotlib.pyplot as plt

R_EARTH = 6.371e6  # m


class Plotting:
    def __init__(
        self,
        simulation,
        diagnostics=None,
    ):
        self.simulation = simulation
        self.diagnostics = diagnostics

    def plot_3d_trajectory(self, solution_stage_1, solution_stage_2, solution_coast):

        R_EARTH = 6.371e6  # m

        # Stage 1 position
        x1 = solution_stage_1.y[0]
        y1 = solution_stage_1.y[1]
        z1 = solution_stage_1.y[2]

        # Stage 2 position
        x2 = solution_stage_2.y[0]
        y2 = solution_stage_2.y[1]
        z2 = solution_stage_2.y[2]

        # Coast (Engine Shutoff) position
        x3 = solution_coast.y[0]
        y3 = solution_coast.y[1]
        z3 = solution_coast.y[2]

        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Earth sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x_earth = R_EARTH * np.outer(np.cos(u), np.sin(v))
        y_earth = R_EARTH * np.outer(np.sin(u), np.sin(v))
        z_earth = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot Earth
        ax.plot_surface(x_earth, y_earth, z_earth, color="blue", alpha=0.5, linewidth=0)

        # Stage 1 trajectory - red
        ax.plot(x1, y1, z1, color="red", linewidth=3, label="Stage 1")

        # Stage 2 trajectory - orange
        ax.plot(x2, y2, z2, color="orange", linewidth=3, label="Stage 2")

        # Stage Coast trajectory - green
        ax.plot(x3, y3, z3, color="green", linewidth=3, label="Coast Stage")

        # Make axes equal
        max_distance = (
            max(
                R_EARTH,
                np.max(np.abs(x1)),
                np.max(np.abs(y1)),
                np.max(np.abs(z1)),
                np.max(np.abs(x2)),
                np.max(np.abs(y2)),
                np.max(np.abs(z2)),
                np.max(np.abs(x3)),
                np.max(np.abs(y3)),
                np.max(np.abs(z3)),
            )
            * 1.05
        )

        ax.set_xlim(-max_distance, max_distance)
        ax.set_ylim(-max_distance, max_distance)
        ax.set_zlim(-max_distance, max_distance)

        ax.set_box_aspect((1, 1, 1))

        ax.set_xlabel("ECI X (m)")
        ax.set_ylabel("ECI Y (m)")
        ax.set_zlabel("ECI Z (m)")
        ax.set_title("Optimized Rocket Trajectory and One Orbit")

        ax.legend()

        plt.show()

    def plot_speed_and_acceleration(self):
        # Time arrays for stage 1, stage 2, and coast.
        t1, t2, t3 = self.diagnostics.get_times()

        # Speed arrays for stage 1, stage 2, and coast.
        speed1, speed2, speed3 = self.diagnostics.get_speed()

        # Acceleration-magnitude arrays for stage 1, stage 2, and coast.
        acceleration1, acceleration2, acceleration3 = (
            self.diagnostics.compute_acceleration_magnitude()
        )

        fig, ax_speed = plt.subplots(figsize=(10, 6))

        # A second vertical axis lets speed and acceleration use separate units.
        ax_acceleration = ax_speed.twinx()

        # Solid lines = speed.
        ax_speed.plot(t1, speed1, color="red", label="Stage 1 speed")
        ax_speed.plot(t2, speed2, color="orange", label="Stage 2 speed")
        ax_speed.plot(t3, speed3, color="green", label="Coast speed")

        # Dashed lines = acceleration magnitude.
        ax_acceleration.plot(
            t1,
            acceleration1,
            color="red",
            linestyle="--",
            label="Stage 1 acceleration",
        )
        ax_acceleration.plot(
            t2,
            acceleration2,
            color="orange",
            linestyle="--",
            label="Stage 2 acceleration",
        )
        ax_acceleration.plot(
            t3,
            acceleration3,
            color="green",
            linestyle="--",
            label="Coast acceleration",
        )

        ax_speed.set_xlabel("Time (s)")
        ax_speed.set_ylabel("Speed (m/s)")
        ax_acceleration.set_ylabel("Acceleration magnitude (m/s²)")
        ax_speed.set_title("Speed and Acceleration Magnitude")
        ax_speed.grid()

        # Combine legend entries from both vertical axes.
        lines_1, labels_1 = ax_speed.get_legend_handles_labels()
        lines_2, labels_2 = ax_acceleration.get_legend_handles_labels()
        ax_speed.legend(lines_1 + lines_2, labels_1 + labels_2)

        plt.show()

    def plot_fuel_masses(self):
        t1, t2, t3 = self.diagnostics.get_times()

        # Each variable below is a tuple:
        # (stage-1 array, stage-2 array, coast array)
        total_fuel, stage_1_fuel, stage_2_fuel = self.diagnostics.get_fuel_masses()

        # Join all three phases into one continuous line per fuel type.
        time = np.concatenate((t1, t2, t3))
        total = np.concatenate(total_fuel)
        fuel_1 = np.concatenate(stage_1_fuel)
        fuel_2 = np.concatenate(stage_2_fuel)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(time, total, color="black", linewidth=2, label="Total fuel")
        ax.plot(time, fuel_1, color="red", label="Stage 1 fuel")
        ax.plot(time, fuel_2, color="orange", label="Stage 2 fuel")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Fuel mass (kg)")
        ax.set_title("Remaining Propellant Mass")
        ax.grid()
        ax.legend()

        plt.show()

    def plot_dynamic_pressure_vs_time(self):
        # One time and q array for each simulation phase.
        t1, t2, t3 = self.diagnostics.get_times()
        q1, q2, q3 = self.diagnostics.get_q()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Divide by 1000 so the graph uses kPa rather than Pa.
        ax.plot(t1, q1 / 1000, color="red", label="Stage 1")
        ax.plot(t2, q2 / 1000, color="orange", label="Stage 2")
        ax.plot(t3, q3 / 1000, color="green", label="Coast")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Dynamic pressure (kPa)")
        ax.set_title("Dynamic Pressure")
        ax.grid()
        ax.legend()

        plt.show()

    def plot_dynamic_pressure_vs_altitude(self):
        # One time and q array for each simulation phase.
        altitude_1, altitude_2, altitude_3 = self.diagnostics.get_altitude()
        q1, q2, q3 = self.diagnostics.get_q()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Divide by 1000 so the graph uses kPa rather than Pa.
        ax.plot(altitude_1 / 1000, q1 / 1000, color="red", label="Stage 1")
        ax.plot(altitude_2 / 1000, q2 / 1000, color="orange", label="Stage 2")
        ax.plot(altitude_3 / 1000, q3 / 1000, color="green", label="Coast")

        ax.set_xlabel("Altitude (km)")
        ax.set_ylabel("Dynamic pressure (kPa)")
        ax.set_title("Dynamic Pressure vs Altitude")
        ax.grid()
        ax.legend()

        plt.show()

    def plot_dynamic_pressure_vs_times(self):
        # One time and q array for each simulation phase.
        time_1, time_2, time_3 = self.diagnostics.get_times()
        q1, q2, q3 = self.diagnostics.get_q()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Divide by 1000 so the graph uses kPa rather than Pa.
        ax.plot(time_1, q1 / 1000, color="red", label="Stage 1")
        ax.plot(time_2, q2 / 1000, color="orange", label="Stage 2")
        ax.plot(time_3, q3 / 1000, color="green", label="Coast")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Dynamic pressure (kPa)")
        ax.set_title("Dynamic Pressure vs Time")
        ax.grid()
        ax.legend()

        plt.show()

    def plot_mach_numbers_vs_time(self):
        # One time and q array for each simulation phase.
        t1, t2, t3 = self.diagnostics.get_times()
        mach1, mach2, mach3 = self.diagnostics.get_mach()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(t1, mach1, color="red", label="Stage 1")
        ax.plot(t2, mach2, color="orange", label="Stage 2")
        ax.plot(t3, mach3, color="green", label="Coast")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mach Number")
        ax.set_title("Mach Number vs. Time")
        ax.grid()
        ax.legend()

        plt.show()

    def plot_mach_numbers_vs_altitude(self):
        # One time and q array for each simulation phase.
        altitude1, altitude2, altitude3 = self.diagnostics.get_altitude()
        mach1, mach2, mach3 = self.diagnostics.get_mach()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(altitude1/1000, mach1, color="red", label="Stage 1")
        ax.plot(altitude2/1000, mach2, color="orange", label="Stage 2")
        ax.plot(altitude3/1000, mach3, color="green", label="Coast")

        ax.set_xlabel("Altitude (km)")
        ax.set_ylabel("Mach Number")
        ax.set_title("Mach Number vs. Altitude")
        ax.grid()
        ax.legend()

        plt.show()

    def plot_reynolds_numbers_vs_time(self):
        # One time and q array for each simulation phase.
        t1, t2, t3 = self.diagnostics.get_times()
        reynolds1, reynolds2, reynolds3 = self.diagnostics.get_reynolds()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(t1, reynolds1, color="red", label="Stage 1")
        ax.plot(t2, reynolds2, color="orange", label="Stage 2")
        ax.plot(t3, reynolds3, color="green", label="Coast")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Reynolds Number")
        ax.set_title("Reynolds Number vs. Time")
        ax.grid()
        ax.legend()

        plt.show()
    def plot_reynolds_numbers_vs_altitude(self):
        # One time and q array for each simulation phase.
        altitude1, altitude2, altitude3 = self.diagnostics.get_altitude()
        reynolds1, reynolds2, reynolds3 = self.diagnostics.get_reynolds()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(altitude1, reynolds1, color="red", label="Stage 1")
        ax.plot(altitude2, reynolds2, color="orange", label="Stage 2")
        ax.plot(altitude3, reynolds3, color="green", label="Coast")

        ax.set_xlabel("Altitude (s)")
        ax.set_ylabel("Reynolds Number")
        ax.set_title("Reynolds Number vs. Altitude")
        ax.grid()
        ax.legend()

        plt.show()
