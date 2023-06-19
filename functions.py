# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as spec
from tqdm import tqdm

class VortexPanelGeometry():
    def __init__(self, chord, N, alpha, rotation_point, flap_chord = 0, beta = 0):
        self.chord = chord
        self.N = N
        self.alpha = - alpha
        self.rotation_point = rotation_point 
        self.flap_chord = flap_chord
        self.beta = beta

        self.CreatePanels()
        self.CreateVectors()
        self.CalculateLength()

    def CreatePanels(self):
        # flat plate discretization into panels
        N = self.N
        chord = self.chord
        rotation_point = self.rotation_point
        alpha = self.alpha
        flap_chord = self.flap_chord

        self.start = np.zeros((N, 2))
        self.end = np.zeros((N, 2))
        self.controlpoint = np.zeros((N, 2))
        self.vortex = np.zeros((N, 2))
        self.coords = np.zeros((N + 1, 2))

        if flap_chord == 0:
            nodes = np.linspace(0, chord, N + 1)
        else:
            N_flap = int(round(N * flap_chord))             
            nodes = np.concatenate((np.linspace(0, chord * (1 - flap_chord), N - N_flap + 1),    # Main element
                                    np.linspace(chord * (1 - flap_chord), chord, N_flap + 1)[1:] )) # Flap

        self.coords[:, 0] = nodes - rotation_point

        # rotate flap
        if flap_chord != 0:
            cos_beta = np.cos(np.radians(- self.beta))
            sin_beta = np.sin(np.radians(- self.beta))
            for i in range(len(nodes) - N_flap, len(nodes)):
                self.coords[i, 0] = (1 - rotation_point - flap_chord) * chord + cos_beta * (self.coords[i, 0] - (1 - rotation_point - flap_chord) * chord)
                self.coords[i, 1] = sin_beta * (self.coords[i, 0] - (1 - rotation_point - flap_chord) * chord)

        # rotate nodes with alpha
        sin_alpha = np.sin(np.deg2rad(alpha))
        cos_alpha = np.cos(np.deg2rad(alpha))
        for i in range(len(nodes)):
            self.coords[i, 0] = cos_alpha * (self.coords[i, 0]) - sin_alpha * (self.coords[i, 1])
            self.coords[i, 1] = sin_alpha * (self.coords[i, 0]) + cos_alpha * (self.coords[i, 1])

        for i in range(N):
            x_start = self.coords[i, 0]
            x_end = self.coords[i + 1, 0]
            z_start = self.coords[i, 1]
            z_end = self.coords[i + 1, 1]
            self.start[i, :] = [x_start, z_start]
            self.end[i, :] = [x_end, z_end]
            self.controlpoint[i, :] = [0.25 * x_start + 0.75 * x_end, 0.25 * z_start + 0.75 * z_end]
            self.vortex[i, :] = [0.75 * x_start + 0.25 * x_end, 0.75 * z_start + 0.25 * z_end]
            
    def CreateVectors(self):
        # normal and tangential vectors of each panel
        N = self.N
        start = self.start
        end = self.end

        self.norm = np.zeros((N, 2))
        self.tang = np.zeros((N, 2))

        rotation = np.array([[0, -1], 
                             [1, 0]])

        for i in range(N):
            vector = end[i, :] - start[i, :]

            self.norm[i, :] = (rotation @ vector) / np.linalg.norm(vector)
            self.tang[i, :] = vector / np.linalg.norm(vector)

    def CalculateLength(self):
        # computes length of each panel
        N = self.N
        start = self.start
        end = self.end

        self.length = np.zeros(N)

        for i in range(N):
            self.length[i] = np.linalg.norm(end[i, :] - start[i, :])

    def Plot(self, ax):
        # plot airfoil geometry
        N = self.N
        start = self.start
        end = self.end
        controlpoint = self.controlpoint
        vortex = self.vortex

        for i in range(N):
            ax.plot([start[i, 0], end[i, 0]], [start[i, 1], end[i, 1]], 'k-')

        ax.plot(controlpoint[:, 0], controlpoint[:, 1], 'ro', ms = 1)
        ax.plot(vortex[:, 0], vortex[:, 1], 'go', ms = 1)
        ax.set_aspect('equal')
        ax.grid(True)


class Kinematics():
    def __init__(self, geometry, U_inf, W_inf, t, A_pitch, f_pitch, TE_location, shed_vortex_factor):
        self.t = t
        self.A_pitch = np.radians(A_pitch)
        self.f_pitch = f_pitch
        self.U_inf = U_inf
        self.W_inf = W_inf
        self.TE_location = TE_location
        self.shed_vortex_factor = shed_vortex_factor
        self.geometry = geometry

        # pitch displacement
        self.d_pitch = self.A_pitch * np.sin(2 * np.pi * self.f_pitch * self.t) 
        # pitch velocity
        self.u_pitch = 2 * np.pi * self.f_pitch * self.A_pitch * np.cos(2 * np.pi * self.f_pitch * self.t) 
        # translation displacement
        self.d_trans = np.array([self.U_inf, self.W_inf]) * self.t 
        # translation velocity
        self.u_trans = np.array([self.U_inf, self.W_inf]) 

        cos_d_pitch = np.cos(self.d_pitch)
        sin_d_pitch = np.sin(self.d_pitch)
        # transformation matrix inertial to moving frame of reference
        self.tran_Mat_B = np.array([[cos_d_pitch, - sin_d_pitch],
                                    [sin_d_pitch, cos_d_pitch]])
        # transformation matrix moving to inertial frame of reference
        self.tran_Mat_E   = np.array([[cos_d_pitch, sin_d_pitch],
                                      [- sin_d_pitch, cos_d_pitch]])

        self.CreatePanels()
        self.PanelVelocity()
        self.ShedVortexLocation()

    def CreatePanels(self):
        # creates panels in inertial reference of frame
        N = self.geometry.N

        d_trans = self.d_trans

        self.start_E = np.zeros((N, 2))
        self.end_E = np.zeros((N, 2))
        self.vortex_E = np.zeros((N, 2))
        self.controlpoint_E = np.zeros((N, 2))

        for i in range(N):
            self.start_E[i,:] = d_trans + self.tran_Mat_E @ self.geometry.start[i,:]
            self.end_E[i,:] = d_trans + self.tran_Mat_E @ self.geometry.end[i,:]
            self.vortex_E[i,:] = d_trans + self.tran_Mat_E @ self.geometry.vortex[i,:]
            self.controlpoint_E[i,:] = d_trans + self.tran_Mat_E @ self.geometry.controlpoint[i,:]

    def PanelVelocity(self):
        # computes velocity in control points for both reference of frames
        N = self.geometry.N

        d_trans = self.d_trans
        u_trans = self.u_trans
        u_pitch = self.u_pitch

        self.vel_E = np.zeros((N, 2)) # w.r.t. inertial reference frame
        self.vel_B = np.zeros((N, 2)) # w.r.t. moving reference frame

        for i in range(N):
            r = self.controlpoint_E[i] - d_trans
            self.vel_E[i, :] = -(u_trans + u_pitch * np.array([r[1], - r[0]]))
            self.vel_B[i, :] = self.tran_Mat_B @ self.vel_E[i]

    def ShedVortexLocation(self):
        # computes shed vortex coordinates in inertial reference of frame
        chord = self.geometry.chord

        shed_vortex_factor = self.shed_vortex_factor
        TE_location = self.TE_location
        end_E = self.end_E

        self.shed_vortex = np.zeros(2)

        if shed_vortex_factor == -1: # steady state, shed vortex at downwind infinity
            self.shed_vortex[0] = - 1.e7 * chord
        else: # unsteady, calculate with trailing edge location
            self.shed_vortex[:] = (shed_vortex_factor * (np.abs(TE_location - end_E[-1]))) + end_E[-1]


class SystemSolution():
    def __init__(self, geometry, kinematics, iteration, dt, previous_prop, wake):
        self.geometry = geometry
        self.kinematics = kinematics
        self.iteration = iteration
        self.dt = dt
        self.previous_prop = previous_prop
        self.wake = wake

        self.GetInductionMatrix()
        self.RHS()
        self.Solve()

    def InducedVelocity(self, controlpoint, vortex):
        # computes induced velocity due to a lumped vortex element with unitary circulation
        r = controlpoint - vortex
        r_norm = np.linalg.norm(r)
        scale = 1 / (2 * np.pi * r_norm ** 2)

        return scale * (np.array([[0, 1], [-1, 0]]) @ r)
    
    def GetInductionMatrix(self):
        # computes induction matrix of bound and shed vortices
        N = self.geometry.N
        controlpoint = self.geometry.controlpoint
        vortex = self.geometry.vortex
        normal = self.geometry.norm

        controlpoint_E = self.kinematics.controlpoint_E
        shed_vortex = self.kinematics.shed_vortex
        tran_Mat_B = self.kinematics.tran_Mat_B

        self.InductionMatrix = np.zeros((N + 1, N + 1))

        # influence of bound vortices with known location
        for i in range(N):
            for j in range(N):
                V_induced = self.InducedVelocity(controlpoint[i, :], vortex[j, :])
                self.InductionMatrix[i, j] = np.dot(V_induced, normal[i, :])

        # influence of shed vortices with unknown location
        for i in range(N):
            V_induced_E = self.InducedVelocity(controlpoint_E[i, :], shed_vortex) # inertial reference of frame
            V_induced_B = tran_Mat_B @ V_induced_E
            self.InductionMatrix[i, -1] = np.dot(V_induced_B, normal[i, :])

        self.InductionMatrix[-1, :] = 1 # kelvin's theorem

    def RHS(self):
        # computes right-hand side vector with boundary conditions
        N = self.geometry.N
        normal = self.geometry.norm

        vel_B = self.kinematics.vel_B
        controlpoint_E = self.kinematics.controlpoint_E
        tran_Mat_B = self.kinematics.tran_Mat_B

        iteration = self.iteration

        self.rhs = np.zeros(N + 1)

        if iteration == 1: 
            for i in range(N):
                self.rhs[i] = - np.dot(vel_B[i, :], normal[i, :])
        else:
            V_induced_shed_B = np.zeros((N, 2))
            for i in range(N):
                for j in range(1, iteration): # loops over all shed vortices from previous time steps
                    circ = self.wake[j].circulation
                    lctn = self.wake[j].location
                    
                    V_induced_shed_E = (circ * self.InducedVelocity(controlpoint_E[i, :], lctn))
                    V_induced_shed_B[i, :] += tran_Mat_B @ V_induced_shed_E

                self.rhs[i] = - np.dot(vel_B[i] + V_induced_shed_B[i], normal[i])

            self.rhs[-1] = self.previous_prop.total_circulation

    def Solve(self):
        # solves linear system
        self.circulation = np.linalg.solve(self.InductionMatrix, self.rhs)

    def FlowField(self, x_lim, z_lim, Nx, Nz, U_inf, W_inf):
        # computes velocity and pressure in flowfield
        print('Computing Flowfield')
        N = self.geometry.N
        vortex = self.geometry.vortex

        tran__Mat_B = self.kinematics.tran_Mat_B
        d_trans = self.kinematics.d_trans

        circulation = self.circulation
        iteration = self.iteration
        wake = self.wake

        x = np.linspace(x_lim[0], x_lim[1], Nx + 1)
        z = np.linspace(z_lim[0], z_lim[1], Nz + 1)
        X, Z = np.meshgrid(x, z, indexing = 'ij') # moving reference of frame

        rotation = np.array([[0 , -1],
                             [1,  0]])

        U = U_inf * np.ones((Nx, Nz))
        W = W_inf * np.ones((Nx, Nz))
        Cp = np.zeros((Nx, Nz))

        U_mag = np.linalg.norm([U_inf, W_inf])

        for i in tqdm(range(Nx)):
            for j in range(Nz):
                for k in range(N): # bound vortices influence
                    r = np.array([(X[i, j] + X[i + 1, j]) / 2, (Z[i, j] + Z[i, j + 1]) / 2]) - vortex[k, :]
                    V_mag = -circulation[k] / (2 * np.pi * np.linalg.norm(r))
                    normal = (rotation @ r) / np.linalg.norm(r)

                    U[i, j] -= normal[0] * V_mag
                    W[i, j] += normal[1] * V_mag
                if type(self.wake) != int: # shed vortices influence
                    for k in range(1, iteration):
                        wake_location_B = tran__Mat_B @ (wake[k].location - d_trans) 
                        r = np.array([(X[i, j] + X[i + 1, j]) / 2, (Z[i, j] + Z[i, j + 1]) / 2]) -  wake_location_B
                        V_mag = -wake[k].circulation / (2 * np.pi * np.linalg.norm(r))
                        normal = (rotation @ r) / np.linalg.norm(r)

                        U[i, j] -= normal[0] * V_mag
                        W[i, j] += normal[1] * V_mag
                Cp[i, j] = 1 - (np.linalg.norm(U[i, j]) / U_mag) ** 2

        return X, Z, U, W, Cp


class WakeProperties():
    def __init__(self, solution, kinematics, geometry, wake, dt, iteration):
        self.solution = solution
        self.kinematics = kinematics
        self.geometry = geometry
        self.wake = wake
        self.dt = dt
        self.iteration = iteration

        self.circulation = self.solution.circulation[-1]
        # location in inertial frame of reference
        self.location = self.kinematics.shed_vortex

        # vortex wake rollup
        N = self.geometry.N
        controlpoint_E = self.kinematics.controlpoint_E
        tran_Mat_B = self.kinematics.tran_Mat_B

        V_induced = np.zeros(2)

        for i in range(N):
            V_induced_E = self.solution.InducedVelocity(self.location, controlpoint_E[i, :])

            V_induced[:] += tran_Mat_B @ V_induced_E

        if iteration != 1: # velocity induced by shed vortices
            for j in range(1, iteration):
                circulation = wake[j].circulation
                location = wake[j].location
                V_induced_wake_E = circulation * self.solution.InducedVelocity(self.location, location)

                V_induced[:] += tran_Mat_B @ V_induced_wake_E

        # update location of vortex
        self.location += V_induced * dt


class SolutionProperties():
    def __init__(self, geometry, kinematics, solution, wake, iteration, previous_prop, rho, U_mag, dt):
        self.geometry = geometry
        self.kinematics = kinematics
        self.solution = solution
        self.wake = wake
        self.iteration = iteration
        if iteration > 1:
            self.previous_prop = previous_prop
        self.rho = rho
        self.U_mag = U_mag
        self.dt = dt

        self.PanelCirculation()
        self.InducedVelocityWake()
        self.PanelPressure()
        self.Forces()

    def PanelCirculation(self):
        # computes panel circulation
        N = self.geometry.N

        self.panel_circulation = self.solution.circulation[0:-1]

        self.total_circulation = np.sum(self.panel_circulation)

    def InducedVelocityWake(self):
        # computes induced velocity by the wake at control points (frame of reference)
        N = self.geometry.N

        shed_vortex = self.kinematics.shed_vortex
        controlpoint_E = self.kinematics.controlpoint_E
        tran_Mat_B = self.kinematics.tran_Mat_B

        iteration = self.iteration

        self.V_induced_wake = np.zeros((N, 2))

        if iteration != 1: # velocity induced by shed vortices at previous time steps
            for i in range(N):
                for j in range(1, iteration):
                    circulation = self.wake[j].circulation
                    location = self.wake[j].location
                    V_induced_wake_E = circulation * self.solution.InducedVelocity(controlpoint_E[i, :], location)

                    self.V_induced_wake[i, :] += tran_Mat_B @ V_induced_wake_E
        
        for i in range(N): # velocity induced by shed vortex at latest time step
            V_induced_wake_E = self.solution.circulation[-1] * self.solution.InducedVelocity(controlpoint_E[i, :], shed_vortex)

            self.V_induced_wake[i, :] += tran_Mat_B @ V_induced_wake_E

    def PanelPressure(self):
        # computes pressure difference at each panel
        N = self.geometry.N
        tangential = self.geometry.tang
        length = self.geometry.length

        vel_B = self.kinematics.vel_B

        panel_circulation = self.panel_circulation
        V_induced_wake = self.V_induced_wake
        iteration = self.iteration
        dt = self.dt
        rho = self.rho

        # potential flow
        self.potential_diff = np.zeros(N)
        for i in range(N):
            self.potential_diff[i] = np.sum(self.panel_circulation[0:i + 1])

        # pressure difference
        self.panel_pressure = np.zeros(N)
        for i in range(N):
            t1 = np.dot(vel_B[i, :] + V_induced_wake[i, :], tangential[i, :]) * panel_circulation[i] / length[i]
            if iteration == 1:
                t2 = 0
            else:
                t2 = (self.potential_diff[i] - self.previous_prop.potential_diff[i]) / dt

            self.panel_pressure[i] = rho * (t1 + t2)

    def Forces(self):
        # computes force per panel and total force
        N = self.geometry.N
        length = self.geometry.length
        normal = self.geometry.norm
        chord = self.geometry.chord

        tran_Mat_E = self.kinematics.tran_Mat_E

        panel_pressure = self.panel_pressure
        rho = self.rho
        U_mag = self.U_mag

        self.panel_force = np.zeros((N, 2))

        for i in range(N):
            force_body = panel_pressure[i] * length[i] * normal[i, :]
            self.panel_force[i, :] = tran_Mat_E @ force_body

        self.total_force = np.array([np.sum(self.panel_force[:, 0]), np.sum(self.panel_force[:, 1])])
        self.Cl = self.total_force[1] / (0.5 * rho * U_mag ** 2 * chord)


def PlotVelocityField(X, Z, U, W, U_mag, geometry, x, z):
    # plot velocity field
    V_mag = np.sqrt(U ** 2 + W ** 2)
    if len(x) == len(z):
        fig, ax = plt.subplots(figsize = (6, 6))
    else:
        fig, ax = plt.subplots(figsize = (12, 6))
    im = ax.pcolormesh(X, Z, V_mag / U_mag, cmap = 'viridis')
    fig.colorbar(im, ax = ax, label=r'$V / V_\infty$')
    # plot airfoil
    geometry.Plot(ax)
    # plot quiver
    X, Z = np.meshgrid(x, z, indexing='ij')
    ax.quiver(X, Z, -U, W)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$z/c$')


def PlotPressureField(X, Z, Cp, geometry):
    # plot pressure field
    fig, ax = plt.subplots(figsize = (6, 6))
    im = ax.pcolormesh(X, Z, Cp, cmap = 'viridis')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$z/c$')
    fig.colorbar(im, ax = ax, label=r'$C_p$')
    # plot airfoil
    geometry.Plot(ax)


def PlotLiftCoefficient(alpha, Cl, multi_variate = False, beta = 0):
    if multi_variate == False:
        fig = plt.figure(figsize = (12, 6))
        ax = fig.add_subplot(111)
        Cl_true = 2 * np.pi * np.deg2rad(alpha)
        ax.plot(alpha, Cl_true, 'r-', lw = 1, label = r'$2\pi\alpha$')
        ax.plot(alpha, Cl, 'k-o', label = r'Obtained Value')
        ax.set_ylabel(r'$C_l$')
    else:
        Alpha, Beta = np.meshgrid(alpha, beta, indexing = 'ij')
        fig = plt.figure(figsize = (6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Alpha, Beta, Cl)
        ax.set_ylabel(r'$\beta$ [degrees]')
        ax.set_zlabel(r'$C_l$')
            
    ax.set_xlabel(r'$\alpha$ [degrees]')
    
    ax.grid()
    ax.legend()


def Theodorsen(kinematics, solutionProperties, time, kappa, chord, rotation_point, f_pitch, A_pitch, pitch, dt, Cl_steady = [], alpha_steady = []):
    # unsteady lift of a harmonically oscillating airfoil
    hankel2_0 = spec.hankel2(0, kappa)
    hankel2_1 = spec.hankel2(1, kappa)
    theodorsen_function = hankel2_1 / (hankel2_1 + (1j * hankel2_0))
    
    a = (rotation_point - (chord / 2)) / (chord / 2)
    theodorsen_coeff = 1j * np.pi * kappa + a * np.pi * kappa ** 2 + 2 * np.pi * theodorsen_function + 2 * np.pi * theodorsen_function * 1j * kappa * (0.5 - a)

    alpha = []
    Cl_unsteady = []
    Cl_theodorsen = []
    alpha_theodorsen = []

    T = 1 / f_pitch

    for i in range(int(- T / dt - 2), 0):
        alpha.append(np.rad2deg(kinematics[i].d_pitch) + pitch)
        Cl_unsteady.append(solutionProperties[i].Cl)
        Cl_theodorsen.append(np.real(2 * np.pi * np.deg2rad(pitch) + theodorsen_coeff * np.deg2rad(A_pitch) * np.exp(1j * f_pitch * 2 * np.pi * time[i])))
        alpha_theodorsen.append(np.rad2deg(np.real(np.deg2rad(pitch) + np.deg2rad(A_pitch) * np.exp(1j * f_pitch * 2 * np.pi * time[i]))))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(alpha_theodorsen, Cl_theodorsen, 'r-', lw = 1, ms = 1, label=r'Theodorsen')
    ax.plot(alpha_steady, Cl_steady, 'b-', lw = 1, ms = 1, label=r'Steady')
    ax.plot(alpha, Cl_unsteady, 'k-o', lw = 1, ms = 1, label=r'Unsteady')
    ax.grid()
    ax.set_xlabel(r'$\alpha$ [deg]')
    ax.set_ylabel(r'$C_l$')
    ax.legend()

    # Split results into upper and lower array
    i0 = np.argmin(alpha_theodorsen)
    i1 = np.argmax(alpha_theodorsen)
    if i0 > i1:
        thaTop = alpha_theodorsen[i1:i0]
        thaBot = np.concatenate((alpha_theodorsen[i0:],alpha_theodorsen[:i1]))
        thcTop = Cl_theodorsen[i1:i0]
        thcBot = np.concatenate((Cl_theodorsen[i0:],Cl_theodorsen[:i1]))
    else:
        thaTop = np.concatenate((alpha_theodorsen[i1:],alpha_theodorsen[:i0]))
        thaBot = alpha_theodorsen[i0:i1]
        thcTop = np.concatenate((Cl_theodorsen[i1:],Cl_theodorsen[:i0]))
        thcBot = Cl_theodorsen[i0:i1]

    i0 = np.argmin(alpha)
    i1 = np.argmax(alpha)
    if i0 > i1:
        unaTop = alpha[i1:i0]
        unaBot = np.concatenate((alpha[i0:],alpha[:i1]))
        uncTop = Cl_unsteady[i1:i0]
        uncBot = np.concatenate((Cl_unsteady[i0:],Cl_unsteady[:i1]))
    else:
        unaTop = np.concatenate((alpha[i1:],alpha[:i0]))
        unaBot = alpha[i0:i1]
        uncTop = np.concatenate((Cl_unsteady[i1:],Cl_unsteady[:i0]))
        uncBot = Cl_unsteady[i0:i1]

    # Make arrays increasing
    if unaTop[0] > unaTop[-1]:
        unaTop = unaTop[::-1]; uncTop = uncTop[::-1]
    if unaBot[0] > unaBot[-1]:
        unaBot = unaBot[::-1]; uncBot = uncBot[::-1]
    if thaTop[0] > thaTop[-1]:
        thaTop = thaTop[::-1]; thcTop = thcTop[::-1]
    if thaBot[0] > thaBot[-1]:
        thaBot = thaBot[::-1]; thcBot = thcBot[::-1]
    
    # Calculate difference between CL and theodorssen
    RMSE = 0
    for i in range(len(unaTop)):
        RMSE += (uncTop[i] - np.interp(unaTop[i], thaTop, thcTop))**2
    for i in range(len(unaBot)): RMSE += (uncBot[i] - np.interp(unaBot[i], thaBot, thcBot))**2
    RMSE = np.sqrt(RMSE/(len(unaTop)+len(unaBot)))
    return RMSE


def SensitivityNumberOfElements(A_pitch, f_pitch, pitch_0, chord, rotation_point, U_inf, rho, shed_vortex_factor, kappa):
    # effect of value of N for unsteady state flow 
    dt = 0.01   
    Nt = 700
    T = dt * (Nt - 1)
    time = np.linspace(0, T, Nt)

    N_distribution = np.array([1, 3, 5, 7, 10, 15, 20])

    CLdiff = np.zeros(len(N_distribution))

    for j in range(len(N_distribution)):
        print('N = ' + str(N_distribution[j]))
        arr_kinematics = np.empty(Nt, dtype='object')
        arr_properties = np.empty(Nt, dtype='object')
        arr_wake = np.empty(Nt, dtype='object')

        airfoil = VortexPanelGeometry(chord, N_distribution[j], pitch_0, rotation_point)
        arr_kinematics[0] = Kinematics(airfoil, U_inf, 0, 0, A_pitch, f_pitch, airfoil.end[-1], shed_vortex_factor)
    
        for i in tqdm(range(1, Nt)):
            arr_kinematics[i] = Kinematics(airfoil, U_inf, 0, time[i], A_pitch, f_pitch, arr_kinematics[i - 1].end_E[-1], shed_vortex_factor)

            solution = SystemSolution(airfoil, arr_kinematics[i], i, dt, arr_properties[i - 1], arr_wake[0:i])
            arr_properties[i] = SolutionProperties(airfoil, arr_kinematics[i], solution, arr_wake[0:i], i, arr_properties[i-1], rho, -U_inf, dt)

            arr_wake[i] = WakeProperties(solution, arr_kinematics[i], airfoil, arr_wake[0:i], dt, i)

        CLdiff[j] = Theodorsen(arr_kinematics, arr_properties, time, kappa, chord, rotation_point, f_pitch, A_pitch, pitch_0, dt)

    # plot result
    fig, ax = plt.subplots(figsize = (12, 6))
    ax.plot(N_distribution, CLdiff, 'k-o', ms = 1, lw = 1)
    ax.set_xlabel(r'$N_{panels}$')
    ax.set_ylabel(r'RMSE($C_l$)')
    ax.grid()


def SensitivityTimeStep(A_pitch, f_pitch, pitch_0, chord, rotation_point, U_inf, rho, shed_vortex_factor, kappa):
    # effect of value of dt for unsteady state flow 
    dt_distribution = [0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 1.0]
    N = 20
    Nt = 700

    CLdiff = np.zeros(len(dt_distribution))

    for j in range(len(dt_distribution)):
        dt = dt_distribution[j]
        print('dt =' + str(dt))

        T = dt * (Nt - 1)                           
        time = np.linspace(0, T, Nt)

        arr_kinematics = np.empty(Nt, dtype='object')
        arr_properties = np.empty(Nt, dtype='object')
        arr_wake = np.empty(Nt, dtype='object')

        airfoil = VortexPanelGeometry(chord, N, pitch_0, rotation_point)
        arr_kinematics[0] = Kinematics(airfoil, U_inf, 0, 0, A_pitch, f_pitch, airfoil.end[-1], shed_vortex_factor)
    
        for i in tqdm(range(1, Nt)):
            arr_kinematics[i] = Kinematics(airfoil, U_inf, 0, time[i], A_pitch, f_pitch, arr_kinematics[i - 1].end_E[-1], shed_vortex_factor)

            solution = SystemSolution(airfoil, arr_kinematics[i], i, dt, arr_properties[i - 1], arr_wake[0:i])
            arr_properties[i] = SolutionProperties(airfoil, arr_kinematics[i], solution, arr_wake[0:i], i, arr_properties[i-1], rho, -U_inf, dt)

            arr_wake[i] = WakeProperties(solution, arr_kinematics[i], airfoil, arr_wake[0:i], dt, i)

        CLdiff[j] = Theodorsen(arr_kinematics, arr_properties, time, kappa, chord, rotation_point, f_pitch, A_pitch, pitch_0, dt)

    # Plot result
    fig, ax = plt.subplots(figsize = (12, 6))
    ax.plot(dt_distribution, CLdiff, 'k-o', ms = 1, lw = 1)
    ax.set_xlabel(r'$\Delta t [s]$')
    ax.set_ylabel(r'RMSE($C_l$)')
    ax.grid()

