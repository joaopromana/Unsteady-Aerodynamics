# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import functions as fcn
from tqdm import tqdm

# program parameters
rho = 1.0                       # density [kg/m^3]
U_inf = -50                     # x-component of inertial velocity [m/s]
chord = 1                       # airfoil chord [m]
N = 20                          # number of vortex elements
pitch_0 = 5                     # pitch in steady state (AoA) [degrees]                 
rotation_point = 0.5 * chord   # rotation point of airfoil
Nx = 100                        # number of elements in x-direction                     (UNSTEADY)
Nz = 40                         # number of elements in z-direction                     (UNSTEADY)
Nt = 350                        # number of time steps                                  (UNSTEADY)
dt = 0.01                       # time step [sec]                                       (UNSTEADY)
A_pitch = 5                     # pitch amplitude (sinusoidal pitching) [degrees]       (UNSTEADY)
kappa = 0.02                    # reduced frequency                                     (UNSTEADY)
shed_vortex_factor = 0.25       # between 0.2 and 0.3                                   (UNSTEADY)

T = dt * (Nt - 1)
time = np.linspace(0, T, Nt)
f_pitch = - 2 * U_inf * kappa / (2 * np.pi * chord)

# Present options to user
print('1 - Steady Airfoil at Different Angles of Attack')
print('2 - Sensitivity Study with respect to Number of Elements')
print('3 - Pitching Airfoil at Different Reduced Frequencies')
print('4 - Sensitivity Study with respect to Time Step')
print('5 - Steady Airfoil with Flap at Different Amplitudes')
print('6 - Pitching Airfoil with Flap at Different Reduced Frequencies')

choice = input('Select an option: ')

if choice == '1':
    print()

    x_lim = [-1.5 * chord, 1.5 * chord]
    z_lim = [-1.5 * chord, 1.5 * chord]
    Nx = 200
    Nz = 200
    alpha_range = [-10, 25]
    N_alpha = 40
    
    alpha = np.linspace(alpha_range[0], alpha_range[1], N_alpha)
 
    Cl = np.zeros((N_alpha))

    print('Started Simulation')

    for i in tqdm(range(N_alpha)):
        airfoil = fcn.VortexPanelGeometry(chord, N, alpha[i], rotation_point)
        kinematics = fcn.Kinematics(airfoil, U_inf, 0, 0, 0, 0, airfoil.end[-1], -1)
        solution = fcn.SystemSolution(airfoil, kinematics, 1, 1, 0, 0)
        solution_properties = fcn.SolutionProperties(airfoil, kinematics, solution, 0, 1, 0, rho, -U_inf, 1)
        Cl[i] = solution_properties.Cl

    airfoil = fcn.VortexPanelGeometry(chord, N, pitch_0, rotation_point)
    kinematics = fcn.Kinematics(airfoil, U_inf, 0, 0, 0, 0, airfoil.end[-1], -1)
    solution = fcn.SystemSolution(airfoil, kinematics, 1, 1, 0, 0)
    X, Z, U, W, Cp = solution.FlowField(x_lim, z_lim, Nx, Nz, U_inf, 0)

    print("Finished Simulation")

    fcn.PlotPressureField(X, Z, Cp, airfoil)
    fcn.PlotVelocityField(X, Z, U, W, -U_inf, airfoil, np.linspace(x_lim[0], x_lim[1], Nx), np.linspace(z_lim[0], z_lim[1], Nz))

    plt.show()

    fcn.PlotLiftCoefficient(alpha, Cl)


elif choice == '2':
    print()

    fcn.SensitivityNumberOfElements(chord, rotation_point, U_inf, rho)


elif choice == '3':
    print()

    x_lim = [-1.5 * chord, 4.5 * chord]
    z_lim = [-1.5 * chord, 1.5 * chord]

    # storage variables
    arr_kinematics = np.empty(Nt, dtype = 'object')
    arr_properties = np.empty(Nt, dtype = 'object')
    arr_wake = np.empty(Nt, dtype = 'object')

    print('Started simulation with kappa = %0.2f' % kappa)

    # Create initial geometry (t=0)
    airfoil = fcn.VortexPanelGeometry(chord, N, pitch_0, rotation_point)
    arr_kinematics[0] = fcn.Kinematics(airfoil, U_inf, 0, 0, A_pitch, f_pitch, airfoil.end[-1], shed_vortex_factor)

    for i in tqdm(range(1, Nt)):
        arr_kinematics[i] = fcn.Kinematics(airfoil, U_inf, 0, time[i], A_pitch, f_pitch, arr_kinematics[i - 1].end_E[-1], shed_vortex_factor)
  
        solution = fcn.SystemSolution(airfoil, arr_kinematics[i], i, dt, arr_properties[i - 1], arr_wake[0:i])
        arr_properties[i] = fcn.SolutionProperties(airfoil, arr_kinematics[i], solution, arr_wake[0:i], i, arr_properties[i-1], rho, -U_inf, dt)
    
        arr_wake[i] = fcn.WakeProperties(solution, arr_kinematics[i])

    # create final geometry (t=T)
    airfoil = fcn.VortexPanelGeometry(chord, N, pitch_0 + np.rad2deg(arr_kinematics[-1].d_pitch), rotation_point)
    X, Z, U, W, Cp = solution.FlowField(x_lim, z_lim, Nx, Nz, U_inf, 0)

    print('Finished Simulation')

    fcn.PlotVelocityField(X, Z, U, W, -U_inf, airfoil, np.linspace(x_lim[0], x_lim[1], Nx), np.linspace(z_lim[0], z_lim[1], Nz))

    plt.show()

    fcn.Theodorsen(arr_kinematics, arr_properties, time, kappa, chord, rotation_point, f_pitch, A_pitch, pitch_0, dt)


elif choice == '4':
    print()

    fcn.SensitivityTimeStep(A_pitch, f_pitch, pitch_0, chord, rotation_point, U_inf, rho, shed_vortex_factor, kappa)


elif choice == '5':
    print()

    x_lim = [-1.5 * chord, 1.5 * chord]
    z_lim = [-1.5 * chord, 1.5 * chord]
    Nx = 200
    Nz = 200
    alpha_range = [-10, 25]
    N_alpha = 40
    
    alpha = np.linspace(alpha_range[0], alpha_range[1], N_alpha)

    flap_chord = 0.25           
    beta_range = [0, 15]
    N_beta = 4

    beta = np.linspace(beta_range[0], beta_range[1], N_beta)
 
    Cl = np.zeros((N_alpha, N_beta))

    print('Started Simulation')

    for j in tqdm(range(N_beta)):
        for i in range(N_alpha):
            airfoil = fcn.VortexPanelGeometry(chord, N, alpha[i], rotation_point, flap_chord, beta[j])
            kinematics = fcn.Kinematics(airfoil, U_inf, 0, 0, 0, 0, airfoil.end[-1], -1)
            solution = fcn.SystemSolution(airfoil, kinematics, 1, 1, 0, 0)
            solution_properties = fcn.SolutionProperties(airfoil, kinematics, solution, 0, 1, 0, rho, -U_inf, 1)
            Cl[i, j] = solution_properties.Cl

    print("Finished Simulation")

    airfoil = fcn.VortexPanelGeometry(chord, N, pitch_0, rotation_point, flap_chord, beta[-1])
    kinematics = fcn.Kinematics(airfoil, U_inf, 0, 0, 0, 0, airfoil.end[-1], -1)
    solution = fcn.SystemSolution(airfoil, kinematics, 1, 1, 0, 0)
    solution_properties = fcn.SolutionProperties(airfoil, kinematics, solution, 0, 1, 0, rho, -U_inf, 1)
    X, Z, U, W, Cp = solution.FlowField(x_lim, z_lim, Nx, Nz, U_inf, 0)

    print("Finished Simulation")

    fcn.PlotVelocityField(X, Z, U, W, -U_inf, airfoil, np.linspace(x_lim[0], x_lim[1], Nx), np.linspace(z_lim[0], z_lim[1], Nz))

    plt.show()
    
    fcn.PlotLiftCoefficient(alpha, Cl[:, -1])

    plt.show()

    fcn.PlotLiftCoefficient(alpha, Cl, True, beta)


elif choice == '6':
    print()

    flap_chord = 0.25  
    beta_0 = 5

    x_lim = [-1.5 * chord, 4.5 * chord]
    z_lim = [-1.5 * chord, 1.5 * chord]

    # storage variables
    arr_kinematics = np.empty(Nt, dtype = 'object')
    arr_properties = np.empty(Nt, dtype = 'object')
    arr_wake = np.empty(Nt, dtype = 'object')

    print('Started simulation with kappa = %0.2f' % kappa)

    # Create initial geometry (t=0)
    airfoil = fcn.VortexPanelGeometry(chord, N, pitch_0, rotation_point, flap_chord, beta_0)
    arr_kinematics[0] = fcn.Kinematics(airfoil, U_inf, 0, 0, A_pitch, f_pitch, airfoil.end[-1], shed_vortex_factor)

    for i in tqdm(range(1, Nt)):
        arr_kinematics[i] = fcn.Kinematics(airfoil, U_inf, 0, time[i], A_pitch, f_pitch, arr_kinematics[i - 1].end_E[-1], shed_vortex_factor)
  
        solution = fcn.SystemSolution(airfoil, arr_kinematics[i], i, dt, arr_properties[i - 1], arr_wake[0:i])
        arr_properties[i] = fcn.SolutionProperties(airfoil, arr_kinematics[i], solution, arr_wake[0:i], i, arr_properties[i-1], rho, -U_inf, dt)
    
        arr_wake[i] = fcn.WakeProperties(solution, arr_kinematics[i])

    # create final geometry (t=T)
    airfoil = fcn.VortexPanelGeometry(chord, N, pitch_0 + np.rad2deg(arr_kinematics[-1].d_pitch), rotation_point, flap_chord, beta_0 + np.rad2deg(arr_kinematics[-1].d_pitch))
    X, Z, U, W, Cp = solution.FlowField(x_lim, z_lim, Nx, Nz, U_inf, 0)

    print('Finished Simulation')
    print()
    fcn.PlotVelocityField(X, Z, U, W, -U_inf, airfoil, np.linspace(x_lim[0], x_lim[1], Nx), np.linspace(z_lim[0], z_lim[1], Nz))

    plt.show()

    fcn.Theodorsen(arr_kinematics, arr_properties, time, kappa, chord, rotation_point, f_pitch, A_pitch, pitch_0, dt)


plt.show()

