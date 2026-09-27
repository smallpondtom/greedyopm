"""SU2 configuration for the laminar unsteady NACA 0012 case at Re_c = 1000.

Units are convective: chord c = 1, free-stream speed U_inf = 1, density 1,
so mu = 1/Re and time is in c/U_inf.  The angle of attack enters only through
the free-stream direction; the mesh is the same for every angle.
"""
from __future__ import annotations

import math


def make_config(mesh: str, alpha_deg: float, *, Re: float = 1000.0,
                dt: float = 0.02, t_end: float = 60.0, inner_iter: int = 30,
                cfl: float = 10.0, write_every: int = 10, inner_conv: float = -7.0,
                restart: bool = False, restart_iter: int = 0) -> str:
    a = math.radians(alpha_deg)
    ux, uy = math.cos(a), math.sin(a)
    n_steps = int(round(t_end / dt))
    return f"""% Laminar unsteady NACA 0012, Re_c = {Re:g}, alpha = {alpha_deg:g} deg
SOLVER= INC_NAVIER_STOKES
KIND_TURB_MODEL= NONE
MATH_PROBLEM= DIRECT
RESTART_SOL= {"YES" if restart else "NO"}
RESTART_ITER= {restart_iter}

% --- unsteady ---
TIME_DOMAIN= YES
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= {dt:.10g}
MAX_TIME= {t_end + dt:.10g}
TIME_ITER= {n_steps}
INNER_ITER= {inner_iter}

% --- free stream (convective units) ---
INC_DENSITY_INIT= 1.0
INC_VELOCITY_INIT= ( {ux:.16e}, {uy:.16e}, 0.0 )
INC_NONDIM= INITIAL_VALUES
AOA= {alpha_deg:.10g}
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= {1.0 / Re:.16e}

REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= 1.0
REF_AREA= 1.0

% --- boundaries ---
MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )

% --- numerics ---
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= {cfl:g}
CFL_ADAPT= NO
MAX_DELTA_TIME= 1E6
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ILU_FILL_IN= 0
LINEAR_SOLVER_ERROR= 1E-8
LINEAR_SOLVER_ITER= 20
MGLEVEL= 0
CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= NONE
TIME_DISCRE_FLOW= EULER_IMPLICIT

% --- inner-iteration convergence ---
CONV_FIELD= RMS_PRESSURE
CONV_RESIDUAL_MINVAL= {inner_conv:g}
CONV_STARTITER= 3
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1E-6
WINDOW_CAUCHY_CRIT= NO

% --- input / output ---
MESH_FILENAME= {mesh}
MESH_FORMAT= SU2
TABULAR_FORMAT= CSV
CONV_FILENAME= history
RESTART_FILENAME= restart_flow
SOLUTION_FILENAME= restart_flow
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
VOLUME_OUTPUT= ( COORDINATES, SOLUTION )
OUTPUT_FILES= ( RESTART, CSV )
OUTPUT_WRT_FREQ= {write_every}
WRT_VOLUME_OVERWRITE= YES
WRT_SURFACE_OVERWRITE= YES
SCREEN_OUTPUT= ( TIME_ITER, INNER_ITER, CUR_TIME, RMS_PRESSURE, RMS_VELOCITY-X, LIFT, DRAG, WALL_TIME )
SCREEN_WRT_FREQ_INNER= 10
SCREEN_WRT_FREQ_TIME= 1
HISTORY_OUTPUT= ( TIME_ITER, INNER_ITER, CUR_TIME, RMS_RES, AERO_COEFF )
HISTORY_WRT_FREQ_INNER= 1
HISTORY_WRT_FREQ_TIME= 1
"""
