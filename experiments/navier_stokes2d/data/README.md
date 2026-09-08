# data/

Populated by `python ../run_param.py`, which writes one file per angle:

    naca_a04.npz  naca_a08.npz  naca_a12.npz  naca_a16.npz  naca_a20.npz  naca_a24.npz

Each `.npz` holds `x, y, t, u, v, omega, chi` plus metadata (Re, alpha_deg, dt,
chord, U_inf) at 512 x 256 over t in [0, 30], snapshot spacing 0.2. Each file is
about 200 MB (the six-angle set is ~1.2 GB), so the raw data is not committed here;
regenerate with `python run_param.py`.
