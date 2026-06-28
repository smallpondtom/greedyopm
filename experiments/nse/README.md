For this experiment, run the following files in order:
- `compile_nse.py` to compile the raw snapshot data into a single file.
- `prep_nse.py` to perform POD compression on the compiled snapshots.
- `recover_dof_coords.py` to recover the coordinates for flow field visualization.
- `nse_compare.py` to compare the performance of different optimization methods on the NSE problem
- `nse_greedyopm_field.py` to visualize the flow fields obtained from the greedy OPM method.
