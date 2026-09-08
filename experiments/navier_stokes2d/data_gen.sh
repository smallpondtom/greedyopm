for a in 4 6 8 10 12 14 16 18 20 24; do
  NS_ALPHA=$a NS_NX=512 NS_NY=256 NS_TEND=30 NS_SNAPDT=0.2 NS_SAVEVEL=1 \
    NS_OUT=data/naca_a$(printf '%02d' $a).npz \
    python naca_solver.py
done