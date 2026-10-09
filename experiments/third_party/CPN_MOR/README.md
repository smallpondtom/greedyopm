This is the implementation of the method proposed in [1] https://www.arxiv.org/abs/2502.05088

The code relies on the library tensap [2], and features implementations of compositional polynomial networks, with sparse 
approximation (CPN-S) or low-rank approximation (CPN-LR). It also contains implementations of the different manifold approximation methods 
used for comparisons in [3, 4, 5].

To use the code : 
- Clone the repository

  ```
  git clone https://github.com/JoelSOFFO/CPN_MOR.git
  ```
  
  - Install the package tensap, see https://github.com/anthony-nouy/tensap
      
  ```
  pip install tensap
  ```
   To avoid potential errors due to updates, you can install tensap manually
  
  ```
    git clone https://github.com/anthony-nouy/tensap.git
    cd tensap
    pip install -e .
    cd ..
    ```
  
- Install the required libraries

  ```
  pip install -r requirements.txt
  ```
  
- Data format: data is given as a numpy array of shape (D, m), whoses columns represent m samples in R^D.
    It is given in a npy file whose path is given in config files (cpn.yaml and sota.yaml).

  ```
  cd CPN_MOR
  ```
  
- To run CPN, use the command line

  ```
  python3 CPN_MOR.py --config_path configs/cpn.yaml --mode choose_a_mode 
  ```

  - The parameters of the method can be changed in the config file cpn.yaml.
  - There are 3 modes available :  "--mode train", "--mode test" and "--mode plot" for respectively training the model, testing it, or plotting the 
  compositional networks (graphs) of some indices (The indices must be defined in "indices_to_plot" in the config file).

  
- To run the other methods used in the article, use the command line

  ```
    python3 sota_MOR.py --config_path configs/sota.yaml 
  ```

  - The parameters related to these methods can be changed in the config file sota.yaml.

The datasets used can be found here : https://drive.google.com/drive/folders/1LdhMTs0U1fMQrGGr5kj7LDj8qJteujYb?usp=sharing

If you make use of this code in a paper, please cite [1]

```
@misc{bensalah2025nonlinearmanifoldapproximationusing,
      title={Nonlinear manifold approximation using compositional polynomial networks}, 
      author={Antoine Bensalah and Anthony Nouy and Joel Soffo},
      year={2025},
      eprint={2502.05088},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2502.05088}, 
}
```

[1] Antoine Bensalah, Anthony Nouy, and Joel Soffo. Nonlinear manifold approximation using compositional polynomial networks. arXiv e-prints arXiv:2502.05088, Feb. 2025.

[2] Anthony Nouy and Erwan Grelier. tensap, doi:10.5281/zenodo.3894378, June 2020.

[3] Joshua Barnett and Charbel Farhat. Quadratic approximation manifold for mitigating
the kolmogorov barrier in nonlinear projection-based model order reduction. Journal of
Computational Physics, 464:111348, September 2022.

[4] Rudy Geelen, Stephen Wright, and Karen Willcox. Operator inference for non-intrusive
model reduction with quadratic manifolds. Computer Methods in Applied Mechanics and
Engineering, 403:115717, January 2023.

[5]  Rudy Geelen, Laura Balzano, Stephen Wright, and Karen Willcox. Learning physics-based
reduced-order models from data using nonlinear manifolds. Chaos, 34(3):033122, March 2024.
