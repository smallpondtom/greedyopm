import sys
sys.path.append("sota")

import linear, quadratic, additive_am, sparse, low_rank
import yaml
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    method = config["params"]["method"]

    if method == "quadratic":
        quadratic.run(config)
    elif method == "additive_am":
        additive_am.run(config)
    elif method == "sparse":
        sparse.run(config)
    elif method == "low_rank":
        low_rank.run(config)
    else:
        linear.run(config)
