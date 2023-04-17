import os
from pathlib import Path


class ProjectPaths:

    root = Path(os.path.abspath(os.path.dirname(__file__))).parent
    data = os.path.join(root, "data")
