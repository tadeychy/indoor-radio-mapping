import numpy as np
import open3d as o3d
import open3d as o3d
import numpy as np
from typing import Union, Optional, Tuple

import functions as IJS
pcd_main = o3d.io.read_point_cloud("ply/pcd/office2.ply")

o3d.visualization.draw_geometries([pcd_main])




