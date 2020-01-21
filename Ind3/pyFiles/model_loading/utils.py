from pyFiles.model_loading.parser import OBJParser
from typing import Tuple
import numpy as np


def obj_sort(parser: OBJParser) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    textures: np.ndarray = parser.textures()
    normals: np.ndarray = parser.normals()
    vertexes: np.ndarray = parser.vertex()

    textures_sorted: np.ndarray = np.zeros((len(textures), 2))
    normals_sorted: np.ndarray = np.zeros((len(normals), 3))
    vertexes_sorted: np.ndarray = vertexes.copy()

    for vind, tind, nind in zip(parser.vertex_idx(), parser.tex_idx(), parser.normal_idx()):
        for i in range(0, 3, 1):
            textures_sorted[vind[i]] = np.array(textures[tind[i], :2])
            normals_sorted[vind[i]] = np.array(normals[nind[i]])

    return vertexes_sorted, textures_sorted, normals_sorted
