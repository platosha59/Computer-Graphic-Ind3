from pyFiles.model_loading.parser import OBJParser
from pyFiles.model_loading.utils import obj_sort
from pyFiles.graphic_scene import GraphicScene


__ALL__ = [
    "mainloop"
]


DATA_DIR = "data"
SHADERS_DIR = "shaders"
MODELS_DIR = "models"
TEXTURES_DIR = "textures"
LIGHT_CONF = "light.json"
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080


def mainloop():
    gs = GraphicScene(WINDOW_WIDTH, WINDOW_HEIGHT, DATA_DIR, MODELS_DIR, TEXTURES_DIR, SHADERS_DIR,
                      LIGHT_CONF, obj_sort)
    gs.mainloop()

