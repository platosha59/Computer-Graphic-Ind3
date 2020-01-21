import pygame
import sys
import os
import json

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import numpy as np

from typing import List, NoReturn, Tuple, Dict, Any, Callable
from pyFiles.model_loading.parser import OBJParser


class GraphicScene:
    def __init__(self, window_width: int, window_height: int, data_dir: str, obj_dir: str,
                 textures_dir: str, shaders_dir: str, light_conf_file: str,
                 sorting_func: Callable[[OBJParser], Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self._width: int = window_width
        self._height: int = window_height
        self._window_title: str = "Graphic scene"

        self._vertexes_container: List[np.ndarray] = []
        self._textures_container: List[np.ndarray] = []
        self._normals_container: List[np.ndarray] = []
        self._indexes_container: List[np.ndarray] = []

        self._light_sources: List[np.ndarray] = []
        self._light_settings: List[Dict[str, Any]] = []
        self._global_ambient: np.ndarray = np.zeros(shape=(4,), dtype=float)

        self._models_settings: List[Dict[str, Any]] = []
        self._textures: List[int] = []

        self._data_dir: str = data_dir
        self._shaders_dir: str = shaders_dir
        self._textures_dir: str = textures_dir
        self._light_conf_file: str = light_conf_file
        self._sorting_function: Callable[[OBJParser], Tuple[np.ndarray, np.ndarray, np.ndarray]] = sorting_func
        self._obj_dir: str = obj_dir
        self._available_light_systems: List[str] = [directory for directory in
                                                    os.listdir(os.path.join(data_dir, shaders_dir)) if
                                                    os.path.isdir(os.path.join(data_dir, shaders_dir, directory))]
        assert len(self._available_light_systems), "Light systems are not found"
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
        glut.glutCreateWindow(self._window_title)
        glut.glutReshapeWindow(self._width, self._height)
        glut.glutReshapeFunc(self.reshape)
        glut.glutDisplayFunc(self.draw)
        glut.glutIdleFunc(self.draw)
        glut.glutSpecialFunc(self.keyboard)
        self._load_models_with_settings()

        self._program = gl.glCreateProgram()
        self._programs: List = []
        self._programs_preparing()
        self._load_light_conf()

        self._eye_position = [0.0, 0.0, 20.0]
        self._target_position = [0.0, 0.0, 0.0]

        self._rotation_x = 0.0
        self._rotation_y = 0.0
        self._rotation_z = 0.0

        self._shift_x = 0.0
        self._shift_y = 0.0
        self._shift_z = 0.0

        self._shift_light_x = 0.0
        self._shift_light_y = 0.0
        self._shift_light_z = 0.0

    def _load_light_system(self, index) -> Tuple[str, str, str]:
        shaders: List[str] = os.listdir(
            os.path.join(self._data_dir, self._shaders_dir, self._available_light_systems[index]))
        if "vertex" in shaders[0]:
            vertex_shader_file: str = shaders[0]
            fragment_shader_file: str = shaders[1]
        else:
            vertex_shader_file: str = shaders[1]
            fragment_shader_file: str = shaders[0]
        vertex_shader_file = os.path.join(self._data_dir, self._shaders_dir, self._available_light_systems[index],
                                          vertex_shader_file)
        fragment_shader_file = os.path.join(self._data_dir, self._shaders_dir, self._available_light_systems[index],
                                            fragment_shader_file)

        with open(vertex_shader_file, "r") as shader_file:
            vertex_shader: str = shader_file.read()

        with open(fragment_shader_file, "r") as shader_file:
            fragment_shader: str = shader_file.read()

        return vertex_shader, fragment_shader, self._available_light_systems[index]

    def _load_models_with_settings(self) -> NoReturn:
        conf_file: str = [file for file in os.listdir(os.path.join(self._data_dir, self._obj_dir)) if
                          os.path.isfile(os.path.join(self._data_dir, self._obj_dir, file)) and ".json" in file][0]
        with open(os.path.join(self._data_dir, self._obj_dir, conf_file), "r") as file_descriptor:
            conf: Dict[str, Any] = json.loads(file_descriptor.read())

        for obj_file, settings in conf.items():
            parser = OBJParser(os.path.join(self._data_dir, self._obj_dir, obj_file))
            vertexes, textures, normals = self._sorting_function(parser)
            self._models_settings.append(settings)
            self._vertexes_container.append(vertexes)
            self._textures_container.append(textures)
            self._normals_container.append(normals)
            self._indexes_container.append(parser.vertex_idx())
            self._textures.append(self._load_texture(settings["texture"]))

    def _load_texture(self, texture_file: str) -> NoReturn:
        image = pygame.image.load(os.path.join(self._data_dir, self._textures_dir, texture_file))
        image_as_bytes, width, height = pygame.image.tostring(image, "RGBA", 1), image.get_size()[0], image.get_size()[
            1]

        texture_id: int = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                        image_as_bytes)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        return texture_id

    def _load_light_conf(self):
        with open(os.path.join(self._data_dir, self._light_conf_file), "r") as file_descriptor:
            conf = json.loads(file_descriptor.read())
        self._global_ambient = conf["global_ambient"]
        for light, light_setting in conf.items():
            if "global_ambient" in light:
                continue
            self._light_sources.append(light_setting["location"])
            self._light_settings.append(light_setting["settings"])

    def _set_texture(self, texture_id):
        texture_location = gl.glGetUniformLocation(self._program, "texture")
        assert texture_location != -1, "Uniform not found"

        gl.glUniform1i(texture_location, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

    def _use_program(self, program_index: int):
        self._program = self._programs[program_index]
        gl.glUseProgram(self._program)

    def _uniform_init(self, index: int):
        loc = gl.glGetUniformLocation(self._program, 'Global_ambient')
        if loc not in (None, -1):
            gl.glUniform4f(loc, self._global_ambient[0], self._global_ambient[1],
                           self._global_ambient[2], self._global_ambient[3])

        loc = gl.glGetUniformLocation(self._program, 'Light_location')
        if loc not in (None, -1):
            gl.glUniform3f(loc, self._light_sources[0][0] + self._shift_light_x,
                           self._light_sources[0][1] + self._shift_light_y,
                           self._light_sources[0][2] + self._shift_light_z)

        loc = gl.glGetUniformLocation(self._program, "Eye_location")
        if loc not in (None, -1):
            gl.glUniform3f(loc, self._eye_position[0], self._eye_position[1], self._eye_position[2])

        loc = gl.glGetUniformLocation(self._program, 'specular_power')
        if loc not in (None, -1):
            gl.glUniform1f(loc, self._models_settings[index]["specular_power"])

        loc = gl.glGetUniformLocation(self._program, 'specular_color')
        if loc not in (None, -1):
            gl.glUniform4f(loc, self._models_settings[index]["specular_color"][0] / 255.,
                           self._models_settings[index]["specular_color"][1] / 255.,
                           self._models_settings[index]["specular_color"][2] / 255.,
                           self._models_settings[index]["specular_color"][3])

        loc = gl.glGetUniformLocation(self._program, 'minnaert_power')
        if loc not in (None, -1):
            gl.glUniform1f(loc, self._models_settings[index]["minnaert_power"])

        loc = gl.glGetUniformLocation(self._program, 'edge_power')
        if loc not in (None, -1):
            gl.glUniform1f(loc, self._models_settings[index]["edge_power"])

        loc = gl.glGetUniformLocation(self._program, 'warm_diffuse')
        if loc not in (None, -1):
            gl.glUniform1f(loc, self._models_settings[index]["warm_diffuse"])

        loc = gl.glGetUniformLocation(self._program, 'cold_diffuse')
        if loc not in (None, -1):
            gl.glUniform1f(loc, self._models_settings[index]["cold_diffuse"])

        loc = gl.glGetUniformLocation(self._program, 'warm_color')
        if loc not in (None, -1):
            gl.glUniform3f(loc, self._light_settings[0]["warm_color"][0], self._light_settings[0]["warm_color"][1],
                           self._light_settings[0]["warm_color"][2])

        loc = gl.glGetUniformLocation(self._program, 'cold_color')
        if loc not in (None, -1):
            gl.glUniform3f(loc, self._light_settings[0]["cold_color"][0], self._light_settings[0]["cold_color"][1],
                           self._light_settings[0]["cold_color"][2])

        loc = gl.glGetUniformLocation(self._program, 'rim_power')
        if loc not in (None, -1):
            gl.glUniform1f(loc, self._models_settings[index]["rim_power"])

        loc = gl.glGetUniformLocation(self._program, 'rim_bias')
        if loc not in (None, -1):
            gl.glUniform1f(loc, self._models_settings[index]["rim_bias"])

        loc = gl.glGetUniformLocation(self._program, 'rim_color')
        if loc not in (None, -1):
            gl.glUniform4f(loc, self._models_settings[index]["rim_color"][0] / 255.,
                           self._models_settings[index]["rim_color"][1] / 255.,
                           self._models_settings[index]["rim_color"][2] / 255.,
                           self._models_settings[index]["rim_color"][3])

    def _programs_preparing(self):
        light_systems_count: int = len(self._available_light_systems)
        for ind in range(0, light_systems_count):
            program = gl.glCreateProgram()
            vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
            fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

            vertex_shader_text, fragment_shader_text, light_system_name = self._load_light_system(ind)
            gl.glShaderSource(vertex_shader, vertex_shader_text)
            gl.glShaderSource(fragment_shader, fragment_shader_text)
            print(f"F{ind + 1} - {light_system_name} light model")
            gl.glCompileShader(vertex_shader)
            assert gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS), "ERROR: Bad compilation of vertex shader"

            gl.glCompileShader(fragment_shader)
            assert gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS), "ERROR: Bad compilation of fragment shader"

            gl.glAttachShader(program, vertex_shader)
            gl.glAttachShader(program, fragment_shader)

            gl.glLinkProgram(program)
            assert gl.glGetProgramiv(program, gl.GL_LINK_STATUS), "ERROR: Problem with program linking"

            gl.glDetachShader(program, vertex_shader)
            gl.glDetachShader(program, fragment_shader)

            self._programs.append(program)

    def draw_models(self):
        for i in range(0, len(self._indexes_container)):
            self._uniform_init(i)

            texture_location = gl.glGetUniformLocation(self._program, "texture")
            # assert texture_location != -1, "Uniform not found"

            gl.glUniform1i(texture_location, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._textures[i])

            count = self._indexes_container[i].shape[0]

            gl.glTranslated(self._models_settings[i]["shifts"][0],
                            self._models_settings[i]["shifts"][1],
                            self._models_settings[i]["shifts"][2])
            gl.glScale(self._models_settings[i]["scales"][0],
                            self._models_settings[i]["scales"][1],
                            self._models_settings[i]["scales"][2])

            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, self._vertexes_container[i])

            gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
            gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, self._textures_container[i])

            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            gl.glNormalPointer(gl.GL_FLOAT, 0, self._normals_container[i])

            gl.glEnableClientState(gl.GL_INDEX_ARRAY)
            gl.glIndexPointer(gl.GL_INT, 0, self._indexes_container[i])

            gl.glDrawElements(gl.GL_TRIANGLES, 3 * count, gl.GL_UNSIGNED_INT, self._indexes_container[i])
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

            gl.glDisableClientState(gl.GL_INDEX_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)

            gl.glTranslated(-1*(self._models_settings[i]["shifts"][0] + self._shift_x),
                            -1*(self._models_settings[i]["shifts"][1] + self._shift_y),
                            -1*(self._models_settings[i]["shifts"][2] + self._shift_z))
            gl.glScale(1. / self._models_settings[i]["scales"][0],
                       1. / self._models_settings[i]["scales"][1],
                       1. / self._models_settings[i]["scales"][2])

    def draw(self) -> NoReturn:
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # gl.glClearColor(12. / 255., 22. / 255., 33. / 255., 1.0)
        gl.glPushMatrix()
        gl.glTranslated(self._shift_x, self._shift_y, self._shift_z)
        gl.glPopMatrix()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(65.0, self._width / self._height, 0.1, 1000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glu.gluLookAt(self._eye_position[0], self._eye_position[1], self._eye_position[2],
                      self._target_position[0], self._target_position[1], self._target_position[2], 0.0, 1.0, 0.0)

        gl.glRotatef(self._rotation_x, 1.0, 0.0, 0.0)
        gl.glRotatef(self._rotation_y, 0.0, 1.0, 0.0)
        gl.glRotatef(self._rotation_z, 0.0, 0.0, 1.0)

        self.draw_models()

        glut.glutSwapBuffers()

    def reshape(self, width, height) -> NoReturn:
        gl.glViewport(0, 0, width, height)

    def keyboard(self, key, *args):
        mod = glut.glutGetModifiers()
        if key == b'\x1b':
            sys.exit()
        if mod != glut.GLUT_ACTIVE_ALT and mod != glut.GLUT_ACTIVE_CTRL and mod != glut.GLUT_ACTIVE_SHIFT:
            if key == glut.GLUT_KEY_UP:
                self._target_position[1] += 1.
            if key == glut.GLUT_KEY_DOWN:
                self._target_position[1] -= 1.
            if key == glut.GLUT_KEY_LEFT:
                self._target_position[0] -= 1.
            if key == glut.GLUT_KEY_RIGHT:
                self._target_position[0] += 1.
            if key == glut.GLUT_KEY_HOME:
                self._target_position[2] += 1.
            if key == glut.GLUT_KEY_END:
                self._target_position[2] -= 1.
        elif mod == glut.GLUT_ACTIVE_CTRL:
            if key == glut.GLUT_KEY_UP:
                self._rotation_x += 2.0
            if key == glut.GLUT_KEY_DOWN:
                self._rotation_x -= 2.0
            if key == glut.GLUT_KEY_LEFT:
                self._rotation_y += 2.0
            if key == glut.GLUT_KEY_RIGHT:
                self._rotation_y -= 2.0
            if key == glut.GLUT_KEY_HOME:
                self._rotation_z += 2.0
            if key == glut.GLUT_KEY_END:
                self._rotation_z -= 2.0
        elif mod == glut.GLUT_ACTIVE_ALT:
            if key == glut.GLUT_KEY_UP:
                self._shift_light_y += .5
            if key == glut.GLUT_KEY_DOWN:
                self._shift_light_y -= .5
            if key == glut.GLUT_KEY_LEFT:
                self._shift_light_x -= .5
            if key == glut.GLUT_KEY_RIGHT:
                self._shift_light_x += .5
            if key == glut.GLUT_KEY_HOME:
                self._shift_light_z += .5
            if key == glut.GLUT_KEY_END:
                self._shift_light_z -= .5
        elif mod == glut.GLUT_ACTIVE_SHIFT:
            if key == glut.GLUT_KEY_UP:
                self._eye_position[1] += .5
            if key == glut.GLUT_KEY_DOWN:
                self._eye_position[1] -= .5
            if key == glut.GLUT_KEY_LEFT:
                self._eye_position[0] -= .5
            if key == glut.GLUT_KEY_RIGHT:
                self._eye_position[0] += .5
            if key == glut.GLUT_KEY_HOME:
                self._eye_position[2] += 1.
            if key == glut.GLUT_KEY_END:
                self._eye_position[2] -= 1.

        if key == glut.GLUT_KEY_F1:
            self._use_program(0)
        if key == glut.GLUT_KEY_F2:
            self._use_program(1)
        if key == glut.GLUT_KEY_F3:
            self._use_program(2)
        if key == glut.GLUT_KEY_F7:
            self._shift_light_x, self._shift_light_y, self._shift_light_z = 0., 0., 0.
            
        if key == glut.GLUT_KEY_INSERT:
            glut.glutLeaveMainLoop()
        glut.glutPostRedisplay()

    def mainloop(self):
        print("Left, Up, Right, Down, Home or End - camera moving")
        print("Ctrl + Left, Up, Right, Down, Home or End - scene moving")
        print("Alt + Left, Up, Right, Down, Home or End - light source moving")
        print("Ins - exit")
        self._use_program(0)

        gl.glEnable(gl.GL_DEPTH_TEST)
        glut.glutMainLoop()
