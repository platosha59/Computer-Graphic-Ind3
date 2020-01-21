import numpy as np


class OBJParser:
    def __init__(self, filename):
        with open(filename, "r") as file_descriptor:
            self._lines: list = file_descriptor.read().split("\n")

        self._format_specification: dict = {
            "v": self._vertex,
            "vt": self._texture,
            "vn": self._norm,
            "vp": self._vspace,
            "f": self._face,
            "g": self._group,
            "o": self._object,
            "mtllib": self._mtl,
            "#": self._comment
        }
        self._faces: list = []
        self._log: str = "{}"
        self._vertexes: list = []
        self._textures: list = []
        self._norms: list = []
        self._vspaces: list = []
        self._poistions: list = []
        self._normal_pos: list = []
        self._tex_pos: list = []

        self._parse()

    def top(self, n: int = 5) -> list:
        return self._lines[:n]

    def tail(self, n: int = 5) -> list:
        return self._lines[-n:]

    def faces(self) -> list:
        return self._faces

    def log(self) -> str:
        return self._log

    def vertex_idx(self):
        return np.array(self._poistions, dtype=np.int)

    def normal_idx(self):
        return np.array(self._normal_pos, dtype=np.int)

    def tex_idx(self):
        return np.array(self._tex_pos, dtype=np.int)

    def vertex(self):
        return np.array(self._vertexes, dtype=np.float32)

    def normals(self):
        return np.array(self._norms, dtype=np.float32)

    def textures(self):
        return np.array(self._textures, dtype=np.float32)

    def _parse(self):
        for line in self._lines:
            try:
                line_elements: list = line.split(" ")

                if '' in line_elements:
                    line_elements.remove('')

                if len(line_elements) == 0:
                    continue

                self._format_specification.get(line_elements[0], self._default)(line_elements[1:])
            except Exception as e:
                print(e)
                print("Line:", line)
                raise Exception(e)

    def _default(self, line: list):
        self._log.format("Unknown info: " + " ".join(line) + "\n{}")

    def _vertex(self, line: list):
        vertex: np.ndarray = np.zeros(shape=(3,), dtype=float)
        ind: int = 0
        for i in range(0, len(line)):
            try:
                if line[i] != '':
                    vertex[ind] = line[i]
                    ind += 1
            except Exception as e:
                print(e)
                print("Line:", line)
                raise ValueError("Bad input")

        self._vertexes.append(vertex)

    def _texture(self, line: list):
        texture: np.ndarray = np.zeros(shape=(3,), dtype=float)
        ind: int = 0
        for i in range(0, len(line)):
            texture[ind] = line[i]
            ind += 1
        self._textures.append(texture)

    def _norm(self, line: list):
        norm: np.ndarray = np.zeros(shape=(3,), dtype=float)
        ind: int = 0
        for i in range(0, len(line)):
            norm[ind] = line[i]
            ind += 1
        self._norms.append(norm)

    def _vspace(self, line: list):
        vspace: list = []
        for i in range(0, len(line)):
            vspace.append(float(line[i]))
        self._vspaces.append(vspace)

    def _face(self, line: list):
        if len(line) == 3:
            position: list = []
            norm: list = []
            tex: list = []
            for i in range(0, 3):
                try:
                    vertex_num: int = int(line[i].split("/")[0]) - 1
                    texture_num: int = int(line[i].split("/")[1]) - 1
                    normal_num: int = int(line[i].split("/")[2]) - 1
                    position.append(vertex_num)
                    tex.append(texture_num)
                    norm.append(normal_num)
                except Exception as e:
                    self._comment([e])

            self._poistions.append(position)
            self._normal_pos.append(norm)
            self._tex_pos.append(tex)
        else:
            for j in range(len(line) - 2):
                position: list = []
                norm: list = []
                tex: list = []
                for i in range(j, j + 3):
                    try:
                        vertex_num: int = int(line[i].split("/")[0]) - 1
                        texture_num: int = int(line[i].split("/")[1]) - 1
                        normal_num: int = int(line[i].split("/")[2]) - 1
                        position.append(vertex_num)
                        tex.append(texture_num)
                        norm.append(normal_num)
                    except Exception as e:
                        self._comment([e])

                self._poistions.append(position)
                self._normal_pos.append(norm)
                self._tex_pos.append(tex)

    def _comment(self, line: list):
        self._log = self._log.format(" ".join(line) + "\n{}")

    def _group(self, line: list):
        pass

    def _object(self, line: list):
        pass

    def _mtl(self, line: list):
        pass

    def update_vertexes(self, faces: np.ndarray) -> None:
        for i in range(faces.shape[0]):
            position: list = self._poistions[i]
            face: np.ndarray = faces[i]

            for j in range(0, len(position)):
                try:
                    point: np.ndarray = face[j]
                    at: int = position[j]
                except Exception as e:
                    print("Position:", position)
                    print("Face:", face)
                    print(e)
                    raise Exception("Bad input")

                vertex: list = [point[0], point[1], point[2]]
                self._vertexes[at] = vertex

    def to_obj_file(self, filename) -> None:
        i: int = 0
        line: str = self._lines[i]
        while not line.startswith("v "):
            i += 1
            line: str = self._lines[i]

        v_num: int = 0
        while line.startswith("v "):
            new_line = "v"
            for element in self._vertexes[v_num]:
                new_line += " " + str(element)
            self._lines[i] = new_line

            i += 1
            v_num += 1
            line: str = self._lines[i]

        with open(filename, "w") as f:
            f.write("\n".join(self._lines))


