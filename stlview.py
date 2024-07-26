# Courtesy of Liam Dempsey (2022)
# Github: Drayux
#
# Modified: Jonathan Shulgach (2024)
# Github: JShulgach

import math
import struct
import sys
import tkinter

# DEFAULT CUBE DATA
# Num faces
# F1: V1-X, V1-Y, V1-Z,  V2-X, V2-Y, V2-Z,  V3-X, V3-Y, V3-Z,  N-X, N-Y, N-Z
# F2: V1-X, V1-Y, V1-Z, ...
cube = [12,
        -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -0.0, 0.0, 1.0,
        1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0,
        1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -0.0, -0.0, -1.0,
        -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0,
        -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0,
        -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -0.0, 1.0, 0.0,
        -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0,
        1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -0.0, -1.0, -0.0,
        1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 0.0, -1.0, 0.0]


# Vertex class (used as component of Face class; basically a vector)
# Supports basic vector operations
class Vertex:
    def __init__(self, x=0, y=0, z=0, w=1, parent=None, pr=-1, pc=-1):
        self._parent = parent  # Pointer to parent matrix (if applicable)
        self._pr = pr
        self._pc = pc

        if isinstance(x, (list, tuple)):
            self.x = x[0] if len(x) > 0 else 0
            self.y = x[1] if len(x) > 1 else 0
            self.z = x[2] if len(x) > 2 else 0
            self.w = x[3] if len(x) > 3 else w
        else:
            self.x = x
            self.y = y
            self.z = z
            self.w = w

        if type(x) is list or type(x) is tuple:
            try:
                self.x = x[0]
            except IndexError:
                self.x = 0

            try:
                self.y = x[1]
            except IndexError:
                self.y = y

            try:
                self.z = x[2]
            except IndexError:
                self.z = z

            try:
                self.w = x[3]
            except IndexError:
                self.w = w

        # Not currently checking for appropriate types
        else:
            self.x = x
            self.y = y
            self.z = z
            self.w = w

    def __repr__(self):
        return f"[{self.x}, {self.y}, {self.z}, {self.w}]"

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __lt__(self, other):
        return self.z < other.z

    def __getitem__(self, i):
        if i == 0 or i == -4:
            return self.x
        if i == 1 or i == -3:
            return self.y
        if i == 2 or i == -2:
            return self.z
        if i == 3 or i == -1:
            return self.w
        raise IndexError("index out of range [0, 3]")

    def __setitem__(self, i, v):
        if i == 0 or i == -4:
            self.x = v
        elif i == 1 or i == -3:
            self.y = v
        elif i == 2 or i == -2:
            self.z = v
        elif i == 3 or i == -1:
            self.w = v
        else:
            raise IndexError("index out of range [0, 3]")

        if self._parent is not None:
            # Vector was a row
            if self._pr >= 0:
                self._parent.data[(self._pr * 4) + i] = v

            # Vector was a column (elif not else so matricies don't get borked)
            elif self._pc >= 0:
                self._parent.data[self._pc + (4 * i)] = v

    def __len__(self):
        return 4

    def __eq__(self, other):
        return (abs(self.x - other.x) < 0.01) \
            and (abs(self.y - other.y) < 0.01) \
            and (abs(self.z - other.z) < 0.01) \
            and (abs(self.w - other.w) < 0.01)

    # Add vectors (ignores W)
    def __add__(self, other):
        # Scalar addition
        if type(other) is int or type(other) is float:
            return Vertex(other + self.x, other + self.y, other + self.z)

        # Vector addition
        if type(other) is Vertex:
            return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)

        raise TypeError(f"unsupported type {type(other)}: expected int, float, or Vertex")

    # Subtract vectors (ignores W)
    def __sub__(self, other):
        # Scalar subtraction
        if type(other) is int or type(other) is float:
            return Vertex(self.x - other, self.y - other, self.z - other)

        # Vector subtraction
        if type(other) is Vertex:
            return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)

        raise TypeError(f"unsupported type {type(other)}: expected int, float, or Vertex")

    def __mul__(self, other):
        # Return scaled vector if other is scalar (ignores W)
        if type(other) is int or type(other) is float:
            return Vertex(other * self.x, other * self.y, other * self.z)

        # Return dot product if other is Vertex (ignores W)
        if type(other) is Vertex:
            return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

        # Return transformed vector if other is Matrix
        if type(other) is Matrix:
            coords = []
            for i in range(4):
                coords.append((self.x * other[0][i]) \
                              + (self.y * other[1][i]) \
                              + (self.z * other[2][i]) \
                              + (self.w * other[3][i]))
            return Vertex(coords[0], coords[1], coords[2], coords[3])

        raise TypeError(f"unsupported type {type(other)}: expected int, float, Vertex, or Matrix")

    # Cross product
    def __pow__(self, other):
        # Type must be a Vertex
        if type(other) is Vertex:
            x = (self.y * other.z) - (self.z * other.y)
            y = (self.z * other.x) - (self.x * other.z)
            z = (self.x * other.y) - (self.y * other.x)
            return Vertex(x, y, z)

        raise TypeError(f"unsupported type {type(other)}: expected Vertex")

    # Magnitude (aka length, size, etc)
    def size(self):
        # Dot product w/ self yields a^2 + b^2 + c^2
        return (self * self) ** 0.5

    # Normalized
    def unit(self):
        scalar = 0 if self.size() == 0 else 1 / self.size()
        return self * scalar

    # 'Full' dot product including W
    def mdot(self, other):
        if type(other) is Vertex:
            return (self.x * other.x) \
                + (self.y * other.y) \
                + (self.z * other.z) \
                + (self.w * other.w)
        raise TypeError(f"unsupported type {type(other)}: Vertex")

    # Return the Vertex data as a tuple
    def tuple(self):
        return (self.x, self.y, self.z, self.w)


# Simple 4x4 matrix class for object transformations
class Matrix:
    # Initializes as identity matrix
    def __init__(self):
        self.data = [0 for i in range(16)]  # Raw data: (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), .... (4, 4)

        # Populate values
        for i in range(4): self.data[(i * 4) + i] = 1

    '''
    # REFACTORED!
    # Workaround for directly assigning values of matrix with [i][j] syntax
    def _update(self):
        if self._ri >= 0:
            self.data[self._ri * 4] = self._rdata[0]
            self.data[(self._ri * 4) + 1] = self._rdata[1]
            self.data[(self._ri * 4) + 2] = self._rdata[2]
            self.data[(self._ri * 4) + 3] = self._rdata[3]

            self._ri = -1
            self._rdata = []

    def __getitem__(self, i):
        if i < 0 or i > 3: raise IndexError("index out of range [0, 3]")

        self._ri = i
        self._rdata.append(self.data[i * 4])
        self._rdata.append(self.data[(i * 4) + 1])
        self._rdata.append(self.data[(i * 4) + 2])
        self._rdata.append(self.data[(i * 4) + 3])

        return self._rdata
    '''

    def __getitem__(self, i):
        if i < 0 or i > 3: raise IndexError("index out of range [0, 3]")
        return Vertex(self.data[i * 4:(i * 4) + 4], parent=self, pr=i)

    def __len__(self):
        return 4

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        out = ""
        for i, row in enumerate(self.rows()):
            out += row.__repr__() + ("\n" if i < 3 else "")
        return out

    # Matrix multiplication
    def __mul__(self, other):
        if type(other) is Matrix:
            result = Matrix()
            for i, row in enumerate(self.rows()):
                for j, col in enumerate(other.cols()):
                    result.data[(i * 4) + j] = row.mdot(col)
            return result
        raise TypeError(f"unsupported type {type(other)}: expected Matrix")

    # Return list of rows (as Vertex objects)
    def rows(self):
        arr = []
        for i in range(4):
            row = []
            for j in range(4):
                row.append(self.data[(i * 4) + j])
            # arr.append(Vertex(row, parent = self, pr = i))
            arr.append(Vertex(row, parent=self))
        return arr

    # Return list of columns (as Vertex objects)
    def cols(self):
        arr = []
        for i in range(4):
            col = []
            for j in range(4):
                col.append(self.data[(j * 4) + i])
            # arr.append(Vertex(col, parent = self, pc = i))
            arr.append(Vertex(col, parent=self))
        return arr


# Simple face class (used as component of Object class)
# A face will ALWAYS consist of exactly three vertices (order matters!)
# Each face is rendered by a draw call, which takes in a transformation matrix
class Face:
    def __init__(self, v1, v2=None, v3=None):
        if type(v1) is list or type(v1) is tuple:
            if (len(v1) != 3): raise TypeError(f"expected tuple with length 3: got {len(v1)}")
            if not (type(v1[0]) is Vertex and type(v1[1]) is Vertex and type(v1[2]) is Vertex):
                raise TypeError(f"unsupported type: expected Vertex")

            self._v1 = Vertex(v1[0].tuple())
            self._v2 = Vertex(v1[1].tuple())
            self._v3 = Vertex(v1[2].tuple())

        elif v2 is not None and v3 is not None:
            if not (type(v1) is Vertex and type(v2) is Vertex and type(v3) is Vertex):
                raise TypeError(f"unsupported type: expected Vertex")

            self._v1 = Vertex(v1.tuple())
            self._v2 = Vertex(v2.tuple())
            self._v3 = Vertex(v3.tuple())

        else:
            raise TypeError(f"expected 3 vertices: got {1 + int(v2 is not None) + int(v3 is not None)}")

        self._normal = ((self._v2 - self._v1) ** (self._v3 - self._v1)).unit()
        self._changed = False

    def __repr__(self):
        return f"[{self._v1}, {self._v2}, {self._v3}]"

    # Todo display cross product
    def __str__(self):
        return f"V1: {self._v1}\nV2: {self._v2}\nV3: {self._v3}\nNormal: {self.normal()}"

    def __lt__(self, other):
        # Find min Z values
        # amin = min(self._v1, self._v2, self._v3)
        amax = max(self._v1, self._v2, self._v3)
        bmin = min(other._v1, other._v2, other._v3)
        bmax = max(other._v1, other._v2, other._v3)

        # Calculate vector between two points
        # vec = bmax - amin

        # Determine which face is on the outside
        # (outside - inside) * normal is positive
        # return vec * self.normal() > 0.0

        if amax < bmin:
            return True
        else:
            return amax < bmax

    # Returns the specified vertex itself
    def __getitem__(self, i):
        self._changed = True
        if i == 0 or i == -3: return self._v1
        if i == 1 or i == -2: return self._v2
        if i == 2 or i == -1: return self._v3

        self._changed = False
        raise IndexError("index out of range [0, 2]")

    def __len__(self):
        return 3

    # Return a new face with the computed transformation
    # This is an 'after the fact functionality' function, so the implimentation
    #   is not perfect and very specific
    def transform(self, canvas, mat):
        v1t = self._v1 * mat
        v2t = self._v2 * mat
        v3t = self._v3 * mat

        # Coordinate conversion (object to screen)
        scr = Matrix()
        scr[3][0] = canvas.width / 2
        scr[3][1] = canvas.height / 2

        v1t *= scr
        v2t *= scr
        v3t *= scr

        # Factor the projection skew
        v1t.x /= v1t.w
        v1t.y /= v1t.w
        v2t.x /= v2t.w
        v2t.y /= v2t.w
        v3t.x /= v3t.w
        v3t.y /= v3t.w

        # Calculate normal and determine if face should be rendered
        face = Face(v1t, v2t, v3t)
        if not canvas.culling or ((Vertex(0.0, 0.0, -1.0) * face.normal()) > 0.0): return face

        return None

    # Draw the face to the (referenced) canvas
    # NOTE! This function should ONLY be called on a triangle returned by the
    #   transform member function! This is because memory allocation is slow!
    #   However, this modify face member values!
    def draw(self, canvas, mat=None):
        '''
             ALL THE FUNCTIONALITY HERE WAS MOVED TO Face.transform()!
             Now this fuction only performs math for a coupld draw modes
               and performs the *actual* drawing part!
        '''

        # v1t = Vertex(self._v1.tuple())
        # v2t = Vertex(self._v2.tuple())
        # v3t = Vertex(self._v3.tuple())

        # v1t = self._v1
        # v2t = self._v2
        # v3t = self._v3
        #
        # if type(mat) is Matrix:
        #     v1t *= mat
        #     v2t *= mat
        #     v3t *= mat
        # print(v1t.__repr__(), v2t.__repr__(), v3t.__repr__())
        # print(nt.__repr__())

        # Coordinate conversion (object to screen)
        # scr = Matrix()
        # scr[3][0] = canvas.width / 2
        # scr[3][1] = canvas.height / 2

        # v1t *= scr
        # v2t *= scr
        # v3t *= scr

        # Factor the projection skew
        # v1t.x /= v1t.w
        # v1t.y /= v1t.w
        # v2t.x /= v2t.w
        # v2t.y /= v2t.w
        # v3t.x /= v3t.w
        # v3t.y /= v3t.w

        # Basic culling algorithm
        # viewport = Vertex(0.0, 0.0, -1.0)  # Vector pointing at the user
        # normal = ((v2t - v1t) ** (v3t - v1t)).unit()

        # if viewport * normal <= 0: return

        if not canvas.fill:
            canvas.create_polygon([(self._v1.x, self._v1.y), (self._v2.x, self._v2.y), (self._v3.x, self._v3.y)], \
                                  fill="", outline="#000")
            canvas.verts += 3
            return

        # The face *should* be rendered
        fillstr = "#FFF"
        if canvas.shading:
            light = Vertex(0.5773502, -0.5773502, -0.5773502)  # Vector pointing towards the light source
            brightness = int((light * self.normal() * 127.5) + 127.5)
            fillstr = f"#{brightness:X}{brightness:X}{brightness:X}"

        canvas.create_polygon([(self._v1.x, self._v1.y), (self._v2.x, self._v2.y), (self._v3.x, self._v3.y)], \
                              fill=fillstr, outline=canvas.outlinestr)
        canvas.verts += 3

        # Debug goodies
        # print(v1t.__repr__())
        # print(v2t.__repr__())
        # print(v3t.__repr__())
        # print()

    # Returns a copy of the vertex (kinda like C++ vector.at())
    def vertex(self, i):
        return Vertex(self.__getitem__(self, i).tuple())

    def normal(self):
        if self._changed: self._normal = ((self._v2 - self._v1) ** (self._v3 - self._v1)).unit()
        return Vertex(self._normal.tuple())

    def tuple(self):
        return (Vertex(self._v1.tuple()), Vertex(self._v2.tuple()), Vertex(self._v3.tuple()))


# Object class for structured vertex storage and manipulation
# Iterable by faces
# Rendering performed by member method 'draw()' which references the canvas size
class Object:
    def __init__(self, inf=None):
        self.name = inf  # Name of the file that contains the vertex data
        self.cname = ('Default-Cube' if self.name is None else self.name.split('.')[0])
        self.data = []  # Array of face objects that comprise the model
        self.numfaces = 0

        # Unused for this project: world position information, but this is where I'd put it

        self.tx = 0
        self.ty = 0
        self.tz = 0
        self.rx = -0.523599  # Radians
        self.ry = -0.785398  # Radians
        self.rz = 0  # Radians
        self.sx = 0
        self.sy = 0
        self.sz = 0

        self._coffset = Matrix()  # Additional translation matrix to offset center of object
        self._tr = Matrix()  # Translation matrix
        self._rx = Matrix()  # Rotation matrix (around X axis)
        self._ry = Matrix()  # Rotation matrix (around Y axis)
        self._rz = Matrix()  # Rotation matrix (around Z axis)
        self._sc = Matrix()  # Scale matrix
        self.transform = None  # Final transformation matrix for any given draw call

    # Return an iterator for the object (iterates through the faces)
    def __iter__(self):
        return ObjectIter(self)

    def __getitem__(self, i):
        return self.data[i]

    # In retrospect, there's no reason to set an object in this way
    # def __setitem__(self, i, v):
    #     if type(v) is Face: self.data[i] = v
    #     else: raise TypeError(f"unsupported type {type(v)}: expected Face")

    def __len__(self):
        return len(self.data)

    # Return a copy of the face at a specified index
    def face(self, i):
        return Face(self.__getitem__(self, i).tuple())

    # Construct the object from its specified STL file
    def build(self):
        if len(self.data) > 0:
            print(f"ERROR: {self.cname} has already been built")
            return

        print(f"Creating object {self.cname}...")

        # Get the array of values
        global cube
        objdata = (cube if self.name is None else parseSTL(self.name))
        objiter = iter(objdata)

        # Total number of faces (used to warn of corrupt file, but will not invalidate object)
        try:
            numfaces = next(objiter)
        except StopIteration:
            print(f"ERROR: Unable to build object {self.cname}, no data")
            return

        badfaces = []

        # Parse values from the data array into their respective objects
        minx = float('inf')
        maxx = float('-inf')

        miny = float('inf')
        maxy = float('-inf')

        while True:
            try:
                # Vertex 1
                v1x = next(objiter)
                v1y = next(objiter)
                v1z = next(objiter)

                # Vertex 2
                v2x = next(objiter)
                v2y = next(objiter)
                v2z = next(objiter)

                # Vertex 3
                v3x = next(objiter)
                v3y = next(objiter)
                v3z = next(objiter)

                # Face normal
                nx = next(objiter)
                ny = next(objiter)
                nz = next(objiter)

                # Used to compute the initial object scale
                minx = min(minx, v1x, v2x, v3x)
                maxx = max(maxx, v1x, v2x, v3x)

                miny = min(miny, v1y, v2y, v3y)
                maxy = max(maxy, v1y, v2y, v3y)

                # Build the face object
                v1 = Vertex(v1x, v1y, v1z)
                v2 = Vertex(v2x, v2y, v2z)
                v3 = Vertex(v3x, v3y, v3z)

                face = Face(v1, v2, v3)
                self.numfaces += 1

                # Compare the face normals
                if face.normal() != Vertex(nx, ny, nz):
                    badfaces.append(self.numfaces)

                self.data.append(face)

            except StopIteration:
                break

        # Check the integrity of what was parsed
        # Not enough data was present to create a face
        if self.numfaces < 1:
            print(f"ERROR: Unable to build object {self.cname}, insufficent data")
            return

        # Mismatched number of faces generated versus faces specifed (in the file header)
        if self.numfaces < numfaces:
            print(
                f"WARNING: Your STL file may be corrupt! Expected {numfaces} trangles but only {self.numfaces} were generated")
        elif self.numfaces > numfaces:
            print(
                f"WARNING: Your STL file may be corrupt! Expected {numfaces} trangles but {self.numfaces} were generated")

        # At least one face's normal did not match with what was computed from its points
        if len(badfaces) > 0:
            print(f"WARNING: Your STL file may be corrupt! The following faces have mismatching normals:")
            print(badfaces)

        # Warn for potential bad performance
        if self.numfaces > 1000:
            print("WARNING: Object has more than 1000 faces! Program performance may be limited!")

        # Object adjustments
        sizex = maxx - minx
        sizey = maxy - miny

        # Center the object
        coffx = -minx - (sizex / 2)
        coffy = -miny - (sizey / 2)
        self._coffset[3][0] = coffx
        self._coffset[3][1] = coffy

        # Calculate the inital scale
        scale = 200 / max(sizex, sizey)
        self.sx = scale
        self.sy = scale
        self.sz = scale

        # Output information about the data
        print("Number of faces:", self.numfaces)
        print("Size:", sizex, sizey)
        print("Center offset:", coffx, coffy)
        print("Initial scale:", scale)
        print()

    # Object transformation stuff
    def translate(self, x, y, z):
        self.tx += x
        self.ty += y
        self.tz += z

    def rotate(self, x, y, z):
        self.rx += x
        self.ry += y
        self.rz += z

    def scale(self, x, y, z, additive=False):
        if additive:
            self.sx += x
            self.sy += y
            self.sz += z

        else:
            self.sx *= x
            self.sy *= y
            self.sz *= z

    def draw(self, canvas):
        # Calc updated transformation matricies
        # Translation
        self._tr[3][0] = self.tx
        self._tr[3][1] = self.ty
        self._tr[3][2] = self.tz

        # Rotation
        self._rx[1][1] = math.cos(self.rx)
        self._rx[1][2] = math.sin(self.rx) * -1
        self._rx[2][1] = math.sin(self.rx)
        self._rx[2][2] = math.cos(self.rx)

        self._ry[0][0] = math.cos(self.ry)
        self._ry[0][2] = math.sin(self.ry)
        self._ry[2][0] = math.sin(self.ry) * -1
        self._ry[2][2] = math.cos(self.ry)

        self._rz[0][0] = math.cos(self.rz)
        self._rz[0][1] = math.sin(self.rz) * -1
        self._rz[1][0] = math.sin(self.rz)
        self._rz[1][1] = math.cos(self.rz)

        # Scale
        self._sc[0][0] = self.sx
        self._sc[1][1] = self.sy
        self._sc[2][2] = self.sz

        # Projection
        proj = Matrix()
        proj[2][3] = canvas.POV

        # Final transform matrix
        # self.transform = self._coffset * self._rz * self._ry * self._rx * self._tr * self._sc * proj
        self.transform = self._coffset * self._ry * self._rx * proj * self._sc  # Ignores matricies not used for the model viewer

        # Render each face of the object
        queue = []
        for face in self:
            f = face.transform(canvas, self.transform)
            if f is not None: queue.append(f)

        # This is the *least* clean portion of my code for the entirety of the project
        # The problem this section of code seeks to solve is vertex layering
        # However, the biggest issue with solving this problem, is that my
        # graphics 'interface' is simply drawing a triangle
        # I have no z-buffer, nor vertex-shader API eqivalent, so I have no means
        # of clipping pixels one-by-one without doubling the size of this project
        # As such, the next best solution is to modify the order in which triangles are drawn

        # Account for object face layering (sort the queue)
        if canvas.layering: queue.sort(reverse=True)

        for face in queue: face.draw(canvas)


class ObjectIter:
    def __init__(self, obj):
        self._obj = obj
        self._index = 0

    def __next__(self):
        if self._index >= len(self._obj.data): raise StopIteration
        # ret = self._obj.face(i)   # TODO: Consider returning a copy vs original
        ret = self._obj.data[self._index]
        self._index += 1
        return ret


class Canvas(tkinter.Canvas):
    def __init__(self, parent, **args):
        super().__init__(parent, **args)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()
        self.verts = 0
        self.lastx = 0
        self.lasty = 0

        # TODO: Type checking and fancy insert stuff for objects here
        # self.objects = []
        self.object = None

        # View mode parameters
        self.outlinestr = "#000"
        self.culling = False
        self.fill = False
        self.layering = False
        self.shading = False
        self.POV = 0.0

        self.focus_set()

        self.bind("<Configure>", self.resize)
        self.bind("<Button-1>", self.click)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<MouseWheel>", self.zoom)
        self.bind("<Button-2>", self.debug)
        self.bind("<w>", self.togglewf)
        self.bind("<c>", self.toggleculling)
        self.bind("<f>", self.togglefill)
        self.bind("<l>", self.togglelayering)
        self.bind("<s>", self.toggleshading)
        self.bind("<p>", self.toggleview)

    # Called when the window is resized
    def resize(self, event):
        self.height = event.height
        self.width = event.width

        self.drawobjects()

    def click(self, event):
        self.lastx = event.x
        self.lasty = event.y

    def drag(self, event):
        deltax = (self.lastx - event.x) / -200
        self.lastx = event.x

        deltay = (self.lasty - event.y) / 200
        self.lasty = event.y

        # Stops the object from becoming flipped upside down
        if deltay > 0 and self.object.rx >= (math.pi / 2):
            deltay = 0
        elif deltay < 0 and self.object.rx <= (math.pi / -2):
            deltay = 0

        # Update rotation for all objects associated with canvas
        # for obj in self.objects: obj.rotate(deltay, deltax, 0)
        self.object.rotate(deltay, deltax, 0)

        # Redraw the objects
        self.drawobjects()

    def zoom(self, event):
        factor = 1
        if event.delta > 0:
            factor = 1.05
        elif event.delta < 0:
            factor = 0.95
        self.object.scale(factor, factor, factor)

        # Redraw the objects
        self.drawobjects()

    def togglewf(self, event):
        if self.fill and len(self.outlinestr) == 0:
            self.outlinestr = "#000"
            print("DISPLAY: Wireframe enabled")

        elif self.fill and len(self.outlinestr) >= 0:
            self.outlinestr = ""
            print("DISPLAY: Wireframe disabled")

        self.drawobjects()

    def toggleculling(self, event):
        if self.culling:
            self.culling = False
            print("DISPLAY: Culling disabled")

        else:
            self.culling = True
            print("DISPLAY: Culling enabled")

        self.drawobjects()

    def togglefill(self, event):
        if self.fill:
            self.fill = False
            if len(self.outlinestr) == 0: print("DISPLAY: Wireframe enabled")
            self.outlinestr = "#000"
            print("DISPLAY: Fill disabled")
            self.layering = False
            # print("DISPLAY: Layering disabled")

        else:
            self.fill = True
            print("DISPLAY: Fill enabled")

        self.drawobjects()

    def togglelayering(self, event):
        if self.fill and self.layering:
            self.layering = False
            print("DISPLAY: Layering disabled")

        elif self.fill:
            self.layering = True
            print("DISPLAY: Layering enabled")

        self.drawobjects()

    def toggleshading(self, event):
        if self.fill and self.shading:
            self.shading = False
            print("DISPLAY: Shading disabled")

        elif self.fill:
            self.shading = True
            print("DISPLAY: Shading enabled")

        self.drawobjects()

    def toggleview(self, event):
        if self.POV == 0.0:
            self.POV = 0.05
            print("DISPLAY: Perspective view")
        else:
            self.POV = 0.0
            print("DISPLAY: Isometric view")

        self.drawobjects()

    def debug(self, event):
        self.drawobjects()
        print(f"DEBUG: Canvas width is {self.width}")
        print(f"DEBUG: Canvas height is {self.height}")
        print(f"DEBUG: Object rotation is ({self.object.rx} rad, {self.object.ry} rad, {self.object.rz} rad)")
        # print(f"DEBUG: Object transform is ({self.object.tx}, {self.object.ty}, {self.object.tz})")
        print(f"DEBUG: Object scale is ({self.object.sx}, {self.object.sy}, {self.object.sz})")
        print(f"DEBUG: Rendered {self.verts} vertices\n")
        # print("DEBUG: Transform matrix:")

    # Draws all objects associated with the canvas
    def drawobjects(self):
        # Clear the canvas
        self.delete("all")
        self.verts = 0

        # 'Paint' the background (not included in count of vertices)
        self.create_polygon([(0, 0), (0, self.height), (self.width, self.height), (self.width, 0)], fill="#585C6C")

        # Draw the object(s)
        # for obj in self.objects: obj.draw(self)
        self.object.draw(self)


# File IO functions
# ParseSTL reads in a binary(!) STL file and returns a list of the important datapoints
def parseSTL(path):
    data = []
    with open(path, 'rb') as inf:
        # Skip the header
        inf.seek(80)

        # Get the number of triangles (unsigned 32-bit int, little endian)
        try:
            raw = inf.read(4)

            triangles = struct.unpack('<I', raw)
            data.append(triangles[0])

        except StopIteration:
            return data

        # Process the face values (32-bit single precision float, little endian)
        while True:
            try:
                nraw = inf.read(12)
                vraw = inf.read(36)
                extra = inf.read(2)

                normal = struct.unpack('<fff', nraw)
                verts = struct.unpack('<fffffffff', vraw)

                for vert in verts: data.append(vert)
                for vert in normal: data.append(vert)

            except (StopIteration, struct.error):
                break

    return data


if __name__ == "__main__":

    # If running from the terminal and wanting to view a file, the filepath must be passed within quotes
    # Example:
    #  ```bash
    #  python stlview.py "/path/to/models/probe.stl"
    #  ```
    objfile = None
    if len(sys.argv) < 2:
        # Run the program with a pre-built cube for testing
        pass  # debug()
    elif len(sys.argv) == 2:
        # Specify the file for the program to read in
        objfile = sys.argv[1]
    else:
        # Print error message and exit
        print("Usage: stlview.py [filename]")
        exit(-1)

    print("Welcome to STL View!")
    print((f"Using object file: {objfile}") if objfile else "No object file specified")
    print()

    # Window management API
    root = tkinter.Tk()
    root.title("[Python] STL View")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Canvas to draw points to (subcomponent of window 'root' object)
    canvas = Canvas(root)
    canvas.grid(column=0, row=0, sticky="NSEW")

    # Create the object to render
    obj = Object(objfile)
    obj.build()

    # Example transformations
    # obj.translate(1, 1, 0)
    # obj.scale(2, 2, 2)
    # obj.rotate(-0.3, 0, 0)

    # Associate that object with the program's canvas
    # canvas.objects.append(obj)
    canvas.object = obj
    canvas.drawobjects()

    # Create the window and event listener
    root.mainloop()
