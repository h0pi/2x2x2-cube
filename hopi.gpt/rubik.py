import pyray as pr
import numpy as np
import random

class Cube:
    def __init__(self, size, center, face_color):
        self.size=size
        self.center=center
        self.face_color=face_color
        self.orientation = np.eye(3)


        self.model=None
        self.gen_meshe(size)
        #create and position meshes
        self.create_model()
    def gen_meshe(self, scale:tuple):
        # create the central cube mesh
        self.mesh = pr.gen_mesh_cube(*scale)
        
    def create_model(self):
        self.model = pr.load_model_from_mesh(self.mesh)
        self.model.transform = pr.matrix_translate(self.center[0], self.center[1], self.center[2])
    
    def rotate(self, axis, theta):
        # create the rotation matrix based on the specified axis
        if axis == 0:
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        elif axis ==1:
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif axis==2:
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        else: 
            raise ValueError("Invalid axis. Use 'x', 'y' or 'z'.")
        
        # apply the rotation matrix to update the position
        self.center = rotation_matrix @ self.center

        # update the orientation
        self.orientation = rotation_matrix @ self.orientation

    def get_rotation_axis_angle(self):
        # calculate the rotation axis and angle from orientation matrix 
        x = (np.trace(self.orientation) - 1) / 2
        x = np.clip(x, -1.0, 1.0)
        angle = np.arccos(x)
        if angle==0:
            axis = np.array([0, 0, 0])
        else:
            rx = self.orientation[2, 1] - self.orientation[1, 2]
            ry = self.orientation[0, 2] - self.orientation[2, 0]
            rz = self.orientation[1, 0] - self.orientation[0, 1]
            axis = np.array([rx, ry, rz])/(2 * np.sin(angle))
        axis = pr.Vector3(axis[0], axis[1], axis[2])
        return axis, np.degrees(angle)







class Rubik:
    def __init__(self) -> None:
        self.cubes=[]

        self.is_rotating=False
        self.rotation_angle = 0
        self.rotation_axis = None
        self.level = None
        self.segment = None
        self.target_rotation = 0
        self.generate_rubik(2)
    
    def generate_rubik(self, size):
        colors = [pr.WHITE, pr.YELLOW, pr.BLUE, pr.GREEN, pr.ORANGE, pr.RED]
        offset = size-0.7
        size_z=size*0.9, size*0.9, size*0.1
        size_x=size*0.9, size*0.1, size*0.9
        size_y=size*0.1, size*0.9, size*0.9
        n=2

        self.cubes=[]

        for x in range(n):
            for y in range(n):
                for z in range(n):
                    face_colors = [
                        pr.BLACK if z != (n-1) else colors[0],  # Front
                        pr.BLACK if z != 0     else colors[1],  # Back
                        pr.BLACK if x != (n-1) else colors[2],  # Right
                        pr.BLACK if x != 0     else colors[3],  # Left
                        pr.BLACK if y != (n-1) else colors[4],  # Top
                        pr.BLACK if y != 0     else colors[5]   # Bottom
                    ]

                    # centriranje za n=2: koordinate budu ±0.5*offset
                    cx = (x - (n-1)/2) * offset
                    cy = (y - (n-1)/2) * offset
                    cz = (z - (n-1)/2) * offset
                    center_position = np.array([cx, cy, cz])

                    center = Cube((size, size, size), center_position, pr.BLACK)

                    front_position  = np.array([cx, cy, cz + size/2])
                    back_position   = np.array([cx, cy, cz - size/2])
                    right_position  = np.array([cx + size/2, cy, cz])
                    left_position   = np.array([cx - size/2, cy, cz])
                    top_position    = np.array([cx, cy + size/2, cz])
                    bottom_position = np.array([cx, cy - size/2, cz])

                    front  = Cube(size_z, front_position,  face_colors[0])
                    back   = Cube(size_z, back_position,   face_colors[1])
                    right  = Cube(size_y, right_position,  face_colors[2])
                    left   = Cube(size_y, left_position,   face_colors[3])
                    top    = Cube(size_x, top_position,    face_colors[4])
                    bottom = Cube(size_x, bottom_position, face_colors[5])

                    self.cubes.append([center, front, back, left, right, top, bottom])

        return self.cubes

    def chose_piece(self, piece, axis_index, level):
        v = round(piece[0].center[axis_index], 1)

        # 2x2: samo 2 sloja
        if level == 0:
            return v < 0
        elif level == 1:
            return v > 0

        return False

    
    def get_face(self, axis, level):
        axis_index= np.nonzero(axis)[0][0]
        segment = [i for i, cube in enumerate(self.cubes)\
                   if self.chose_piece(cube, axis_index, level)]
        return segment


    def handle_rotation(self, rotation_queue, animation_step=None):
        # check if there is a request and if not already rotating
        if rotation_queue and not self.is_rotating:

            # get the next rotation axis and level
            self.target_rotation, self.rotation_axis, self.level = rotation_queue.pop(0)

            if self.target_rotation>0:
                self.target_rotation += random.uniform(0,1) * 10**-3
            else:
                self.target_rotation -= random.uniform(0, 1)* 10**-3
            
            self.segment = self.get_face(self.rotation_axis, self.level)

            # reset rotation angle at the start of a new rotation
            self.rotation_angle =0 

            # set rotating to true to start rotation
            self.is_rotating = True
        
        if self.is_rotating:
            if(self.rotation_angle != self.target_rotation):
                diff = abs(self.target_rotation - self.rotation_angle)
                delta_angle = min(np.radians(5), diff)

                # increment the rotation angle in the correct direction
                self.rotation_angle += delta_angle if self.target_rotation >0\
                                        else -delta_angle
            else:
                delta_angle =0

                # stop rotating when target rotation is reached
                self.is_rotating=False

                if animation_step is not None:
                    animation_step+=1
            
            for id, cube in enumerate(self.cubes):
                axis_index = np.nonzero(self.rotation_axis)[0][0]

                if id in self.segment:
                    for part_id, _ in enumerate(cube):
                        if self.target_rotation >0:
                            self.cubes[id][part_id].rotate(axis_index, delta_angle)
                        else:
                            self.cubes[id][part_id].rotate(axis_index, -delta_angle)
                        
                        pos_x, pos_y, pos_z = self.cubes[id][part_id].center

                        translation = pr.matrix_translate(pos_x, pos_y, pos_z)
                        rota, angle =self.cubes[id][part_id].get_rotation_axis_angle()
                        rotation = pr.matrix_rotate(rota, np.radians(angle))
                        transform = pr.matrix_multiply(rotation, translation)
                        self.cubes[id][part_id].model.transform = transform
        #else:
            #self.is_rotating = True
        
        return rotation_queue, animation_step
