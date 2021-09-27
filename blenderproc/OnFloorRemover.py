"""
To be placed into {BlenderProcRoot}/src/object/.
"""

import bpy
import mathutils
import numpy as np

from src.main.Module import Module
from src.utility.BlenderUtility import check_intersection, check_bb_intersection, get_bounds


class OnFloorRemover(Module):
    """
    Simple module to delete objects which have dropped onto the floor after physics simulation.
    """

    def __init__(self, config):
        Module.__init__(self, config)
        self.up_direction = config.get_vector3d("up_direction", mathutils.Vector([0., 0., 1.])).normalized()
        self.surface = None

    def run(self):
        for obj in bpy.data.objects:
            if obj.name == 'selected_table':
                self.surface = obj
        if self.surface is None:
            f"Room does not contain the selected table anymore"
            exit(0)

        bpy.ops.object.select_all(action='DESELECT')
        objects_to_delete = []
        print(f"Table location: {self.surface.location[2], self.surface.location[2] + self.surface.dimensions[2]}")

        table_bb = get_bounds(self.surface)
        table_bb = np.array(table_bb)
        t0 = table_bb.min(axis=0)
        tx0, ty0, tz0 = t0
        t1 = table_bb.max(axis=0)
        tx1, ty1, table_height = t1
        for obj in bpy.data.objects:
            if 'coarse_grained_class' in obj:
                if obj['coarse_grained_class'] == 'selected_object':
                    bb = get_bounds(obj)
                    bb_center = np.mean(bb, axis=0)
                    if bb_center[-1] < table_height:
                        objects_to_delete.append(obj)
                        print(f"Deleting {obj.name}, {bb_center} because it's below the table")
                    elif (not (tx0 < bb_center[0] < tx1)) or (not (ty0 < bb_center[1] < ty1)):
                        objects_to_delete.append(obj)
                        print(f"Deleting {obj.name}, {bb_center} because it's outside the table's bounds")
                    else:
                        print(f"Leaving {obj}, {bb_center}")
        for obj in bpy.context.scene.objects:
            obj.select_set(False)
        for obj in objects_to_delete:
            obj.select_set(True)
        bpy.ops.object.delete()
