"""
To be placed into {BlenderProcRoot}/src/object/.
"""

import random
import bpy
import mathutils
import numpy as np

from src.main.Module import Module
from src.utility.BlenderUtility import check_intersection, check_bb_intersection, get_bounds


class TableExtractor(Module):
    """
    Custom module to search for a table inside a SunCG house.
    """
    def __init__(self, config):
        Module.__init__(self, config)
        self.up_direction = config.get_vector3d("up_direction", mathutils.Vector([0., 0., 1.])).normalized()

    def run(self):
        tables = []
        for obj in bpy.data.objects:
            if 'coarse_grained_class' in obj:
                if obj['coarse_grained_class'] == 'table':
                    if obj.location[2] < 0.1:
                        tables.append(obj)

        if tables == []:
            print(f"Room has no tables. Exiting.")
            exit(0)

        table = random.choice(tables)
        table.name = 'selected_table'
        self.surface = table

        # delete all objects on top of the table, if any
        bpy.ops.object.select_all(action='DESELECT')
        objects_to_delete = []
        bb_table = np.array(get_bounds(self.surface))
        tx0, ty0, tz0 = bb_table.min(axis=0)
        tx1, ty1, table_height = bb_table.max(axis=0)

        for obj in bpy.data.objects:
            if obj == table:
                continue
            bb = get_bounds(obj)
            bb_center = np.mean(bb, axis=0)
            if (table_height + 0.5) > bb_center[-1] > np.mean([tz0, table_height]) and ((tx0 < bb_center[0] < tx1) and (ty0 < bb_center[1] < ty1)):
                print('Deleting', obj.name, obj['coarse_grained_class'] if 'coarse_grained_class' in obj.keys() else 'no class', bb_center, tx0, tx1, ty0, ty1, tz0, table_height)
                objects_to_delete.append(obj)

        for obj in bpy.context.scene.objects:
            obj.select_set(False)
        for obj in objects_to_delete:
            obj.select_set(True)
        bpy.ops.object.delete()

        # outcomment this line if you want to have the table as segmentation id 1
        #self._add_seg_properties([table])

    def check_above_surface(self, obj):
        """ Check if all corners of the bounding box are "above" the surface

        :param obj: Object for which the check is carried out. Type: blender object.
        :return: True if the bounding box is above the surface, False - if not.
        """
        inv_world_matrix = self.surface.matrix_world.inverted()

        for point in get_bounds(obj):
            ray_start = inv_world_matrix @ (point + self.up_direction)
            ray_direction = inv_world_matrix @ (self.surface.location + (-1 * self.up_direction))

            is_hit, hit_location, _, _ = self.surface.ray_cast(ray_start, ray_direction)

            if not is_hit:
                return False

        return True

    def _add_seg_properties(self, objects):
        """ Sets all custom properties of all given objects according to the configuration.

        :parameter objects: A list of objects which should receive the custom properties
        """

        properties = {"cp_category_id": 1}

        for obj in objects:
            for key, value in properties.items():
                if key.startswith("cp_"):
                    key = key[3:]
                    obj[key] = value
                else:
                    raise RuntimeError(
                        "Loader modules support setting only custom properties. Use 'cp_' prefix for keys. "
                        "Use manipulators.Entity for setting object's attribute values.")
