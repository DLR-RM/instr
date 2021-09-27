"""
To be placed into {BlenderProcRoot}/src/loader/.
"""

import bpy

from src.loader.LoaderInterface import LoaderInterface
from src.utility.Utility import Utility
from src.utility.LabelIdMapping import LabelIdMapping


class CustomObjectLoader(LoaderInterface):
    """
    Custom object loader which, in addition to the default ObjectLoader, also corrects materials (basically a merged
    ObjectLoader and ShapeNetLoader).
    Adapted from {BlenderProcRoot}/src/loader/ObjectLoader.py and {BlenderProcRoot}/src/loader/ShapeNetLoader.py.
    """
    def __init__(self, config):
        LoaderInterface.__init__(self, config)

    def run(self):
        if self.config.has_param('path') and self.config.has_param('paths'):
            raise Exception("Objectloader can not use path and paths in the same module!")
        if self.config.has_param('path'):
            file_path = Utility.resolve_path(self.config.get_string("path"))
            loaded_obj = Utility.import_objects(filepath=file_path)
        elif self.config.has_param('paths'):
            file_paths = self.config.get_list('paths')
            loaded_obj = []
            # the file paths are mapped here to object names
            cache_objects = {}
            for file_path in file_paths:
                resolved_file_path = Utility.resolve_path(file_path)
                current_objects = Utility.import_objects(filepath=resolved_file_path, cached_objects=cache_objects)
                loaded_obj.extend(current_objects)
        else:
            raise Exception("Loader module needs either a path or paths config value")

        if not loaded_obj:
            raise Exception("No objects have been loaded here, check the config.")

        self._correct_materials(loaded_obj)

        self._set_properties(loaded_obj)

        if "void" in LabelIdMapping.label_id_map:  # Check if using an id map
            for obj in loaded_obj:
                obj['category_id'] = LabelIdMapping.label_id_map["void"]

        # removes the x axis rotation found in all ShapeNet objects, this is caused by importing .obj files
        # the object has the same pose as before, just that the rotation_euler is now [0, 0, 0]
        LoaderInterface.remove_x_axis_rotation(loaded_obj)

        # move the origin of the object to the world origin and on top of the X-Y plane
        # makes it easier to place them later on, this does not change the `.location`
        LoaderInterface.move_obj_origin_to_bottom_mean_point(loaded_obj)
        bpy.ops.object.select_all(action='DESELECT')

        # Set the add_properties of all imported objects
        self._set_properties(loaded_obj)

    def _correct_materials(self, objects):
        """
        If the used material contains an alpha texture, the alpha texture has to be flipped to be correct

        :param objects: objects where the material maybe wrong
        """

        for obj in objects:
            for mat_slot in obj.material_slots:
                material = mat_slot.material
                nodes = material.node_tree.nodes
                links = material.node_tree.links
                texture_nodes = Utility.get_nodes_with_type(nodes, "ShaderNodeTexImage")
                if texture_nodes and len(texture_nodes) > 1:
                    principled_bsdf = Utility.get_the_one_node_with_type(nodes, "BsdfPrincipled")
                    # find the image texture node which is connect to alpha
                    node_connected_to_the_alpha = None
                    for node_links in principled_bsdf.inputs["Alpha"].links:
                        if "ShaderNodeTexImage" in node_links.from_node.bl_idname:
                            node_connected_to_the_alpha = node_links.from_node
                    # if a node was found which is connected to the alpha node, add an invert between the two
                    if node_connected_to_the_alpha is not None:
                        invert_node = nodes.new("ShaderNodeInvert")
                        invert_node.inputs["Fac"].default_value = 1.0
                        Utility.insert_node_instead_existing_link(links, node_connected_to_the_alpha.outputs["Color"],
                                                                  invert_node.inputs["Color"],
                                                                  invert_node.outputs["Color"],
                                                                  principled_bsdf.inputs["Alpha"])
