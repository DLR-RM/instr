"""
Helper script to create config files for BlenderProc.
"""

import os
import yaml
import random
import numpy as np
import binascii

# these paths have to be manually set before creating a config
BLENDERPROC_ROOT = ''  # /path/to/BlenderProc
SHAPENET_ROOT = ''  # /path/to/ShapeNetCore.v2
SUNCG_ROOT = ''  # /path/to/suncg
DEST = ''  # /path/to/output_folder


def get_random_house_path():
    with open(os.path.join(BLENDERPROC_ROOT, 'suncg_houses.txt'), 'r') as f:
        house_paths = f.readlines()
    return os.path.join(SUNCG_ROOT, random.choice(house_paths)).strip()


def get_base_cfg():
    with open(os.path.join(BLENDERPROC_ROOT, 'base_config.yaml'), 'r') as f:
        base_cfg = yaml.load(f)
    return base_cfg


def get_random_obj_configs(n=10):
    obj_configs, scale_configs, mat_configs, sample_configs, physic_configs, gravoff_configs = [], [], [], [], [], []
    with open(os.path.join(BLENDERPROC_ROOT, 'shapenet_objects.txt'), 'r') as f:
        obj_paths = f.readlines()
    for i in range(n):
        scale = np.random.uniform(0.1, 0.4)
        recalculate_uv = np.random.uniform(0., 1.)
        obj_base_cfg = {
            "module": "loader.CustomObjectLoader",
            "config": {
                "path": os.path.join(SHAPENET_ROOT, random.choice(obj_paths)[:-1]),
                "scale": [scale, scale, scale],
                "add_properties": {
                    "cp_object_to_scale": True,
                    "cp_sample_pose": True,
                    "cp_category_id": int(i+2),
                    "cp_coarse_grained_class": "selected_object",
                    "cp_type": "Object",
                    "cp_physics": True,
                    "cp_cc_texture": True
                },
            }
        }
        scale_base_cfg = {
            "module": "manipulators.EntityManipulator",
                "config": {
                    "selector": {
                        "provider": "getter.Entity",
                        "conditions": {
                            "cp_category_id": int(i+2),
                        }
                    },
                "scale": [scale, scale, scale],
                "cf_add_modifier": {
                    "name": "Solidify",
                    "thickness": 0.0025
                },
                "cf_randomize_materials": {
                    "randomization_level": 1.,
                    "materials_to_replace_with": {
                        "provider": "getter.Material",
                        "conditions": {
                            "cp_is_cc_texture": True
                        }
                    }
                },
            }
        }
        mat_base_cfg = {
            "module": "manipulators.MaterialManipulator",
            "config": {
                "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_category_id": int(i + 2),
                    }
                },
                "cf_set_Roughness": {
                    "provider": "sampler.Value",
                    "type": "float",
                    "min": 0.05,
                    "max": 0.5,
                },
                "cf_set_Specular": {
                    "provider": "sampler.Value",
                    "type": "float",
                    "min": 0.5,
                    "max": 1.0,
                },
                "cf_color_link_to_displacement": {
                    "provider": "sampler.Value",
                    "type": "float",
                    "min": 0.001,
                    "max": 0.15
                },
                "cf_set_Alpha": 1.0,
                "mode": "once_for_each"
            }
        }
        sampler_base_cfg = {
            "module": "object.OnSurfaceSampler",
            "config": {
                "objects_to_sample": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_category_id": int(i+2)
                    }
                },
                "surface": {
                    "provider": "getter.Entity",
                    "index": 0,
                    "conditions": {
                        "name": "selected_table"
                    }
                },
                "pos_sampler": {
                    "provider": "sampler.UpperRegionSampler",
                    "to_sample_on": {
                        "provider": "getter.Entity",
                        "index": 0,
                        "conditions": {
                            "name": "selected_table"
                        }
                    },
                    "min_height": 1,
                    "max_height": 4,
                    "face_sample_range": [0.4, 0.6],
                    "use_ray_trace_check": False,
                },
                "min_distance": 0.1,
                "max_distance": 1.5,
                "rot_sampler": {
                    "provider": "sampler.Uniform3d",
                    "min": [0, 0, 0],
                    "max": [6.28, 6.28, 6.28]
                }
            }
        }
        physics_base_cfg = {
            "module": "object.PhysicsPositioning",
            "config": {
                "min_simulation_time": 0.5,
                "max_simulation_time": 2,
                "check_object_interval": 1,
            }
        }
        grav_off_cfg = {
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_category_id": int(i + 2),
                    }
                },
                "cp_physics": False,
            }
        }

        scale_base_cfg["config"]["cf_add_uv_mapping"] = {
                                     "projection": "cylinder",
                                     "forced_recalc_of_uv_maps": True if recalculate_uv > 0.5 else False
                                 }
        mat_base_cfg["config"]["cf_add_uv_mapping"] = {
                                     "projection": "cylinder",
                                     "forced_recalc_of_uv_maps": True if recalculate_uv > 0.5 else False
                                 }

        obj_configs.append(obj_base_cfg)
        scale_configs.append(scale_base_cfg)
        mat_configs.append(mat_base_cfg)
        sample_configs.append(sampler_base_cfg)
        physic_configs.append(physics_base_cfg)
        gravoff_configs.append(grav_off_cfg)
    return obj_configs, scale_configs, mat_configs, sample_configs, physic_configs, gravoff_configs


def create_config():
    base_cfg = get_base_cfg()

    baseline = 0.065
    focal_length_x = 541.14
    focal_length_y = 541.14

    base_cfg['modules'][8]['config']['intrinsics']['interocular_distance'] = baseline
    base_cfg['modules'][8]['config']['intrinsics']['cam_K'] = [focal_length_x, 0.0, 320.0, 0.0, focal_length_y, 240.0, 0.0, 0.0, 1.0]

    # add objects
    num_objs = np.random.randint(5, 12)
    obj_configs, scale_configs, mat_configs, sample_configs, physic_configs, gravoff_configs = get_random_obj_configs(n=num_objs)
    for obj_config, scale_config, mat_config, sample_config, physics_config, gravoff_config in zip(obj_configs, scale_configs, mat_configs, sample_configs, physic_configs, gravoff_configs):
        base_cfg['modules'].insert(6, obj_config)
        base_cfg['modules'].insert(7, scale_config)
        base_cfg['modules'].insert(8, sample_config)
        base_cfg['modules'].insert(9, physics_config)
        base_cfg['modules'].insert(10, gravoff_config)

    # set house path
    base_cfg['modules'][1]['config']['path'] = get_random_house_path()

    # replace house with cctextures
    house_cc_texture_config = {
      "module": "manipulators.EntityManipulator",
      "config": {
        "selector": {
          "provider": "getter.Entity",
          "conditions": {
            "type": "MESH"
          }
        },
        "cf_randomize_materials": {
          "randomization_level": 0.4,
          "materials_to_replace_with": {
            "provider": "getter.Material",
            "random_samples": 1,
            "conditions": {
              "cp_is_cc_texture": True  # this will return one random loaded cc textures
            }
          }
        }
      }
    }
    base_cfg['modules'].insert(4, house_cc_texture_config)

    # set output dir
    output_prefix = os.urandom(20)
    output_prefix = binascii.hexlify(output_prefix)
    output_prefix = str(output_prefix)[2:-1]
    output_path = os.path.join(DEST, output_prefix)
    os.makedirs(output_path)
    base_cfg['modules'][0]['config']['global']['output_dir'] = output_path

    with open(os.path.join(DEST, output_prefix + '/config.yaml'), 'w') as f:
        yaml.dump(base_cfg, f)
    return os.path.join(DEST, output_prefix + '/config.yaml')


if __name__ == '__main__':
    path = create_config()
    print(path)
