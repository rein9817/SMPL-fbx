import bpy
import os
import numpy as np
import sys
import json
from mathutils import Matrix, Vector, Quaternion
from math import radians


bone_name_from_index = {
    0: 'Pelvis',
    1: 'L_Hip',
    2: 'R_Hip',
    3: 'Spine1',
    4: 'L_Knee',
    5: 'R_Knee',
    6: 'Spine2',
    7: 'L_Ankle',
    8: 'R_Ankle',
    9: 'Spine3',
    10: 'L_Foot',
    11: 'R_Foot',
    12: 'Neck',
    13: 'L_Collar',
    14: 'R_Collar',
    15: 'Head',
    16: 'L_Shoulder',
    17: 'R_Shoulder',
    18: 'L_Elbow',
    19: 'R_Elbow',
    20: 'L_Wrist',
    21: 'R_Wrist',
    22: 'L_Hand',
    23: 'R_Hand'
}


fps_origin = 30  # origin data fps
fps_target = 30  # output to blender 
sample_rate = fps_origin / fps_target

male_model_path = 'model/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'
female_model_path = 'model/SMPL_f_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'


def load_data():
    data_path = "SMPL_data/smpl"
    pose_list = []
    shapes_list = []
    global_translations = []
    keypoint_data = []
    
    for file in os.listdir(data_path):
        with open(os.path.join(data_path, file), "r") as f:
            temp = json.load(f)
            global_translations.append(temp[0]["Th"][0])
            global_rotation = np.array(temp[0]["Rh"][0])
            poses = np.array(temp[0]["poses"][0])
            combined_pose = np.concatenate([global_rotation, poses])
            pose_list.append(combined_pose)
            shapes_list.append(temp[0]["shapes"][0])
    
    data_path = "SMPL_data/keypoints3d"
    for file in os.listdir(data_path):
        with open(os.path.join(data_path, file), "r") as f:
            temp = json.load(f)
            keypoint_data.append(temp[0]["keypoints3d"])

    return np.array(global_translations), np.array(pose_list), np.array(shapes_list), keypoint_data


def setup(model_path=male_model_path):
    scene = bpy.data.scenes["Scene"]
    scene.render.fps = fps_origin

    bpy.ops.object.mode_set(mode="OBJECT")

    if "Cube" in bpy.data.objects:
        cube = bpy.data.objects["Cube"]
        cube.select_set(True)
        bpy.ops.object.delete()

    bpy.ops.import_scene.fbx(filepath=model_path)


def Rodrigues(rotvec: np.ndarray):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec.reshape(3, 1)
    cost = np.cos(theta)
    mat = np.array([[0, -r[2, 0], r[1, 0]],
                   [r[2, 0], 0, -r[0, 0]],
                   [-r[1, 0], r[0, 0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)


def export_fbx(output_path):
    if not output_path:
        output_path = "output.fbx"
        
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not output_dir:
        output_path = os.path.join(".", output_path)

    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except:
        pass
    
    bpy.ops.object.select_all(action="DESELECT")

    armature = bpy.data.objects.get("Armature")
    if armature:
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        
        # 選擇 Armature 的 mesh
        if armature.children:
            armature.children[0].select_set(True)

    if output_path.endswith(".fbx"):
        print(f"Exporting to FBX: {output_path}")
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True, add_leaf_bones=False)
    elif output_path.endswith(".glb"):
        print(f"Exporting to glb: {output_path}")
        bpy.ops.export_scene.glb(filepath=output_path, use_selection=True, add_leaf_bones=False)
    else:
        print("Invalid output file format")
        sys.exit(1)


def process_poses(gender):
    global_translations, pose_list, shapes_list, keypoint_data = load_data()
    armature = bpy.data.objects["Armature"]
    # print(armature)
    
    if gender == "female":
        for key, value in bone_name_from_index.items():
            bone_name_from_index[key] = "f_avg_" + value
    elif gender == "male":
        for key, value in bone_name_from_index.items():
            bone_name_from_index[key] = "m_avg_" + value
    else:
        print("Invalid gender")
        sys.exit(1)

    scene = bpy.data.scenes["Scene"]
    sample_rate = int(fps_origin / fps_target)
    print("sample_rate", sample_rate)
    
    bpy.ops.object.mode_set(mode="EDIT")
    pelvis_position = Vector(bpy.data.armatures[0].edit_bones[bone_name_from_index[0]].head)
    bpy.ops.object.mode_set(mode="OBJECT")

    offset = np.zeros(3)

    origin_index = 0
    frame = 1
    while origin_index < len(pose_list):
        scene.frame_set(frame)
        process_pose(frame, pose_list[origin_index], global_translations[origin_index], pelvis_position)
        frame += 1
        origin_index += sample_rate
        
    bpy.ops.object.mode_set(mode="OBJECT")


def process_pose(frame, pose, trans, pelvis_position):
    armature = bpy.data.objects["Armature"]
    bones = armature.pose.bones
    
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except:
        pass
        
    if pose.shape[0] == 72:
        rod_rots = pose.reshape(24, 3)
    else:
        rod_rots = pose.reshape(26, 3)
        
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    
    # 設置骨盆位置
    bones[bone_name_from_index[0]].location = Vector((100*trans[1], 100*trans[2], 100*trans[0])) - pelvis_position
    bones[bone_name_from_index[0]].keyframe_insert('location', frame=frame)
    
    for index, mat_rot in enumerate(mat_rots, 0):
        if index >= 24:
            continue
            
        if bone_name_from_index[index] not in bones:
            print(f"Warning: Bone {bone_name_from_index[index]} not found in armature.")
            continue
            
        bone = bones[bone_name_from_index[index]]
        bone.rotation_mode = 'QUATERNION'
        
        bone_rotation = Matrix(mat_rot).to_quaternion()
        quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), radians(-90))
        quat_z_90_cw = Quaternion((0.0, 0.0, 1.0), radians(-90))
        
        if index == 0:
            bone.rotation_quaternion = (quat_x_90_cw @ quat_z_90_cw) @ bone_rotation
        else:
            bone.rotation_quaternion = bone_rotation
            
        bone.keyframe_insert('rotation_quaternion', frame=frame)


if __name__ == "__main__":
    setup()
    process_poses("male")
    export_fbx("output.fbx")