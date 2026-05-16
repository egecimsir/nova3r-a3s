from PIL import Image
import os
import numpy as np
import torch
import json
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.geometry import ldi_to_pts3d, filter_pts3d_with_intrinsics
import os.path as osp
import cv2

def sample_gt_point_cloud(pts3d, valid_mask,num_pts):
    # sample the complete point cloud set
    data = {}
    valid_pts3d = pts3d[valid_mask] # N, 3
    n_valid_pts3d = valid_pts3d.shape[0]

    if n_valid_pts3d > num_pts:
        perm = torch.randperm(n_valid_pts3d)
        sampled_points = valid_pts3d[perm[:num_pts]]
        data["pcd_eval"] = sampled_points
    else:
        perm = torch.randint(0, n_valid_pts3d, (num_pts,))
        data["pcd_eval"] = valid_pts3d[perm]    

    # sample the point cloud from unseen layers
    valid_pts3d_behind = pts3d[:,:,1:,:][valid_mask[:,:,1:]] # N', 3
    n_vpts3d_behind = valid_pts3d_behind.shape[0]
    # sample the point cloud from the visible layer
    valid_pts3d_first = pts3d[:,:,:1,:][valid_mask[:,:,:1]] # N', 3
    n_vpts3d_first = valid_pts3d_first.shape[0]


    if n_vpts3d_behind >= num_pts:
        perm = torch.randperm(n_vpts3d_behind)
        data["pcd_eval_unseen"] = valid_pts3d_behind[perm[:num_pts]]
    else:
        perm = torch.randint(0, n_vpts3d_behind, (num_pts,))
        data["pcd_eval_unseen"] = valid_pts3d_behind[perm]            

    if n_vpts3d_first >= num_pts:
        perm = torch.randperm(n_vpts3d_first)
        data["pcd_eval_visible"] = valid_pts3d_first[perm[:num_pts]]
    else:
        perm = torch.randint(0, n_vpts3d_first, (num_pts,))
        data["pcd_eval_visible"] = valid_pts3d_first[perm]     

    return data


class SCRREAM_MULTI(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, data_path=None, train_list_path=None, test_list_path=None, n_ldi_layers=10,
                 num_pts=100000, max_pts=20000, input_n=1, enforce_img_reso_for_eval=[512,512], use_eval=None, **kwargs):
        # Handle backward compatibility
        if data_path is not None:
            ROOT = data_path
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        
        self.dataset_label = 'SCRREAM'
        self.input_n = input_n  # Single image evaluation
        self.num_views = input_n

        self.enforce_img_reso_for_eval = enforce_img_reso_for_eval
        # Legacy parameters for backward compatibility
        self.data_list_path_dict = {"train": train_list_path, "test": test_list_path}
        self.intrinsic = None
        self.num_pts = num_pts
        self.n_ldi_layers = n_ldi_layers
        self.max_pts = max_pts
        self.loaded_data = self._load_data()
        self.resolution_lari = (1132, 874)
        self.invalid_layer_pix_ratio = 0.03
        self.use_eval = use_eval

    def _load_data(self):
        with open(self.data_list_path_dict[self.split], "tr") as f:
            self._data_list = json.load(f) # "BREAKFAST_MENU"


    def __len__(self):
        return len(self._data_list)

    def _load_data_list_legacy(self):
        """Legacy data loading method"""
        if self.data_list_path_dict and self.data_list_path_dict[self.split]:
            with open(self.data_list_path_dict[self.split], "r") as f:
                self._data_list = json.load(f)
        else:
            # Scan directory structure to create image list
            self._data_list = self._create_image_list_from_directory()

    def __len__(self):
        if hasattr(self, 'image_indices'):
            return len(self.image_indices)
        else:
            return len(self._data_list)

    def _load_intrinsics(self, scene_folder, output_tensor=True):
        """Loads the camera intrinsic matrix from intrinsics.txt"""
        intrinsics_file = os.path.join(scene_folder, "intrinsics.txt")

        if not os.path.exists(intrinsics_file):
            raise FileNotFoundError(f"Error: {intrinsics_file} does not exist!")

        intrinsics = []
        with open(intrinsics_file, "r") as file:
            for line in file:
                row = list(map(float, line.strip().split()))
                intrinsics.append(row)

        if output_tensor:
            intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32).unsqueeze(0)
            return intrinsics_tensor
        else:
            intrinsics = np.array(intrinsics).astype(np.float32)
            return intrinsics

    def _depth_to_pointcloud(self, depthmap, intrinsics, camera_pose=None):
        """Convert depth map to point cloud"""
        H, W = depthmap.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1] 
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Valid depth mask
        valid_mask = (depthmap > 0) & np.isfinite(depthmap)
        
        if not np.any(valid_mask):
            return np.zeros((0, 3), dtype=np.float32)
        
        # Get valid pixels
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depthmap[valid_mask]
        
        # Back-project to 3D
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid
        
        pts3d = np.stack([x, y, z], axis=1)
        
        if camera_pose is not None:
            # Transform to world coordinates
            pts3d_homogeneous = np.hstack([pts3d, np.ones((len(pts3d), 1))])
            pts3d = (camera_pose @ pts3d_homogeneous.T).T[:, :3]
        
        return pts3d.astype(np.float32)
    

    def adjust_resolution(self, image_ori, ldi, intrinsic):
        image, ldi, intrinsic_lari = self.resize_and_completion(image_ori, ldi, intrinsic.copy(), self.resolution_lari[0], self.resolution_lari[1])
        if isinstance(self.enforce_img_reso_for_eval, list):
            # For our evaluation, only resize image into fixed resolution (eg, 512x512) while keeping gt the original resolution
            image, _, intrinsic_img = self.resize_and_completion(image_ori, None, intrinsic.copy(), 
                                                        self.enforce_img_reso_for_eval[0], self.enforce_img_reso_for_eval[1])
            
        return image, ldi, intrinsic_lari, intrinsic_img

    def resize_and_completion(self, image, ldi, intrinsic, target_w, target_h):
        ''' 
        Given an image (in PIL Image format), a numpy array 'ldi' with shape (H, W, L),
        and an intrinsic matrix (3x3), resize all while maintaining aspect ratio,
        then complete the short side using a gray color for the image and -1 for 'ldi'.
        Adjust intrinsic matrix accordingly.
        '''

        w, h = image.size
        
        # Compute new size while maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image while maintaining aspect ratio
        image = image.resize((new_w, new_h), Image.BICUBIC)

        new_image = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        
        # Paste resized image onto the center of the new image
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        intrinsic[0, :] *= scale  # Scale focal length and principal point in x direction
        intrinsic[1, :] *= scale  # Scale focal length and principal point in y direction
        intrinsic[0, 2] += paste_x  # Adjust principal point for x-axis padding
        intrinsic[1, 2] += paste_y  # Adjust principal point for y-axis padding
        
        if ldi is not None and intrinsic is not None:
            # Convert ldi to a PyTorch tensor
            ldi_tensor = torch.tensor(ldi, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, L, H, W)
            
            # Resize ldi using PyTorch interpolation
            ldi_resized = torch.nn.functional.interpolate(ldi_tensor, size=(new_h, new_w), mode='nearest')
            # Create a new ldi tensor with the required resolution and fill missing values with -1
            new_ldi = torch.full((1, ldi.shape[2], target_h, target_w), -1, dtype=torch.float32)
            # Paste resized ldi onto the center of the new ldi tensor
            new_ldi[:, :, paste_y:paste_y + new_h, paste_x:paste_x + new_w] = ldi_resized

            # Convert back to numpy
            new_ldi = new_ldi.squeeze(0).permute(1, 2, 0).numpy()
            # mask = mask.squeeze(0).permute(1, 2, 0).numpy()

                
            # Adjust intrinsic matrix
          

            return new_image, new_ldi, intrinsic
        else:
            return new_image, None, intrinsic

    def filter_out_invalid_layers(self, valid_mask):
        '''
        to filter out layers with extremely small valid areas (such as smaller than 3% of that of the first layer)
        by marking the layered mask to zero
        '''
        area_first_layer = valid_mask[:,:,0].sum()
        area = np.sum(np.reshape(valid_mask, (-1, valid_mask.shape[-1])), axis=0) # L) 

        valid_layer_index = (area > self.invalid_layer_pix_ratio * area_first_layer)[None, None, ...] # 1 1 L
        res = (valid_mask * valid_layer_index).astype(bool) # B H L
        return res 


    def _parse_frame_entry(self, frame_entry):
        """Parse frame entry to extract scene, sequence, and frame ID."""
        parts = frame_entry.strip().split()
        if len(parts) == 2:
            # New format: "scene01/scene01_full_00 35"
            scene_seq_path = parts[0]
            frame_id = int(parts[1])
            return scene_seq_path, frame_id
        elif len(parts) >= 3:
            # Multi-view format: "scene01/scene01_full_00 35 50" or "scene01/scene01_full_00 35 50 60 70"
            scene_seq_path = parts[0]
            frame_ids = [int(x) for x in parts[1:]]
            return scene_seq_path, frame_ids
        else:
            raise ValueError(f"Invalid frame entry format: {frame_entry}")


    def _get_views(self, idx, resolution, rng):
        views = []

        frame_entries = self._data_list[idx]
        if isinstance(frame_entries, str):
            # Handle the multi-frame format: "scene01/scene01_full_00 35 50 60 70"
            scene_seq_path, frame_ids = self._parse_frame_entry(frame_entries)
            if isinstance(frame_ids, int):
                frame_ids = [frame_ids]
        # Ensure we don't exceed the requested number of views
        if len(frame_ids) >= self.input_n:
            frame_ids = frame_ids[:self.input_n]
        else:
            assert False, "Number of available frames is less than input_n, which is not supported."

        for view_idx, frame_id in enumerate(frame_ids):
            view_data = self._load_single_view(scene_seq_path, frame_id, view_idx, resolution, rng)
            views.append(view_data)

        return views

    def _load_single_view(self, obj_path, img_id, view_idx, resolution, rng):

        view_label = f'input_{view_idx}'

        rgb_image = Image.open(os.path.join(self.ROOT, obj_path, "rgb", "{:06d}.png".format(img_id))).convert("RGB")

        ldi = np.load(os.path.join(self.ROOT, obj_path, "ldi", "{:06d}_ldi.npz".format(img_id)))["ldi"]

        intrinsics_path = os.path.join(self.ROOT, obj_path)
        intrinsics = self._load_intrinsics(intrinsics_path, output_tensor=False)

        cam_path = os.path.join(self.ROOT, obj_path, "camera_pose", "{:06d}.txt".format(img_id))
        camera_pose = np.loadtxt(cam_path).astype(np.float32)

        rgb_image, ldi, intrinsics_lari, intrinsic_img = self.adjust_resolution(rgb_image, ldi, intrinsics)

        pts3d_lari, mask = ldi_to_pts3d(ldi, intrinsics_lari)
        lari_data = sample_gt_point_cloud(pts3d_lari, mask, self.max_pts // self.input_n)

        mask = self.filter_out_invalid_layers(mask)

        pts3d_lari[~mask] = 0
        pts3d = pts3d_lari[mask].reshape(-1, 3)
        pts3d = (camera_pose[:3, :3] @ pts3d.T).T + camera_pose[:3, 3]

        mask = mask[...,None]

        depthmap = ldi[:,:,0]

        valid_num = np.array(pts3d.shape[0], dtype=np.int32)

        max_pts = self.max_pts
        if pts3d.shape[0] > max_pts:
            # Random sample to max_pts points
            indices = rng.choice(pts3d.shape[0], max_pts, replace=False)
            pts3d = pts3d[indices]
            valid_num = np.array(max_pts)

        else:
            # Pad with zeros
            padding = np.zeros((max_pts - pts3d.shape[0], 3), dtype=pts3d.dtype)
            valid_num = np.array(pts3d.shape[0])

            pts3d = np.vstack([pts3d, padding])

        pcd_eval = lari_data["pcd_eval"].astype(np.float32)
        pcd_eval_unseen = lari_data["pcd_eval_unseen"].astype(np.float32)
        pcd_eval_visible = lari_data["pcd_eval_visible"].astype(np.float32)

        pcd_eval = (camera_pose[:3, :3] @ pcd_eval.T).T + camera_pose[:3, 3]
        pcd_eval_unseen = (camera_pose[:3, :3] @ pcd_eval_unseen.T).T + camera_pose[:3, 3]
        pcd_eval_visible = (camera_pose[:3, :3] @ pcd_eval_visible.T).T + camera_pose[:3, 3]

        view_data = dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),
            camera_intrinsics=intrinsic_img.astype(np.float32),
            dataset=self.dataset_label,
            label=obj_path,
            instance=f'{img_id}',
            view_label=view_label,
            pts3d_complete=pts3d.astype(np.float32),  # complete point cloud
            pts3d_complete_valid_num=valid_num.astype(np.int32),  # number of valid points in the complete point cloud,
            mask=mask.astype(np.bool_),  # valid mask for the complete point cloud
            pcd_eval=pcd_eval.astype(np.float32),
            pcd_eval_unseen=pcd_eval_unseen.astype(np.float32),
            pcd_eval_visible=pcd_eval_visible.astype(np.float32),
            pts3d_lari=pts3d_lari.astype(np.float32),
        )
        
        return view_data

    def _get_image_and_ldi(self, idx):
        """Legacy method for backward compatibility"""
        item = self._data_list[idx].split(" ")
        obj_path, img_id = item
        img_id = int(img_id)

        try:
            img = Image.open(os.path.join(self.ROOT, obj_path, "rgb", "{:06d}.png".format(img_id))).convert("RGB") 
            
            ldi_path = os.path.join(self.ROOT, obj_path, "ldi", "{:06d}_ldi.npz".format(img_id))
            if os.path.exists(ldi_path):
                ldi = np.load(ldi_path)["ldi"]
                if hasattr(self, 'n_ldi_layers'):
                    ldi = ldi[:,:,:self.n_ldi_layers]
            else:
                ldi = None
            
            intrinsics_path = os.path.join(self.ROOT, obj_path)
            intrinsics = self._load_intrinsics(intrinsics_path, output_tensor=False)

        except Exception as e:
            print("[ERROR] data load error at path: {}, Error: {}".format(os.path.join(self.ROOT, obj_path), e))
            raise

        sample_name = "{} {}".format(obj_path, img_id)
        return img, ldi, None, sample_name, intrinsics
