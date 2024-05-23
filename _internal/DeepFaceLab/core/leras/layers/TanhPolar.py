import numpy as np
from core.leras import nn
tf = nn.tf

class TanhPolar(nn.LayerBase):
    """
    RoI Tanh-polar Transformer Network for Face Parsing in the Wild
    https://github.com/hhj1897/roi_tanh_warping
    """

    def __init__(self, width, height, angular_offset_deg=270, **kwargs):
        self.width = width
        self.height = height

        warp_gridx, warp_gridy = TanhPolar._get_tanh_polar_warp_grids(width,height,angular_offset_deg=angular_offset_deg)
        restore_gridx, restore_gridy = TanhPolar._get_tanh_polar_restore_grids(width,height,angular_offset_deg=angular_offset_deg)

        self.warp_gridx_t = tf.constant(warp_gridx[None, ...])
        self.warp_gridy_t = tf.constant(warp_gridy[None, ...])
        self.restore_gridx_t = tf.constant(restore_gridx[None, ...])
        self.restore_gridy_t = tf.constant(restore_gridy[None, ...])

        super().__init__(**kwargs)

    def warp(self, inp_t):
        batch_t = tf.shape(inp_t)[0]
        warp_gridx_t = tf.tile(self.warp_gridx_t, (batch_t,1,1) )
        warp_gridy_t = tf.tile(self.warp_gridy_t, (batch_t,1,1) )

        if nn.data_format == "NCHW":
            inp_t = tf.transpose(inp_t,(0,2,3,1))

        out_t = nn.bilinear_sampler(inp_t, warp_gridx_t, warp_gridy_t)

        if nn.data_format == "NCHW":
            out_t = tf.transpose(out_t,(0,3,1,2))

        return out_t

    def restore(self, inp_t):
        batch_t = tf.shape(inp_t)[0]
        restore_gridx_t = tf.tile(self.restore_gridx_t, (batch_t,1,1) )
        restore_gridy_t = tf.tile(self.restore_gridy_t, (batch_t,1,1) )

        if nn.data_format == "NCHW":
            inp_t = tf.transpose(inp_t,(0,2,3,1))

        inp_t = tf.pad(inp_t, [(0,0), (1, 1), (1, 0), (0, 0)], "SYMMETRIC")

        out_t = nn.bilinear_sampler(inp_t, restore_gridx_t, restore_gridy_t)

        if nn.data_format == "NCHW":
            out_t = tf.transpose(out_t,(0,3,1,2))

        return out_t

    @staticmethod
    def _get_tanh_polar_warp_grids(W,H,angular_offset_deg):
        angular_offset_pi = angular_offset_deg * np.pi / 180.0

        roi_center = np.array([ W//2, H//2], np.float32 )
        roi_radii = np.array([W, H], np.float32 ) / np.pi ** 0.5
        cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi)
        normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / W),np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / H)), axis=-1)
        radii = normalised_dest_indices[..., 0]
        orientation_x = np.cos(normalised_dest_indices[..., 1])
        orientation_y = np.sin(normalised_dest_indices[..., 1])

        src_radii = np.arctanh(radii) * (roi_radii[0] * roi_radii[1] / np.sqrt(roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2))
        src_x_indices = src_radii * orientation_x
        src_y_indices = src_radii * orientation_y
        src_x_indices, src_y_indices = (roi_center[0] + cos_offset * src_x_indices - sin_offset * src_y_indices,
                                        roi_center[1] + cos_offset * src_y_indices + sin_offset * src_x_indices)

        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)

    @staticmethod
    def _get_tanh_polar_restore_grids(W,H,angular_offset_deg):
        angular_offset_pi = angular_offset_deg * np.pi / 180.0

        roi_center = np.array([ W//2, H//2], np.float32 )
        roi_radii = np.array([W, H], np.float32 ) / np.pi ** 0.5
        cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi)

        dest_indices = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(float)
        normalised_dest_indices = np.matmul(dest_indices - roi_center, np.array([[cos_offset, -sin_offset],
                                                                                [sin_offset, cos_offset]]))
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)
        normalised_dest_indices[..., 0] /= np.clip(radii, 1e-9, None)
        normalised_dest_indices[..., 1] /= np.clip(radii, 1e-9, None)
        radii *= np.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 +
                        roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1]

        src_radii = np.tanh(radii)


        src_x_indices = src_radii * W + 1.0
        src_y_indices = np.mod((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) /
                                2.0 / np.pi) * H, H) + 1.0

        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)


nn.TanhPolar = TanhPolar