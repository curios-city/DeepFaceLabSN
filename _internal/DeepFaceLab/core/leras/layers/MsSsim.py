from core.leras import nn
tf = nn.tf


class MsSsim(nn.LayerBase):
    default_power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    default_l1_alpha = 0.84

    def __init__(self, batch_size, in_ch, resolution, kernel_size=11, use_l1=False, **kwargs):
        # restrict mssim factors to those greater/equal to kernel size
        power_factors = [p for i, p in enumerate(self.default_power_factors) if resolution//(2**i) >= kernel_size]
        # normalize power factors if reduced because of size
        if sum(power_factors) < 1.0:
            power_factors = [x/sum(power_factors) for x in power_factors]
        self.power_factors = power_factors
        self.num_scale = len(power_factors)
        self.kernel_size = kernel_size
        self.use_l1 = use_l1
        if use_l1:
            self.gaussian_weights = nn.get_gaussian_weights(batch_size, in_ch, resolution, num_scale=self.num_scale)

        super().__init__(**kwargs)

    def __call__(self, y_true, y_pred, max_val):
        # Transpose images from NCHW to NHWC
        y_true_t = tf.transpose(tf.cast(y_true, tf.float32), [0, 2, 3, 1])
        y_pred_t = tf.transpose(tf.cast(y_pred, tf.float32), [0, 2, 3, 1])

        # ssim_multiscale returns values in range [0, 1] (where 1 is completely identical)
        # subtract from 1 to get loss
        if tf.__version__ >= "1.14":
            ms_ssim_loss = 1.0 - tf.image.ssim_multiscale(y_true_t, y_pred_t, max_val, power_factors=self.power_factors, filter_size=self.kernel_size)
        else:
            ms_ssim_loss = 1.0 - tf.image.ssim_multiscale(y_true_t, y_pred_t, max_val, power_factors=self.power_factors)

        # If use L1 is enabled, use mix of ms-ssim and L1 (weighted by gaussian filters)
        # H. Zhao, O. Gallo, I. Frosio and J. Kautz, "Loss Functions for Image Restoration With Neural Networks,"
        # in IEEE Transactions on Computational Imaging, vol. 3, no. 1, pp. 47-57, March 2017,
        # doi: 10.1109/TCI.2016.2644865.
        # https://research.nvidia.com/publication/loss-functions-image-restoration-neural-networks

        if self.use_l1:
            diff = tf.tile(tf.expand_dims(tf.abs(y_true - y_pred), axis=0), multiples=[self.num_scale, 1, 1, 1, 1])
            l1_loss = tf.reduce_mean(tf.reduce_sum(self.gaussian_weights[-1, :, :, :, :] * diff, axis=[0, 3, 4]), axis=[1])
            return self.default_l1_alpha * ms_ssim_loss + (1 - self.default_l1_alpha) * l1_loss

        return ms_ssim_loss


nn.MsSsim = MsSsim
