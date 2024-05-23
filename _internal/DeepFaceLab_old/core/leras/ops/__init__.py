import numpy as np
from core.leras import nn
tf = nn.tf
from tensorflow.python.ops import array_ops, random_ops, math_ops, sparse_ops, gradients
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.util import nest

def recompute_grad(call):  # 定义一个名为 recompute_grad 的装饰器函数，它接受一个参数 call，这个参数是被装饰的函数或方法。
    def inner(self, *inputs, **kwargs):  # 定义了内部函数 inner，它接收一个可能的实例 self，任意数量的输入参数 inputs 和关键字参数 kwargs。
        is_function = tf.is_tensor(self)  # 检查 self 是否是一个 TensorFlow 张量。这用于区分传递给装饰器的是一个普通函数还是类的方法。
        @tf.custom_gradient  # 这是一个装饰器，用于定义一个操作的正向传递和自定义梯度计算。
        def call_and_grad(*inputs):  # 定义了一个内嵌函数 call_and_grad，它接收任意数量的输入参数。
            def kernel_call():  # 定义了一个内部函数 kernel_call，用于执行被装饰的函数或方法的调用。
                if is_function:  # 根据 is_function 的值来决定是直接调用 call 函数（对于普通函数），还是以 self 作为第一个参数调用 call（对于类的方法）。
                    return call(*inputs, **kwargs)
                else:
                    return call(self, *inputs, **kwargs)
            outputs = kernel_call()  # 调用 kernel_call 函数并获取其输出。
            def grad_fn(*doutputs, variables=None):  # 定义了用于计算梯度的函数 grad_fn。它接收梯度的输出 doutputs 和一个可选的 variables 参数。
                with tf.control_dependencies(doutputs):  # 创建一个控制依赖环境，确保在计算梯度之前，doutputs 已经被计算。
                    outputs = kernel_call()  # 再次调用 kernel_call 以确保所有操作都是最新的。
                watches = nest.flatten(inputs)  # 将输入参数 inputs 展平，以便于在梯度计算中使用。
                if variables is not None:  # 如果 variables 参数不为空，将其也加入到梯度计算的观察列表中。
                    watches += nest.flatten(variables)
                grads = tf.gradients(outputs, watches, doutputs)
                # 使用 TensorFlow 的 gradients 函数计算 outputs 相对于 watches 的梯度，其中 doutputs 是上游传来的梯度。
                return grads[:len(inputs)], grads[len(inputs):]  # 返回分割后的梯度。前半部分是输入的梯度，后半部分（如果存在）是变量的梯度。
                # grad_fn 函数返回的是两部分梯度：grads[:len(inputs)] 和 grads[len(inputs):]。这种分割的原因在于 TensorFlow 的 tf.gradients 函数的工作方式，以及如何处理输入和可训练变量的梯度。
                # TensorFlow的tf.gradients函数：
                # tf.gradients 函数计算给定输出（outputs）相对于一系列观测值（watches）的梯度。在这种情况下，观测值包括输入参数（inputs）和可能的额外可训练变量（variables）。
                # 为什么要分割梯度：
                # grads 是一个列表，其中包含了相对于每个观测值的梯度。这个列表首先包含对应于输入参数的梯度，然后是对应于可训练变量的梯度（如果有的话）。
                # grads[:len(inputs)] 表示梯度列表中前 len(inputs) 个元素，这些是对应于函数输入参数的梯度。
                # grads[len(inputs):] 表示梯度列表中从 len(inputs) 开始的剩余元素，这些是对应于可训练变量（如果有的话）的梯度。
                # 应用场景：
                # 在自定义梯度计算时，通常需要区分对输入参数的梯度和对可训练变量的梯度。这种分割使得函数能够正确地将梯度传递回上游操作，并同时更新可训练变量的梯度（如果存在的话）。
            return outputs, grad_fn
        if is_function:
            return call_and_grad(self, *inputs)
        else:
            return call_and_grad(*inputs)
    return inner

nn.recompute_grad = recompute_grad

'''
tf.gradients(ys, xs, 
			 grad_ys=None, 
			 name='gradients',
			 colocate_gradients_with_ops=False,
			 gate_gradients=False,
			 aggregation_method=None,
			 stop_gradients=None)                  
原文链接：https://blog.csdn.net/hustqb/article/details/80260002
对求导函数而言，其主要功能即求导公式：∂ y /∂ x 。在tensorflow中，y 和x 都是tensor。
更进一步，tf.gradients()接受求导值ys和xs不仅可以是tensor，还可以是list，形如[tensor1, tensor2, …, tensorn]。当ys和xs都是list时，它们的求导关系为：
gradients() adds ops to the graph to output the derivatives of ys with respect to xs. It returns a list of Tensor of length len(xs) where each tensor is the sum(dy/dx) for y in ys. 即gradients() 函数向计算图中添加操作，用于输出 ys 相对于 xs 的导数。它返回一个长度为 len(xs) 的张量列表，其中每个张量是对于 ys 中的每个 y 的 dy/dx 之和。
意思是：1. tf.gradients()实现ys对xs求导
2. 求导返回值是一个list，list的长度等于len(xs)
3. 假设返回值是[grad1, grad2, grad3]，ys=[y1, y2]，xs=[x1, x2, x3]。则，真实的计算过程为:
g r a d 1 = ∂y 1/∂x 1 + ∂y 2/∂x 1 
g r a d 2 = ∂y 1/∂x 2 + ∂y 2/∂x 2 
g r a d 3 = ∂y 1/∂x 3 + ∂y 2/∂x 3
'''

def tf_get_value(tensor):
    return nn.tf_sess.run (tensor)
nn.tf_get_value = tf_get_value


def batch_set_value(tuples):
    if len(tuples) != 0:
        with nn.tf.device('/CPU:0'):
            assign_ops = []
            feed_dict = {}

            for x, value in tuples:
                if isinstance(value, nn.tf.Operation) or \
                    isinstance(value, nn.tf.Variable):
                    assign_ops.append(value)
                else:
                    value = np.asarray(value, dtype=x.dtype.as_numpy_dtype)
                    assign_placeholder = nn.tf.placeholder( x.dtype.base_dtype, shape=[None]*value.ndim )
                    assign_op = nn.tf.assign (x, assign_placeholder )
                    assign_ops.append(assign_op)
                    feed_dict[assign_placeholder] = value

            nn.tf_sess.run(assign_ops, feed_dict=feed_dict)
nn.batch_set_value = batch_set_value

def init_weights(weights):
    ops = []

    ca_tuples_w = []
    ca_tuples = []
    for w in weights:
        initializer = w.initializer
        for input in initializer.inputs:
            if "_cai_" in input.name:
                ca_tuples_w.append (w)
                ca_tuples.append ( (w.shape.as_list(), w.dtype.as_numpy_dtype) )
                break
        else:
            ops.append (initializer)

    if len(ops) != 0:
        nn.tf_sess.run (ops)

    if len(ca_tuples) != 0:
        nn.batch_set_value( [*zip(ca_tuples_w, nn.initializers.ca.generate_batch (ca_tuples))] )
nn.init_weights = init_weights

def tf_gradients ( loss, vars ):
    grads = gradients.gradients(loss, vars, colocate_gradients_with_ops=True )
    gv = [*zip(grads,vars)]
    for g,v in gv:
        if g is None:
            raise Exception(f"Variable {v.name} is declared as trainable, but no tensors flow through it.")
    return gv
nn.gradients = tf_gradients

def average_gv_list(grad_var_list, tf_device_string=None):
    if len(grad_var_list) == 1:
        return grad_var_list[0]

    e = tf.device(tf_device_string) if tf_device_string is not None else None
    if e is not None: e.__enter__()
    result = []
    for i, (gv) in enumerate(grad_var_list):
        for j,(g,v) in enumerate(gv):
            g = tf.expand_dims(g, 0)
            if i == 0:
                result += [ [[g], v]  ]
            else:
                result[j][0] += [g]

    for i,(gs,v) in enumerate(result):
        result[i] = ( tf.reduce_mean( tf.concat (gs, 0), 0 ), v )
    if e is not None: e.__exit__(None,None,None)
    return result
nn.average_gv_list = average_gv_list

def average_tensor_list(tensors_list, tf_device_string=None):
    if len(tensors_list) == 1:
        return tensors_list[0]

    e = tf.device(tf_device_string) if tf_device_string is not None else None
    if e is not None: e.__enter__()
    result = tf.reduce_mean(tf.concat ([tf.expand_dims(t, 0) for t in tensors_list], 0), 0)
    if e is not None: e.__exit__(None,None,None)
    return result
nn.average_tensor_list = average_tensor_list

def concat (tensors_list, axis):
    """
    Better version.
    """
    if len(tensors_list) == 1:
        return tensors_list[0]
    return tf.concat(tensors_list, axis)
nn.concat = concat

def gelu(x):
    cdf = 0.5 * (1.0 + tf.nn.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
nn.gelu = gelu

def upsample2d(x, size=2):
    if nn.data_format == "NCHW":
        x = tf.transpose(x, (0,2,3,1))
        x = tf.image.resize_nearest_neighbor(x, (x.shape[1]*size, x.shape[2]*size) )
        x = tf.transpose(x, (0,3,1,2))
        
        
        # b,c,h,w = x.shape.as_list()
        # x = tf.reshape (x, (-1,c,h,1,w,1) )
        # x = tf.tile(x, (1,1,1,size,1,size) )
        # x = tf.reshape (x, (-1,c,h*size,w*size) )
        return x
    else:
        return tf.image.resize_nearest_neighbor(x, (x.shape[1]*size, x.shape[2]*size) )
nn.upsample2d = upsample2d

def resize2d_bilinear(x, size=2):
    h = x.shape[nn.conv2d_spatial_axes[0]].value
    w = x.shape[nn.conv2d_spatial_axes[1]].value

    if nn.data_format == "NCHW":
        x = tf.transpose(x, (0,2,3,1))

    if size > 0:
        new_size = (h*size,w*size)
    else:
        new_size = (h//-size,w//-size)

    x = tf.image.resize(x, new_size, method=tf.image.ResizeMethod.BILINEAR)

    if nn.data_format == "NCHW":
        x = tf.transpose(x, (0,3,1,2))

    return x
nn.resize2d_bilinear = resize2d_bilinear

def resize2d_nearest(x, size=2):
    if size in [-1,0,1]:
        return x


    if size > 0:
        raise Exception("")
    else:
        if nn.data_format == "NCHW":
            x = x[:,:,::-size,::-size]
        else:
            x = x[:,::-size,::-size,:]
    return x

    h = x.shape[nn.conv2d_spatial_axes[0]].value
    w = x.shape[nn.conv2d_spatial_axes[1]].value

    if nn.data_format == "NCHW":
        x = tf.transpose(x, (0,2,3,1))

    if size > 0:
        new_size = (h*size,w*size)
    else:
        new_size = (h//-size,w//-size)

    x = tf.image.resize(x, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if nn.data_format == "NCHW":
        x = tf.transpose(x, (0,3,1,2))

    return x
nn.resize2d_nearest = resize2d_nearest

def flatten(x):
    if nn.data_format == "NHWC":
        # match NCHW version in order to switch data_format without problems
        x = tf.transpose(x, (0,3,1,2) )
    return tf.reshape (x, (-1, np.prod(x.shape[1:])) )

nn.flatten = flatten

def max_pool(x, kernel_size=2, strides=2):
    if nn.data_format == "NHWC":
        return tf.nn.max_pool(x, [1,kernel_size,kernel_size,1], [1,strides,strides,1], 'SAME', data_format=nn.data_format)
    else:
        return tf.nn.max_pool(x, [1,1,kernel_size,kernel_size], [1,1,strides,strides], 'SAME', data_format=nn.data_format)

nn.max_pool = max_pool

def reshape_4D(x, w,h,c):
    if nn.data_format == "NHWC":
        # match NCHW version in order to switch data_format without problems
        x = tf.reshape (x, (-1,c,h,w))
        x = tf.transpose(x, (0,2,3,1) )
        return x
    else:
        return tf.reshape (x, (-1,c,h,w))
nn.reshape_4D = reshape_4D

def random_binomial(shape, p=0.0, dtype=None, seed=None):
    if dtype is None:
        dtype=tf.float32

    if seed is None:
        seed = np.random.randint(10e6)
    return array_ops.where(
        random_ops.random_uniform(shape, dtype=tf.float16, seed=seed) < p,
             array_ops.ones(shape, dtype=dtype), array_ops.zeros(shape, dtype=dtype))
nn.random_binomial = random_binomial

def gaussian_blur(input, radius=2.0):
    def gaussian(x, mu, sigma):
        return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

    def make_kernel(sigma):
        kernel_size = max(3, int(2 * 2 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mean = np.floor(0.5 * kernel_size)
        kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
        np_kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
        kernel = np_kernel / np.sum(np_kernel)
        return kernel, kernel_size

    gauss_kernel, kernel_size = make_kernel(radius)
    padding = kernel_size//2
    if padding != 0:
        if nn.data_format == "NHWC":
            padding = [ [0,0], [padding,padding], [padding,padding], [0,0] ]
        else:
            padding = [ [0,0], [0,0], [padding,padding], [padding,padding] ]
    else:
        padding = None
    gauss_kernel = gauss_kernel[:,:,None,None]

    x = input
    k = tf.tile (gauss_kernel, (1,1,x.shape[nn.conv2d_ch_axis],1) )
    x = tf.pad(x, padding )
    x = tf.nn.depthwise_conv2d(x, k, strides=[1,1,1,1], padding='VALID', data_format=nn.data_format)
    return x
nn.gaussian_blur = gaussian_blur

def style_loss(target, style, gaussian_blur_radius=0.0, loss_weight=1.0, step_size=1):
    def sd(content, style, loss_weight):
        content_nc = content.shape[ nn.conv2d_ch_axis ]
        style_nc = style.shape[nn.conv2d_ch_axis]
        if content_nc != style_nc:
            raise Exception("style_loss() content_nc != style_nc")
        c_mean, c_var = tf.nn.moments(content, axes=nn.conv2d_spatial_axes, keep_dims=True)
        s_mean, s_var = tf.nn.moments(style, axes=nn.conv2d_spatial_axes, keep_dims=True)
        c_std, s_std = tf.sqrt(c_var + 1e-5), tf.sqrt(s_var + 1e-5)
        mean_loss = tf.reduce_sum(tf.square(c_mean-s_mean), axis=[1,2,3])
        std_loss  = tf.reduce_sum(tf.square(c_std-s_std), axis=[1,2,3])
        return (mean_loss + std_loss) * ( loss_weight / content_nc.value )

    if gaussian_blur_radius > 0.0:
        target = gaussian_blur(target, gaussian_blur_radius)
        style = gaussian_blur(style, gaussian_blur_radius)

    return sd( target, style, loss_weight=loss_weight )

nn.style_loss = style_loss
@nn.recompute_grad

def dssim(img1,img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    if img1.dtype != img2.dtype:
        raise ValueError("img1.dtype != img2.dtype")

    not_float32 = img1.dtype != tf.float32

    if not_float32:
        img_dtype = img1.dtype
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.cast(img2, tf.float32)

    filter_size = max(1, filter_size)

    kernel = np.arange(0, filter_size, dtype=np.float32)
    kernel -= (filter_size - 1 ) / 2.0
    kernel = kernel**2
    kernel *= ( -0.5 / (filter_sigma**2) )
    kernel = np.reshape (kernel, (1,-1)) + np.reshape(kernel, (-1,1) )
    kernel = tf.constant ( np.reshape (kernel, (1,-1)), dtype=tf.float32 )
    kernel = tf.nn.softmax(kernel)
    kernel = tf.reshape (kernel, (filter_size, filter_size, 1, 1))
    kernel = tf.tile (kernel, (1,1, img1.shape[ nn.conv2d_ch_axis ] ,1))

    def reducer(x):
        return tf.nn.depthwise_conv2d(x, kernel, strides=[1,1,1,1], padding='VALID', data_format=nn.data_format)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mean0 = reducer(img1)
    mean1 = reducer(img2)
    num0 = mean0 * mean1 * 2.0
    den0 = tf.square(mean0) + tf.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(img1 * img2) * 2.0
    den1 = reducer(tf.square(img1) + tf.square(img2))
    c2 *= 1.0 #compensation factor
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    ssim_val = tf.reduce_mean(luminance * cs, axis=nn.conv2d_spatial_axes )
    dssim = (1.0 - ssim_val ) / 2.0

    if not_float32:
        dssim = tf.cast(dssim, img_dtype)
    return dssim

nn.dssim = dssim

def space_to_depth(x, size):
    if nn.data_format == "NHWC":
        # match NCHW version in order to switch data_format without problems
        b,h,w,c = x.shape.as_list()
        oh, ow = h // size, w // size
        x = tf.reshape(x, (-1, size, oh, size, ow, c))
        x = tf.transpose(x, (0, 2, 4, 1, 3, 5))
        x = tf.reshape(x, (-1, oh, ow, size* size* c ))
        return x
    else:
        return tf.space_to_depth(x, size, data_format=nn.data_format)
nn.space_to_depth = space_to_depth

def depth_to_space(x, size):
    if nn.data_format == "NHWC":
        # match NCHW version in order to switch data_format without problems

        b,h,w,c = x.shape.as_list()
        oh, ow = h * size, w * size
        oc = c // (size * size)

        x = tf.reshape(x, (-1, h, w, size, size, oc, ) )
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        x = tf.reshape(x, (-1, oh, ow, oc, ))
        return x
    else:
        cfg = nn.getCurrentDeviceConfig()
        if not cfg.cpu_only:
            return tf.depth_to_space(x, size, data_format=nn.data_format)
        b,c,h,w = x.shape.as_list()
        oh, ow = h * size, w * size
        oc = c // (size * size)

        x = tf.reshape(x, (-1, size, size, oc, h, w, ) )
        x = tf.transpose(x, (0, 3, 4, 1, 5, 2))
        x = tf.reshape(x, (-1, oc, oh, ow))
        return x
nn.depth_to_space = depth_to_space

def rgb_to_lab(srgb):
    # 这段代码定义了一个函数 rgb_to_lab，它的目的是将RGB颜色空间的像素值转换为CIELAB（简称Lab）颜色空间的值。CIELAB颜色空间是一种颜色表示方法，广泛用于颜色管理和图像处理领域，因为它更接近人类视觉感知，并可以描述所有可见的颜色。下面是对这段代码的详细解释：
    srgb_pixels = tf.reshape(srgb, [-1, 3])
    linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
    exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    rgb_to_xyz = tf.constant([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334], # R
        [0.357580, 0.715160, 0.119193], # G
        [0.180423, 0.072169, 0.950227], # B
    ])
    xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

    xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

    epsilon = 6/29
    linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
    exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

    fxfyfz_to_lab = tf.constant([
        #  l       a       b
        [  0.0,  500.0,    0.0], # fx
        [116.0, -500.0,  200.0], # fy
        [  0.0,    0.0, -200.0], # fz
    ])
    lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])
    return tf.reshape(lab_pixels, tf.shape(srgb))
nn.rgb_to_lab = rgb_to_lab

def total_variation_mse(images):
    """
    Same as generic total_variation, but MSE diff instead of MAE
    """
    pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
    pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
    
    tot_var = ( tf.reduce_sum(tf.square(pixel_dif1), axis=[1,2,3]) +
                tf.reduce_sum(tf.square(pixel_dif2), axis=[1,2,3]) )
    return tot_var
nn.total_variation_mse = total_variation_mse


def pixel_norm(x, axes):
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axes, keepdims=True) + 1e-06)
nn.pixel_norm = pixel_norm
'''
def pixel_norm(x, axes):

这行定义了一个函数 pixel_norm，它接受两个参数：x 和 axes。x 是输入张量，通常是神经网络层的输出；axes 是一个整数或整数列表，指明了要在 x 中进行归一化操作的轴。
tf.square(x)

这个表达式计算 x 的每个元素的平方。这是归一化过程的第一步，用于计算输入张量各元素的平方值。
tf.reduce_mean(tf.square(x), axis=axes, keepdims=True)

这里，tf.reduce_mean 计算 x 的平方沿着指定 axes 的平均值。keepdims=True 参数确保输出的维度与输入 x 相同（除了在 axes 上进行平均的维度）。
tf.rsqrt(... + 1e-06)

tf.rsqrt 函数计算其输入的逆平方根。这里的输入是 tf.reduce_mean 的输出加上一个很小的数（1e-06），以避免除以零的情况。这个小的数值是为了提高数值稳定性。
return x * tf.rsqrt(...)

最后，原始输入 x 与 tf.rsqrt 的结果相乘。这个操作实际上是在对每个像素进行归一化，使其具有单位方差。这样的归一化方法通常有助于防止网络在训练过程中出现数值问题。
综上所述，pixel_norm 函数通过将输入张量的每个像素值除以其局部区域的均方根来归一化数据，从而帮助改善和稳定深度学习模型（特别是 GANs）的训练过程。

pixel_norm 函数不会将输出结果中的每个元素的值限制在0到1之间。这个函数的目的是对输入数据进行归一化，使每个元素的值与其周围值的方差相关。具体来说，它将每个元素的值除以其所在位置的局部均方根。这种归一化方法主要用于控制数据的分布，而不是将其限制在特定的范围内。

归一化的结果依赖于输入数据的特性。如果输入数据的局部区域具有较小的方差，归一化后的值可能会比较大；如果方差较大，归一化后的值可能会比较小。因此，归一化后的数据可能会超出 [0,1] 的范围。
'''


"""
def tf_suppress_lower_mean(t, eps=0.00001):
    if t.shape.ndims != 1:
        raise ValueError("tf_suppress_lower_mean: t rank must be 1")
    t_mean_eps = tf.reduce_mean(t) - eps
    q = tf.clip_by_value(t, t_mean_eps, tf.reduce_max(t) )
    q = tf.clip_by_value(q-t_mean_eps, 0, eps)
    q = q * (t/eps)
    return q
"""



def _get_pixel_value(img, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)
    
    return tf.gather_nd(img, indices)
    
def bilinear_sampler(img, x, y):
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    H_MAX = tf.cast(H - 1, tf.int32)
    W_MAX = tf.cast(W - 1, tf.int32)

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, 0, W_MAX)
    x1 = tf.clip_by_value(x1, 0, W_MAX)
    y0 = tf.clip_by_value(y0, 0, H_MAX)
    y1 = tf.clip_by_value(y1, 0, H_MAX)

    # get pixel value at corner coords
    Ia = _get_pixel_value(img, x0, y0)
    Ib = _get_pixel_value(img, x0, y1)
    Ic = _get_pixel_value(img, x1, y0)
    Id = _get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out
    
nn.bilinear_sampler = bilinear_sampler

