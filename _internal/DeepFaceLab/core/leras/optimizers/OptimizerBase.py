import copy
from core.leras import nn
tf = nn.tf

# 定义一个优化器的基类，继承自 nn.Saveable
class OptimizerBase(nn.Saveable):
    def __init__(self, name=None):
        # 调用父类构造函数
        super().__init__(name=name)

    # 定义一个用于梯度裁剪的方法
    def tf_clip_norm(self, g, c, n):
        """
        如果梯度 g 的 L2 范数 n 超过 c，就裁剪梯度 g。
        参数:
            g: Tensor, 梯度张量。
            c: float >= 0. 当梯度的 L2 范数超过这个值时，将进行裁剪。
            n: Tensor, g 的实际范数。
        返回:
            Tensor, 如果需要，返回裁剪后的梯度。
        """
		# 如果裁剪范数 c 小于等于 0，就不需要添加操作到计算图中
        if c <= 0:  # if clipnorm == 0 no need to add ops to the graph
            return g

        # 判断梯度范数是否超过阈值
        condition = n >= c
        # 如果超过阈值，则进行裁剪
        then_expression = tf.scalar_mul(c / n, g)
        # 否则，保持原样
        else_expression = g

        # 保存形状以避免将稀疏张量转换为密集张量
        if isinstance(then_expression, tf.Tensor):
            g_shape = copy.copy(then_expression.get_shape())
        elif isinstance(then_expression, tf.IndexedSlices):
            g_shape = copy.copy(then_expression.dense_shape)
        
        # 确保条件是布尔类型
        if condition.dtype != tf.bool:
            condition = tf.cast(condition, 'bool')
        
        # 根据条件选择执行裁剪还是保持原样
        g = tf.cond(condition,
                    lambda: then_expression,
                    lambda: else_expression)
        
        # 设置裁剪后的张量形状
        if isinstance(then_expression, tf.Tensor):
            g.set_shape(g_shape)
        elif isinstance(then_expression, tf.IndexedSlices):
            g._dense_shape = g_shape

        return g

# 将 OptimizerBase 类赋值给 nn 模块中的 OptimizerBase
nn.OptimizerBase = OptimizerBase
