from trax import layers as tl
from trax.fastmath import numpy as jnp
def DotProductAttention(query, key, value, mask):
  depth=query.shape[-1]
  dots=jnp.matmul(query,jnp.swapaxes(key,-1,-2))/jnp.sqrt(depth)#计算scaled dotproductattention此时矩阵的大小为（batch，seq，seq）行代表第几个q列代表第几个k，q=k
  jnp.matmul(dots,value)
  if mask is not None:
    dots=dots+jnp.where(jnp.where(mask,dots,jnp.full_like(dots, -1e9)))#根据布尔矩阵的掩膜将矩阵正确的处理，布尔矩阵中为1的位置为原值
  #此时矩阵还不能直接使用因为计算出来的是注意力的值我们还需要进行softmax使其分布在0和1之间
  logsumexp = trax.fastmath.logsumexp(dots, axis=-1, keepdims=True)#指明计算方向
  dots = jnp.exp( dots-logsumexp)#我们使用logsoftmax
  attention=jnp.matmul(dots,value)#回到初始大小
  return attention


def dot_product_self_attention(q, k, v):  # q,k,v通过sequential自动传入不要初始化它
    mask_size = q.shape[1]
    mask = jnp.tril(jnp.ones(1, mask_size, mask_size), k=0)  # 创建mask，别忘了mask外的batch维，1在这里代表的就是batch维度
    mask_dot_attention = DotProductAttention(q, k, v, mask)
    return mask_dot_attention
#用闭包的方式在初始化时记住n_heads和d_head内部的函数的x不需要参数，x是sequential模型自己传入的
def compute_attention_heads_closure(n_heads, d_head):#n_heads是头的数量，d_head是每个头的大小。在论文中为了保证性能
  def compute_attention_heads(x):#x的形状是batch，seq，embedding_size(n_head*n_size)
    batch=x.shape[0]
    seq=x.shape[1]
    x=jnp.reshape(x,[batch,seq,n_heads,d_head])#变形到batch，seq，n_head，n_size
    x=jnp.transpose(x,(0,2,1,3))#变形为batch，n_size，seq，n_head
    x=jnp.reshape(x,[-1,seq,d_head])#之前两步就是为了现在，巧妙的地方在于我们将多头的计算变成了batch的计算，充分利用了并行计算，-1是默认参数的意思jnp会自动计算出正确的值
    return x
  return compute_attention_heads
#通过batch的方式计算完多头注意力后我们需要将其输出为正确的形状我们还是以闭包的方式写
def compute_attention_output_closure(n_heads, d_head):
  def compute_attention_output(x):#此时x的形状为(batch*n_heads,seq,d_head)我们需要它回到(batch，seq，embedding_size(n_head*n_size))的形状
    seq=x.shape[1]
    x=jnp.reshape(x,(-1,n_heads,seq,d_head))
    x=jnp.transpose(x,(0,2,1,3))
    x=jnp.reshape(x,(-1, seq, n_heads * d_head))
    return x
  return compute_attention_output


def CausalAttention(d_feature,
                    n_heads,
                    compute_attention_heads_closure=compute_attention_heads_closure,
                    dot_product_self_attention=dot_product_self_attention,
                    compute_attention_output_closure=compute_attention_output_closure,
                    mode='train'):
    """Transformer-style multi-headed causal attention.

     Args:
         d_feature (int):  dimensionality of feature embedding.
         n_heads (int): number of attention heads.
         mode (str): 'train' or 'eval'.
     Returns:
         trax.layers.combinators.Serial: Multi-headed self-attention model.
     """
    d_head = d_feature // n_heads
    ComputeAttentionHeads = tl.Fn('AttnHeads', compute_attention_heads_closure(n_heads, d_head), n_out=1)
    return tl.Serial(
        tl.Branch([tl.Dense(d_feature, ), ComputeAttentionHeads], [tl.Dense(d_feature, ), ComputeAttentionHeads]),
        # 这一层代码的意思是，1.x进来的形状是(batch，seq，embedded)，然后通过一个并行计算层Branch线性层得到三个输出的矩阵（也就是qkv）（一个[]代表一个并行计算），先通过一个线性层映射到不同的空间，再通过一个多头层变成多头状态"""
        tl.Fn('DotProductAttn', dot_product_self_attention, n_out=1),  # 输入是三个多头QKV矩阵
        tl.Fn('AttnOutput', compute_attention_output_closure(n_heads, d_head), n_out=1),
        # 变形为正常形状即(batch，seq，embedded)但是我们的输出是通过变形多头堆叠在一起的我们还需要一个线性层
        tl.Dense(d_feature))  # 回到输入大小
def DecoderBlock(d_model, d_ff, n_heads,dropout, mode, ff_activation):#通过CausalAttention构建decoder模块
  causal_attention = CausalAttention(
                        d_model,
                        n_heads=n_heads,
                        mode=mode
                        )
  feed_foward=[tl.LayerNorm(),#归一层加速训练
        tl.Dense(d_ff),#全连接层
        ff_activation(),#激活层]
        tl.Dropout(rate=dropout,mode=mode),#防止过拟合
        tl.Dense(d_model),#最后一次线性化
        tl.Dropout(rate=dropout,mode=mode)#防止过拟合
               ]
  return [
      tl.Residual(#两个残差连接层允许训练深度的网络不会出现梯度消失的问题
          tl.LayerNorm(),
         causal_attention,
          tl.Dropout(rate=dropout,mode=mode)
        ),
      tl.Residual(
          feed_foward
        ),]


def TransformerLM(vocab_size=33300,
                  d_model=512,
                  d_ff=2048,
                  n_layers=6,
                  n_heads=8,
                  dropout=0.1,
                  max_len=4096,
                  mode='train',
                  ff_activation=tl.Relu):
    positional_encoder = [  # 位置编码层
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, mode=mode),
        tl.PositionalEncoding(max_len=max_len, mode=mode)]
    decoder_blocks = [
        DecoderBlock(d_model, d_ff, n_heads, dropout, mode, ff_activation)  # 按论文上所写将Decoder重复6次
        for _ in range(n_layers)]
    return tl.Serial(
        tl.ShiftRight(mode=mode),  # 使用teacher force

        positional_encoder
        , decoder_blocks,
        tl.LayerNorm(),
        tl.Dense(vocab_size),  # 最后一个全连接层
        tl.LogSoftmax()  # 输出概率
    )









