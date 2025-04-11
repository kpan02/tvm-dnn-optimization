import os
import tvm
from tvm import te


def make_conv1d_cpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda i: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.any(i - k < 0, i - k >= M),
                tvm.tir.const(0.0, "float32"),
                A[i - k] * W[k]
            ),
            axis=k
        ),
        name="B"
    )

    s = te.create_schedule(B.op)

    # Optimization 1: Loop splitting
    s, inner_axis, outer_axis = conv1d_cpu_optim1(s, B)
    
    # Optimization 2: Vectorization
    s = conv1d_cpu_optim2(s, B, inner_axis)

    # Optimization 3: Parallel execution
    s = conv1d_cpu_optim3(s, B, outer_axis)

    return s, A, W, B


def conv1d_cpu_optim1(s, B):
    # Optimization 1: Loop splitting
    split_factor = 32
    i, = s[B].op.axis
    io, ii = s[B].split(i, factor=split_factor)
    s[B].reorder(io, ii)
    return s, ii, io  


def conv1d_cpu_optim2(s, B, inner_axis):
    # Optimization 2: Vectorization
    s[B].vectorize(inner_axis)
    return s


def conv1d_cpu_optim3(s, B, outer_axis):
    # Optimization 3: Parallel execution
    s[B].parallel(outer_axis)
    return s


def make_conv1d_gpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda i: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.any(i - k < 0, i - k >= M),
                tvm.tir.const(0.0, "float32"),
                A[i - k] * W[k]
            ),
            axis=k
        ),
        name="B"
    )

    s = te.create_schedule(B.op)
    k_axis = s[B].op.reduce_axis[0]
    
    # Optimization 1: Basic thread binding
    s, bx, tx = conv1d_gpu_optim1(s, B)

    # Optimization 2: Reorder axes
    s = conv1d_gpu_optim2(s, B, bx, tx, k_axis)

    # Optimization 3: Auto-unroll pragma
    s = conv1d_gpu_optim3(s, B, tx)

    return s, A, W, B


def conv1d_gpu_optim1(s, B):
    # Optimization 1: Basic thread binding
    i, = s[B].op.axis
    bx, tx = s[B].split(i, factor=128)
    s[B].bind(bx, te.thread_axis("blockIdx.x"))
    s[B].bind(tx, te.thread_axis("threadIdx.x"))
    return s, bx, tx


def conv1d_gpu_optim2(s, B, bx, tx, k):
    # Optimization 2: Reorder axes
    s[B].reorder(bx, tx, k)
    return s


def conv1d_gpu_optim3(s, B, tx):
   # Optimization 3: Auto-unroll pragma
    s[B].pragma(tx, "auto_unroll_max_step", 8)
    return s


def make_gemm_gpu_scheduler(M, K, N):
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), "k")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    
    # Default schedule
    s = te.create_schedule(C.op)
    
    # Optimization 1: Tiling
    s, *block_axes = gemm_gpu_optim1(s, C, A, B)
    
    # Optimization 2: Shared Memory
    s = gemm_gpu_optim2(s, C, A, B, block_axes)
    
    # Optimization 3: Vectorization
    s = gemm_gpu_optim3(s, C, A, B, block_axes)
    
    return s, A, B, C


def gemm_gpu_optim1(s, C, A, B):
    # Optimization 1: Tiling
    x, y = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    
    tile_x = 16
    tile_y = 16
    tile_k = 8
    
    # Tile & reorder axes
    x_outer, x_inner = s[C].split(x, factor=tile_x)
    y_outer, y_inner = s[C].split(y, factor=tile_y)
    k_outer, k_inner = s[C].split(k, factor=tile_k)
    s[C].reorder(x_outer, y_outer, k_outer, x_inner, y_inner, k_inner)
    
    # Map to GPU thread hierarchy
    s[C].bind(x_outer, te.thread_axis("blockIdx.x"))
    s[C].bind(y_outer, te.thread_axis("blockIdx.y"))
    s[C].bind(x_inner, te.thread_axis("threadIdx.x"))
    s[C].bind(y_inner, te.thread_axis("threadIdx.y"))
    
    return s, x_outer, y_outer, k_outer, x_inner, y_inner, k_inner


def gemm_gpu_optim2(s, C, A, B, block_axes):
    # Optimization 2: Shared Memory
    x_outer, y_outer, k_outer, x_inner, y_inner, k_inner = block_axes
    
    A_shared = s.cache_read(A, "shared", [C])
    B_shared = s.cache_read(B, "shared", [C])
    
    s[A_shared].compute_at(s[C], k_outer)
    s[B_shared].compute_at(s[C], k_outer)
    
    aa_x, aa_y = s[A_shared].op.axis
    aa_fused = s[A_shared].fuse(aa_x, aa_y)
    tx, ty = s[C].op.axis
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    
    aa_outer, aa_inner = s[A_shared].split(aa_fused, factor=16)
    s[A_shared].bind(aa_inner, thread_x)
    
    bb_x, bb_y = s[B_shared].op.axis
    bb_fused = s[B_shared].fuse(bb_x, bb_y)
    bb_outer, bb_inner = s[B_shared].split(bb_fused, factor=16)
    s[B_shared].bind(bb_inner, thread_x)
    
    return s


def gemm_gpu_optim3(s, C, A, B, block_axes):
    # Optimization 3: Vectorization
    x_outer, y_outer, k_outer, x_inner, y_inner, k_inner = block_axes
    
    for stage in s.stages:
        if isinstance(stage.op, tvm.te.ComputeOp):
            if stage.op.name == 'A_shared':
                axes = s[stage].op.axis
                if len(axes) >= 2:
                    fused = s[stage].fuse(*axes)
                    outer, inner = s[stage].split(fused, factor=4)
                    s[stage].vectorize(inner)
                
            elif stage.op.name == 'B_shared':
                axes = s[stage].op.axis
                if len(axes) >= 2:
                    fused = s[stage].fuse(*axes)
                    outer, inner = s[stage].split(fused, factor=4)
                    s[stage].vectorize(inner)
    
    s[C].unroll(k_inner)
    return s


def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")
    pad = (K - 1) // 2
    
    padded = te.compute(
        (B, C, H + 2*pad, W + 2*pad),
        lambda b, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(h >= pad, h < H + pad, w >= pad, w < W + pad),
            inp[b, c, h - pad, w - pad],
            tvm.tir.const(0.0, "float32")
        ),
        name="padded"
    )
    
    r_h = te.reduce_axis((0, K), name="r_h")
    r_w = te.reduce_axis((0, K), name="r_w")
    
    # Depthwise convolution 
    out = te.compute(
        (B, C, H, W),
        lambda b, c, h, w: te.sum(
            padded[b, c, h + r_h, w + r_w] * ker[c, 0, r_h, r_w],
            axis=[r_h, r_w]
        ),
        name="out"
    )
    
    # Create schedule
    s = te.create_schedule(out.op)
    
    # Optimization 1: Tiling and Thread Binding
    s, block_axis = dwsp_conv2d_gpu_optim1(s, out, inp, ker, padded)
    
    # Optimization 2: Vectorization and Unrolling
    s = dwsp_conv2d_gpu_optim2(s, out, inp, ker, padded, block_axis)
    
    # Optimization 3: Loop reordering
    s = dwsp_conv2d_gpu_optim3(s, out, inp, ker, padded, block_axis)
    
    return s, inp, ker, out


def dwsp_conv2d_gpu_optim1(s, out, inp, ker, padded):
    # Optimization 1: Tiling and Thread Binding
    
    b, c, h, w = s[out].op.axis
    r_h, r_w = s[out].op.reduce_axis
    tile_size = 16
    
    b_outer, b_inner = s[out].split(b, factor=1)
    c_outer, c_inner = s[out].split(c, factor=4)
    h_outer, h_inner = s[out].split(h, factor=tile_size)
    w_outer, w_inner = s[out].split(w, factor=tile_size)
    s[out].reorder(b_outer, c_outer, h_outer, w_outer, b_inner, c_inner, h_inner, w_inner, r_h, r_w)
    
    # Bind to thread hierarchy
    s[out].bind(b_outer, te.thread_axis("blockIdx.z"))
    s[out].bind(c_outer, te.thread_axis("blockIdx.y"))
    fused_hw = s[out].fuse(h_outer, w_outer)
    s[out].bind(fused_hw, te.thread_axis("blockIdx.x"))
    
    fused_c_inner = s[out].fuse(b_inner, c_inner)
    s[out].bind(fused_c_inner, te.thread_axis("threadIdx.z"))
    s[out].bind(h_inner, te.thread_axis("threadIdx.y"))
    s[out].bind(w_inner, te.thread_axis("threadIdx.x"))
    
    return s, fused_hw


def dwsp_conv2d_gpu_optim2(s, out, inp, ker, padded, block_axis):
    # Optimization 2: Vectorization and Unrolling

    s[padded].compute_inline()
    thread_axes = []
    for ax in s[out].leaf_iter_vars:
        if ax.var.name == 'threadIdx.x' or ax.var.name == 'threadIdx.y':
            thread_axes.append(ax)
    
    if len(thread_axes) >= 1:
        s[out].pragma(thread_axes[0], "auto_unroll_max_step", 512)
        s[out].pragma(thread_axes[0], "unroll_explicit", True)
    
    return s


def dwsp_conv2d_gpu_optim3(s, out, inp, ker, padded, block_axis):
    # Optimization 3: Loop reordering for reduction axes
    reduce_axes = []
    for ax in s[out].op.reduce_axis:
        reduce_axes.append(ax)

    spatial_axes = []
    for ax in s[out].leaf_iter_vars:
        if ax.var.name.startswith('thread'):
            spatial_axes.append(ax)

    if reduce_axes and spatial_axes:
        current_order = list(s[out].leaf_iter_vars)
        new_order = []
        for ax in current_order:
            if ax not in reduce_axes:
                new_order.append(ax)

        if len(reduce_axes) >= 2:
            r_h, r_w = reduce_axes
            new_order.append(r_w)  # Putting r_w before r_h
            new_order.append(r_h)
        else:
            new_order.extend(reduce_axes)

        s[out].reorder(*new_order)
    return s
