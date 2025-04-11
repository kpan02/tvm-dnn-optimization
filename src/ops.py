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
    
    # Apply optimization 1: Tiling
    s, *block_axes = gemm_gpu_optim1(s, C, A, B)
    
    # Apply optimization 2: Shared Memory
    s = gemm_gpu_optim2(s, C, A, B, block_axes)
    
    # Apply optimization 3: Vectorization
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

    # TODO: fill-in start
    out = None
    s = None
    # TODO: fill-in end

    return s, inp, ker, out
