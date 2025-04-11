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

    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    # Default schedule
    s = te.create_schedule(C.op)

    # the i-th block is indexed by blockIdx.x.
    # the number of threads in each block is blockDim.x
    # and the i-th thread within a block is indexed by threadIdx.x
    # overall index of a thread can be calculated as
    # 𝑖=blockIdx.x×blockDim.x+threadIdx.x
    block_x = te.thread_axis("blockIdx.y")
    block_y = te.thread_axis("blockIdx.x")

    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis
    s[C].bind(y, block_y)
    s[C].bind(x, block_x)

    return s, A, B, C


def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")

    # TODO: fill-in start
    out = None
    s = None
    # TODO: fill-in end

    return s, inp, ker, out
