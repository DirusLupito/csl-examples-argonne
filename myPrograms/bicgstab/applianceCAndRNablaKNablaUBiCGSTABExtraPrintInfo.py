#!/usr/bin/env cs_python

# Copyright 2025 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=too-many-function-args
""" test BiCGSTAB of a sparse matrix A built by 7-point stencil

  The following BiCGSTAB algorithm is modified from [1].
  ---
  The algorithm of unpreconditioned BiCGSTAB is
    Given b, x0 and tol = eps*|b|
    k = 0
    x = x0
    r0 = b - A*x
    r = r0
    p = r0
    xi = |r|^2 = (r0, r0)
    rho = xi
    while xi > tol*tol and k < max_ite
        k = k + 1
        v = A*p
        alpha = rho/(r0, v)
        s = r - alpha*v
        t = A*s
        w = (t,s)/(t,t)
        x = x + alpha*p + w*s
        r = s - w*t
        rho_old = rho
        rho = (r0, r)
        beta = (rho/rho_old)*(alpha/w)
        p = r + beta*(p - w*v)
        xi = |r|^2
    end
    x approximates the solution of a linear system Ax = b

  The sparse matrix A is built by a 7-point stenil.
  The 7-point stencil is defined by the following:
  ---
    The Laplacian operator L on 3-dimensional domain can be represented by 7-point
  stencil based on the standard 2nd order Finite Difference Method. The operator form
  with Dirichlet boundary conditions can be written by
         L[u](i,j,k) = u(i+1, j,  k  ) + u(i-1, j,   k  ) +
                       u(i,   j+1,k  ) + u(i,   j-1, k  ) +
                       u(i,   j,  k+1) + u(i,   j,   k-1) +
                      -6*u(i, j, k)
  In general the coefficients of those 7 points can vary. To minimize the memory
  consumption, this example assumes the coefficients are independent of index k and
  whole vector u(i,j,:) is placed in one PE (px=j, py=i).
  The above formula can be re-written by
     c_west   * x[i-1][j  ][k  ] + c_east  * x[i+1][j  ][k  ] +
     c_south  * x[i  ][j-1][k  ] + c_north * x[i  ][j+1][k  ] +
     c_bot    * x[i  ][j  ][k-1] + c_top   * x[i  ][j  ][k+1] +
     c_center * x[i][j][k]
  Each PE only holds 7 coefficients organized by c_west, c_east, c_south, c_north,
  c_bot, c_top and c_center.

  This example provides two modules, one is allreduce and the other is stencil_3d_7pts.
  "allreduce" module can synchronize all PEs to form a reference clock.
  "allreduce" module also computes dot(x,y) over a core rectangle.
  "stencil_3d_7pts" module can compute y = A*x where A is the matrix from 7-point stencil.

  The framework is
  ---
       sync()      // synchronize all PEs to sample the reference clock
       tic()       // record start time
       r = b - A*x
       for k = ...
         v = A*p
         update s
         t = A*s
         update x
         update r
         update xi=(r,r)
         update p
         D2H(xi) to check convergence
       end
       toc()       // record end time
  ---
  This framework does transfer the nrm(r) back to host for each iteration of BiCGSTAB.
  So the I/O pressure is high, not good for performance. device_run.py removes
  this IO pressure.

  The tic() samples "time_start" and toc() samples "time_end". The sync() samples
  "time_ref" which is used to shift "time_start" and "time_end".
  The elapsed time is measured by
       cycles_send = max(time_end) - min(time_start)

  The overall runtime is computed via the following formula
       time_send = (cycles_send / 0.85) *1.e-3 us
  where a PE runs with clock speed 850MHz

  Here is the list of parameters:
    -m=<int> is the height of the core
    -n=<int> is the width of the core
    -k=<int> is size of x and y allocated in the core
    --zDim=<int> is the number of f32 per PE, computed by y = A*x
                 zDim must be not greater than k
    --max-ite=<int> number of iterations
    --channels=<int> specifies the number of I/O channels, no bigger than 16

  WARNING: BiCGSTAB may not converge if rho=(rj, r0) is close to zero.
  If it does happen, we need to restart the BiCGSTAB.
  The algorithm here does not provide the restart.

  Reference:
  [1] H. A. VAN DER VORST, BI-CGSTAB: A FAST AND SMOOTHLY CONVERGING VARIANT
      OF BI-CG FOR THE SOLUTION OF NONSYMMETRIC LINEAR SYSTEMS,
      SIAM J. ScI. STAT. COMPUT. Vol. 13, No. 2, pp. 631-644, March 1992
"""

import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from bicgstab import bicgstab
from cmd_parser import parse_args
from scipy.sparse.linalg import eigs
from util import csr_7_pt_stencil, hwl_2_oned_colmajor, oned_to_hwl_colmajor
import time
import os.path

from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
from cerebras.sdk.client import SdkRuntime, SdkCompiler


def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)


def csl_compile_core(
    cslc: str,
    width: int,  # width of the core
    height: int,  # height of the core
    pe_length: int,
    blockSize: int,
    file_config: str,
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,  # fabric-offsets of the core
    core_fabric_offset_y: int,
    use_precompile: bool,
    arch: Optional[str],
    C0: int,
    C1: int,
    C2: int,
    C3: int,
    C4: int,
    C5: int,
    C6: int,
    C7: int,
    C8: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
):
  if not use_precompile:
    args = []
    args.append(cslc)  # command
    args.append(file_config)
    args.append(f"--fabric-dims={fabric_width},{fabric_height}")
    args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
    args.append(f"--params=width:{width},height:{height},MAX_ZDIM:{pe_length}")
    args.append(f"--params=BLOCK_SIZE:{blockSize}")
    args.append(f"--params=C0_ID:{C0}")
    args.append(f"--params=C1_ID:{C1}")
    args.append(f"--params=C2_ID:{C2}")
    args.append(f"--params=C3_ID:{C3}")
    args.append(f"--params=C4_ID:{C4}")
    args.append(f"--params=C5_ID:{C5}")
    args.append(f"--params=C6_ID:{C6}")
    args.append(f"--params=C7_ID:{C7}")
    args.append(f"--params=C8_ID:{C8}")

    args.append(f"-o={elf_dir}")
    if arch is not None:
      args.append(f"--arch={arch}")
    args.append("--memcpy")
    args.append(f"--channels={channels}")
    args.append(f"--width-west-buf={width_west_buf}")
    args.append(f"--width-east-buf={width_east_buf}")

    print(f"subprocess.check_call(args = {args}")
    subprocess.check_call(args)
  else:
    print("\tuse pre-compile ELFs")


def timing_analysis(height, width, time_memcpy_hwl, time_ref_hwl):
  # time_start = start time of spmv
  time_start = np.zeros((height, width)).astype(int)
  # time_end = end time of spmv
  time_end = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_memcpy_hwl[(h, w, 0)]
      word[1] = time_memcpy_hwl[(h, w, 1)]
      word[2] = time_memcpy_hwl[(h, w, 2)]
      time_start[(h, w)] = make_u48(word)
      word[0] = time_memcpy_hwl[(h, w, 3)]
      word[1] = time_memcpy_hwl[(h, w, 4)]
      word[2] = time_memcpy_hwl[(h, w, 5)]
      time_end[(h, w)] = make_u48(word)

  # time_ref = reference clock
  time_ref = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_ref_hwl[(h, w, 0)]
      word[1] = time_ref_hwl[(h, w, 1)]
      word[2] = time_ref_hwl[(h, w, 2)]
      time_ref[(h, w)] = make_u48(word)

  # adjust the reference clock by the propagation delay
  # the right-bottom PE signals other PEs, the propagation delay is
  #     (h-1) - py + (w-1) - px
  for py in range(height):
    for px in range(width):
      time_ref[(py, px)] = time_ref[(py, px)] - ((width + height - 2) - (px + py))

  # shift time_start and time_end by time_ref
  time_start = time_start - time_ref
  time_end = time_end - time_ref

  # cycles_send = time_end[(h,w)] - time_start[(h,w)]
  # 850MHz --> 1 cycle = (1/0.85) ns = (1/0.85)*1.e-3 us
  # time_send = (cycles_send / 0.85) *1.e-3 us
  #
  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) * 1.0e-3
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")

def k_function(x, y, z):
  """Variable coefficient k(x,y,z)"""
  # return 1 + 0.5 * np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)
  return 1 + x * y

def u_exact(x, y, z):
  """Exact solution u(x,y,z)"""
  # return x*(1-x)*y*(1-y)*z*(1-z)
  return np.sin(np.pi*x) * np.sin(np.pi*y)

def compute_source_term(x, y, z):
  """
  Computes the source term f = ∇k∇u for our specific k and u.
  This can be computed analytically or using finite differences.
  """
  # # Use finite differences to approximate the source term
  # h = 0.001  # Small h for accurate finite difference
  
  # # Compute u and its derivatives at the given point
  # u = u_exact(x, y, z)
  
  # # Compute first derivatives of u using central differences
  # ux = (u_exact(x+h, y, z) - u_exact(x-h, y, z)) / (2*h)
  # uy = (u_exact(x, y+h, z) - u_exact(x, y-h, z)) / (2*h)
  # uz = (u_exact(x, y, z+h) - u_exact(x, y, z-h)) / (2*h)
  
  # # Compute k and its derivatives
  # k = k_function(x, y, z)
  # kx = (k_function(x+h, y, z) - k_function(x-h, y, z)) / (2*h)
  # ky = (k_function(x, y+h, z) - k_function(x, y-h, z)) / (2*h)
  # kz = (k_function(x, y, z+h) - k_function(x, y, z-h)) / (2*h)
  
  # # Compute second derivatives of u
  # uxx = (u_exact(x+h, y, z) - 2*u_exact(x, y, z) + u_exact(x-h, y, z)) / (h*h)
  # uyy = (u_exact(x, y+h, z) - 2*u_exact(x, y, z) + u_exact(x, y-h, z)) / (h*h)
  # uzz = (u_exact(x, y, z+h) - 2*u_exact(x, y, z) + u_exact(x, y, z-h)) / (h*h)
  
  # # Compute ∇k∇u = k∇²u + ∇k·∇u
  # return k * (uxx + uyy + uzz) + kx * ux + ky * uy + kz * uz
  k = 1 + x * y
  delU = -2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
  kx = y
  ky = x
  ux = np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
  uy = np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)
  return k * delU + kx * ux + ky * uy

# How to compile
#   python run.py -m=5 -n=5 -k=5 --latestlink latest --channels=1 \
#   --width-west-buf=0 --width-east-buf=0 --compile-only
# How to run
#   python run.py -m=5 -n=5 -k=5 --latestlink latest --channels=1 \
#   --width-west-buf=0 --width-east-buf=0 --run-only --zDim=5 --max-ite=1
def main():
  """Main method to run the example code."""

  random.seed(127)

  # args, dirname = parse_args()

  # cslc = "cslc"
  # if args.driver is not None:
  #   cslc = args.driver

  # print(f"cslc = {cslc}")

  # width_west_buf = args.width_west_buf
  width_west_buf = 0
  # width_east_buf = args.width_east_buf
  width_east_buf = 0
  # channels = args.channels
  channels = 1
  assert channels <= 16, "only support up to 16 I/O channels"
  assert channels >= 1, "number of I/O channels must be at least 1"

  print(f"width_west_buf = {width_west_buf}")
  print(f"width_east_buf = {width_east_buf}")
  print(f"channels = {channels}")

  # height = args.m
  # width = args.n
  # pe_length = args.k
  # zDim = args.zDim
  # blockSize = args.blockSize
  blockSize = 2
  max_ite = 30000

  # Define a list of sizes to test
  size_values = [81] 
  
  # Lists to store results for plotting
  cpu_times = []
  wse_times = []
  cpu_errors = []
  wse_errors = []
  convergence_rates_cpu = []
  convergence_rates_wse = []
  condition_numbers = []
  h_values = []  # Grid spacing values
  
  print("Testing multiple grid sizes:", size_values)
  print("=" * 60)
  
  for size_idx, size in enumerate(size_values):
    print(f"\n\nRunning test with grid size {size}x{size}x{size}")
    print("-" * 60)
    
    # Set parameters based on the current size
    height = size
    width = size
    pe_length = size
    zDim = size
    
    # Calculate grid spacing
    h = 1.0 / (size - 1)
    h_values.append(h)

    print(f"width={width}, height={height}, pe_length={pe_length}, zDim={zDim}, blockSize={blockSize}")
    print(f"max_ite = {max_ite}")
    assert pe_length >= 2, "the maximum size of z must be greater than 1"
    assert zDim <= pe_length, "[0, zDim) cannot exceed the storage"

    # Initialize solution vectors
    np.random.seed(2)
    x = np.zeros((height, width, pe_length), dtype=np.float32)
    #x = (np.arange(height * width * zDim).reshape(height, width, zDim).astype(np.float32) + 100)
  
    x_1d = hwl_2_oned_colmajor(height, width, zDim, x, np.float32)
    nrm2_x = np.linalg.norm(x_1d.ravel(), 2)
    # |x0|_2 = 1
    if nrm2_x > 0:
      x_1d = x_1d / nrm2_x
      x = x / nrm2_x
  
    # Create a uniform grid in the unit cube [0,1]³
    x_coords = np.linspace(0, 1, width)
    y_coords = np.linspace(0, 1, height)
    z_coords = np.linspace(0, 1, pe_length)

    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    X = np.transpose(X, (1, 0, 2))  # Adjust for (height, width, pe_length) format
    Y = np.transpose(Y, (1, 0, 2))
    Z = np.transpose(Z, (1, 0, 2))

    # Use a 2D solution (constant in z-direction)
    u_exact_values = u_exact(X, Y, Z)

    # Compute the source term (right-hand side)
    source_term_values = compute_source_term(X, Y, Z)

    # Create the right-hand side vector b, applying boundary conditions
    b = np.zeros((height, width, pe_length), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            for k in range(pe_length):
                if (i == 0 or i == height-1 or j == 0 or j == width-1):
                    # Dirichlet boundary condition from exact solution
                    b[i, j, k] = u_exact_values[i, j, k]
                else:
                    # Source term in the interior
                    b[i, j, k] = source_term_values[i, j, k]
                  
    b_1d = hwl_2_oned_colmajor(height, width, pe_length, b, np.float32)

    # stencil coefficients has the following order
    # {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
    stencil_coeff = np.zeros((height, width, 7), dtype=np.float32)
    for i in range(height):
      for j in range(width):
        # Boundary condition adjustment
        if i == 0 or i == height - 1 or j == 0 or j == width - 1:
          stencil_coeff[(i, j, 0)] = 0.0  # west
          stencil_coeff[(i, j, 1)] = 0.0  # east
          stencil_coeff[(i, j, 2)] = 0.0  # south
          stencil_coeff[(i, j, 3)] = 0.0  # north
          stencil_coeff[(i, j, 4)] = 0.0  # bottom
          stencil_coeff[(i, j, 5)] = 0.0  # top
          stencil_coeff[(i, j, 6)] = 1.0  # center
          continue
        
        # # Compute k at cell centers and interfaces
        # k_center = k_function(x[i], y[j], z[l])
        k_center = k_function(x_coords[j], y_coords[i], z_coords[0])  # Use first z coordinate for 2D

        # # X direction interfaces
        # k_i_minus_half = (k_function(x[i-1], y[j], z[l]) + k_center) / 2
        # k_i_plus_half = (k_function(x[i+1], y[j], z[l]) + k_center) / 2
        k_i_minus_half = (k_function(x_coords[j - 1], y_coords[i], z_coords[0]) + k_center) / 2
        k_i_plus_half = (k_function(x_coords[j + 1], y_coords[i], z_coords[0]) + k_center) / 2

        # # Y direction interfaces
        # k_j_minus_half = (k_function(x[i], y[j-1], z[l]) + k_center) / 2
        # k_j_plus_half = (k_function(x[i], y[j+1], z[l]) + k_center) / 2
        k_j_minus_half = (k_function(x_coords[j], y_coords[i + 1], z_coords[0]) + k_center) / 2 # For some reason, y_coords is flipped
        k_j_plus_half = (k_function(x_coords[j], y_coords[i - 1], z_coords[0]) + k_center) / 2 # For some reason, y_coords is flipped

        stencil_coeff[(i, j, 0)] = k_i_minus_half / h**2  # west
        stencil_coeff[(i, j, 1)] = k_i_plus_half / h**2 # east
        stencil_coeff[(i, j, 2)] = k_j_minus_half / h**2  # south
        stencil_coeff[(i, j, 3)] = k_j_plus_half / h**2  # north
        stencil_coeff[(i, j, 4)] = 0.0  # bottom (zero for 2D problem)
        stencil_coeff[(i, j, 5)] = 0.0  # top (zero for 2D problem)
        stencil_coeff[(i, j, 6)] = -(k_i_minus_half + k_i_plus_half + k_j_minus_half + k_j_plus_half) / h**2

        # Also don't add connections to boundary neighbors
        if i == 1:
          stencil_coeff[(i, j, 3)] = 0.0  # north

        if i == height - 2:
          stencil_coeff[(i, j, 2)] = 0.0  # south

        if j == 1:
          stencil_coeff[(i, j, 0)] = 0.0  # west

        if j == width - 2:
          stencil_coeff[(i, j, 1)] = 0.0  # east

    print(f"Setting up for size {size}x{size}x{size}")

    # Create CSR matrix
    A_csr = csr_7_pt_stencil(stencil_coeff, height, width, zDim)

    # # Calculate condition number
    # vals, _ = eigs(A_csr, k=1, which="SM")
    # min_eig = abs(vals[0])
    # vals, _ = eigs(A_csr, k=1, which="LM")
    # max_eig = abs(vals[0])
    # cond = max_eig/min_eig
    # condition_numbers.append(cond)
    # print(f"Matrix condition number = {cond}")
    # # Check if A is semidefinite (its symmetric at this point so we just check if the minimum eigenvalue is non-negative)
    # if min_eig < 0:
    #   print(f"Matrix A is not semidefinite (min_eig = {min_eig})")
    # else:
    #   print(f"Matrix A is semidefinite (min_eig = {min_eig})")

    nrm_b = np.linalg.norm(b_1d.ravel(), 2)
    eps = 1.0e-3
    tol = eps * nrm_b
    print(f"tol = {tol:.6e} (eps = {eps})")
    
    # xf_1d, xi, k = bicgstab(A_csr, x_1d, b_1d, max_ite, tol)
    # print(f"[host] after BiCGSTAB, xi = {xi}, k = {k}")
    # cpu_numerical_solution_bicgstab = oned_to_hwl_colmajor(height, width, zDim, xf_1d, np.float32)
    # error_cpu_bicgstab = cpu_numerical_solution_bicgstab - u_exact_values[:,:,:zDim]
    # max_abs_error_cpu_bicgstab = np.max(np.abs(error_cpu_bicgstab))
    # print(f"CPU maximum absolute error (BiCGSTAB): {max_abs_error_cpu_bicgstab:.6e}")
    # # Calculate RMSE for CPU BiCGSTAB
    # rmse_cpu_bicgstab = np.sqrt(np.mean(np.square(error_cpu_bicgstab)))
    # print(f"CPU root mean square error (BiCGSTAB): {rmse_cpu_bicgstab:.6e}")

    # Run CPU solution
    t0_cpu = time.time()
    xf_1d, xi, k_cpu = bicgstab(A_csr, x_1d, b_1d, max_ite, tol)
    t1_cpu = time.time()
    cpu_time = t1_cpu - t0_cpu
    cpu_times.append(cpu_time)
    print(f"CPU conjugate gradient took {cpu_time:.4f} seconds with {k_cpu} iterations and xi={xi:.6e}")

    # Analyze CPU solution accuracy
    cpu_numerical_solution = oned_to_hwl_colmajor(height, width, zDim, xf_1d, np.float32)
    error_cpu = cpu_numerical_solution - u_exact_values[:,:,:zDim]
    max_abs_error_cpu = np.max(np.abs(error_cpu))
    cpu_errors.append(max_abs_error_cpu)
    print(f"CPU maximum absolute error: {max_abs_error_cpu:.6e}")

    # Calculate RMSE for CPU
    rmse_cpu = np.sqrt(np.mean(np.square(error_cpu)))
    print(f"CPU root mean square error: {rmse_cpu:.6e}")
  
    t0 = time.time()

    # Instantiate compiler
    compiler = SdkCompiler()

    t1 = time.time()

    print(f"Compiler instantiated in {t1 - t0:.8f} seconds")
    
    t0 = time.time()
    
    artifact_path = compiler.compile(
      ".",
      "src/layout.csl",
      f"--fabric-dims=757,996 --fabric-offsets=4,1 --params=width:{size},height:{size},MAX_ZDIM:{size} --params=BLOCK_SIZE:{blockSize} --params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 --params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out --memcpy --channels={channels} --width-west-buf={width_west_buf} --width-east-buf={width_east_buf}",
      "."
    )
    t1 = time.time()
    print(f"Compilation took {t1 - t0:.8f} seconds")

    # Calculate convergence rate if not the first size
    if size_idx > 0:
        # Calculate h₁/h₂
        h_ratio = h_values[size_idx-1] / h_values[size_idx]
        # Calculate log(error₁/error₂)/log(h₁/h₂)
        cpu_rate = np.log(cpu_errors[size_idx-1]/cpu_errors[size_idx]) / np.log(h_ratio)
        convergence_rates_cpu.append(cpu_rate)
        print(f"CPU convergence rate from {size_values[size_idx-1]} to {size}: {cpu_rate:.4f}")

    # Run on WSE
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    
    t0 = time.time()
    with SdkRuntime(artifact_path, simulator=False) as runner:
        t1 = time.time()
        t0_wse = time.time()
        print(f"Asking for appliance to start running took {t1 - t0:.8f} seconds")
        # Copy data to device and run CG
        symbol_b = runner.get_id("b")
        symbol_x = runner.get_id("x")
        symbol_xi = runner.get_id("xi")
        symbol_stencil_coeff = runner.get_id("stencil_coeff")
        symbol_time_buf_u16 = runner.get_id("time_buf_u16")
        symbol_time_ref = runner.get_id("time_ref")

        print("copy vector b and x0")
        runner.memcpy_h2d(
            symbol_b,
            b_1d,
            0,
            0,
            width,
            height,
            zDim,
            streaming=False,
            data_type=memcpy_dtype,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=True,
        )

        runner.memcpy_h2d(
            symbol_x,
            x_1d,
            0,
            0,
            width,
            height,
            zDim,
            streaming=False,
            data_type=memcpy_dtype,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=True,
        )

        print("copy 7 stencil coefficients")
        stencil_coeff_1d = hwl_2_oned_colmajor(height, width, 7, stencil_coeff, np.float32)
        runner.memcpy_h2d(
            symbol_stencil_coeff,
            stencil_coeff_1d,
            0,
            0,
            width,
            height,
            7,
            streaming=False,
            data_type=memcpy_dtype,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=True,
        )

        print("step 0: enable timer")
        runner.launch("f_enable_timer", nonblock=False)

        print("step 1: sync all PEs")
        runner.launch("f_sync", nonblock=False)

        print("step 2: copy reference clock from reduce module")
        runner.launch("f_reference_timestamps", nonblock=False)

        print("step 3: tic() records time_start")
        runner.launch("f_tic", nonblock=True)

        print(f"step 4: BiCGSTAB with max_ite = {max_ite}, zDim = {zDim}")

        print("step 4.1: initialization")
        # - setup the length of all DSDs
        # - setup the size of local tensor
        runner.launch("f_bicgstab_init", np.int16(zDim), nonblock=False)

        k_wse = 0
        print("step 4.2: compute r0=b-A*x0, r=r0, p=r0, and rho=xi=|r0|^2")
        # v = A*x0
        runner.launch("f_spmv_Ax", nonblock=False)
        # r0 = b - v = b - A*x0
        # r = r0
        # p = r0
        # xi = |r0|^2
        runner.launch("f_residual", nonblock=False)

        # rho = xi = (r0, r0)
        runner.launch("f_init_rho", nonblock=False)

        # [optional] D2H(xi)
        xi_wse = np.zeros(1, np.float32)
        runner.memcpy_d2h(
            xi_wse,
            symbol_xi,
            0,
            0,
            1,
            1,
            1,
            streaming=False,
            data_type=memcpy_dtype,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        xi = xi_wse[0]
        print(f"[BiCGSTAB] iter {k_wse}: xi = {xi}")
        # if |r| < tol, then exit
        while (xi > tol * tol) and (k_wse < max_ite):
          k_wse = k_wse + 1
          print("step 4.3: v = A*p")
          runner.launch("f_spmv_Ap", nonblock=False)

          print("step 4.4: compute (r0, v)")
          # r0_dot_v = dot(r0, v)
          runner.launch("f_r0_dot_v", nonblock=False)

          print("step 4.4: update alpha, s and t")
          # alpha = rho / (r0, v)
          # s = r - alpha*v
          # t = A*s
          runner.launch("f_update_alpha_s_t", nonblock=False)

          print("step 4.5: compute (t, s)")
          # t_dot_s = np.dot(t,s)
          runner.launch("f_t_dot_s", nonblock=False)

          print("step 4.6: compute (t, t)")
          # t_dot_t = np.dot(t,t)
          runner.launch("f_t_dot_t", nonblock=False)

          print("step 4.7: update w, x, r and rho ")
          # w = (t,s)/(t,t)
          # x = x + alpha*p + w*s
          # r = s - w*t
          # rho_old = rho
          # rho = (r0, r)
          runner.launch("f_update_w_x_r_rho", nonblock=False)

          # beta = (rho/rho_old)*(alpha/w)
          # p = r + beta*(p - w*v)
          # xi = np.dot(r,r)
          runner.launch("f_update_beta_p_xi", nonblock=False)

          # [optional] D2H(xi)
          runner.memcpy_d2h(
              xi_wse,
              symbol_xi,
              0,
              0,
              1,
              1,
              1,
              streaming=False,
              data_type=memcpy_dtype,
              order=MemcpyOrder.COL_MAJOR,
              nonblock=False,
          )
          xi = xi_wse[0]
          print(f"[BiCGSTAB] iter {k}: xi = {xi}")

        print("step 5: toc() records time_end")
        runner.launch("f_toc", nonblock=False)

        print("step 6: prepare (time_start, time_end)")
        runner.launch("f_memcpy_timestamps", nonblock=False)

        print("step 7: D2H (time_start, time_end)")
        time_memcpy_hwl_1d = np.zeros(height * width * 6, np.uint32)
        runner.memcpy_d2h(
            time_memcpy_hwl_1d,
            symbol_time_buf_u16,
            0,
            0,
            width,
            height,
            6,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_16BIT,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        time_memcpy_hwl = oned_to_hwl_colmajor(height, width, 6, time_memcpy_hwl_1d, np.uint16)

        print("step 8: D2H reference clock")
        time_ref_1d = np.zeros(height * width * 3, np.uint32)
        runner.memcpy_d2h(
            time_ref_1d,
            symbol_time_ref,
            0,
            0,
            width,
            height,
            3,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_16BIT,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        time_ref_hwl = oned_to_hwl_colmajor(height, width, 3, time_ref_1d, np.uint16)

        print("step 9: D2H x[zDim]")
        xf_wse_1d = np.zeros(height * width * zDim, np.float32)
        runner.memcpy_d2h(
            xf_wse_1d,
            symbol_x,
            0,
            0,
            width,
            height,
            zDim,
            streaming=False,
            data_type=memcpy_dtype,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )

    t1_wse = time.time()
    wse_time = t1_wse - t0_wse
    wse_times.append(wse_time)
    print(f"WSE run took {wse_time:.4f} seconds with {k_wse} iterations")

    # Calculate WSE timing from device measurements
    time_start = np.zeros((height, width)).astype(int)
    time_end = np.zeros((height, width)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(width):
        for h in range(height):
            word[0] = time_memcpy_hwl[(h, w, 0)]
            word[1] = time_memcpy_hwl[(h, w, 1)]
            word[2] = time_memcpy_hwl[(h, w, 2)]
            time_start[(h, w)] = make_u48(word)
            word[0] = time_memcpy_hwl[(h, w, 3)]
            word[1] = time_memcpy_hwl[(h, w, 4)]
            word[2] = time_memcpy_hwl[(h, w, 5)]
            time_end[(h, w)] = make_u48(word)

    time_ref = np.zeros((height, width)).astype(int)
    for w in range(width):
        for h in range(height):
            word[0] = time_ref_hwl[(h, w, 0)]
            word[1] = time_ref_hwl[(h, w, 1)]
            word[2] = time_ref_hwl[(h, w, 2)]
            time_ref[(h, w)] = make_u48(word)
            time_ref[(h, w)] = time_ref[(h, w)] - ((width + height - 2) - (w + h))

    time_start = time_start - time_ref
    time_end = time_end - time_ref
    min_time_start = time_start.min()
    max_time_end = time_end.max()
    cycles_send = max_time_end - min_time_start
    time_send = (cycles_send / 0.85) * 1.0e-3
    print(f"WSE time from hardware counters: {time_send:.4f} microseconds")

    # Analyze WSE solution accuracy
    wse_numerical_solution = oned_to_hwl_colmajor(height, width, zDim, xf_wse_1d, np.float32)
    error_wse = wse_numerical_solution - u_exact_values[:,:,:zDim]
    max_abs_error_wse = np.max(np.abs(error_wse))
    wse_errors.append(max_abs_error_wse)
    print(f"WSE maximum absolute error: {max_abs_error_wse:.6e}")
    
    # Calculate RMSE for WSE
    rmse_wse = np.sqrt(np.mean(np.square(error_wse)))
    print(f"WSE root mean square error: {rmse_wse:.6e}")
    
  # Save results to file
  # Format: size h cond cpu_time wse_time cpu_error wse_error cpu_iter wse_iter
  results_file = "results.txt"
  file_exists = os.path.isfile(results_file)

  with open(results_file, "a") as f:
    # Write header if file doesn't exist
    if not file_exists:
      f.write("# size h cpu_time wse_time cpu_max_err wse_max_err cpu_rmse wse_rmse cpu_iter wse_iter\n")
    
    # Write results as a single line
    f.write(f"{size} {h} {cpu_time} {wse_time} {max_abs_error_cpu} {max_abs_error_wse} {rmse_cpu} {rmse_wse} {k_cpu} {k_wse}\n")

  print(f"\nResults saved to {results_file}")
  print("\nSUCCESS!")

  # vals, _ = eigs(A_csr, k=1, which="SM")
  # min_eig = abs(vals[0])
  # vals, _ = eigs(A_csr, k=1, which="LM")
  # max_eig = abs(vals[0])
  # print(f"min(eig) = {min_eig}")
  # print(f"max(eig) = {max_eig}")
  # print(f"cond(A) = {max_eig/min_eig}")

if __name__ == "__main__":
  main()
