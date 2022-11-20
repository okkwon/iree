// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))" %s | FileCheck %s
module attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>]}>]} {
  hal.executable private @forward_dispatch_116 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
      hal.executable.export public @forward_dispatch_116_matmul_128x30522x768 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @forward_dispatch_116_matmul_128x30522x768() {
          %c512 = arith.constant 512 : index
          %c786944 = arith.constant 786944 : index
          %c265458176 = arith.constant 265458176 : index
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c512) alignment(64) : !flow.dispatch.tensor<readonly:tensor<128x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c786944) alignment(64) : !flow.dispatch.tensor<readonly:tensor<768x30522xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c265458176) alignment(64) : !flow.dispatch.tensor<readonly:tensor<30522xf32>>
          %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<128x30522xf32>>
          %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x768xf32>> -> tensor<128x768xf32>
          %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [768, 30522], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<768x30522xf32>> -> tensor<768x30522xf32>
          %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [30522], strides = [1] : !flow.dispatch.tensor<readonly:tensor<30522xf32>> -> tensor<30522xf32>
          %7 = tensor.empty() : tensor<128x30522xf32>
          %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<128x30522xf32>) -> tensor<128x30522xf32>
          %9 = linalg.matmul ins(%4, %5 : tensor<128x768xf32>, tensor<768x30522xf32>) outs(%8 : tensor<128x30522xf32>) -> tensor<128x30522xf32>
          %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %6 : tensor<128x30522xf32>, tensor<30522xf32>) outs(%7 : tensor<128x30522xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %11 = arith.addf %arg0, %arg1 : f32
            linalg.yield %11 : f32
          } -> tensor<128x30522xf32>
          flow.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [128, 30522], strides = [1, 1] : tensor<128x30522xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x30522xf32>>
          return
        }
      }
    }
  }
}

// The specialized workgroup should have vector operations.

// CHECK-LABEL: func.func @forward_dispatch_116_matmul_128x30522x768
//       CHECK: arith.cmpi eq
//       CHECK: scf.if
//       CHECK:   gpu.subgroup_mma_load_matrix
//       CHECK:   gpu.subgroup_mma_load_matrix
//       CHECK:   gpu.subgroup_mma_load_matrix
//       CHECK:   gpu.subgroup_mma_load_matrix
//       CHECK:   gpu.subgroup_mma_compute
//       CHECK:   gpu.subgroup_mma_compute
//       CHECK:   gpu.subgroup_mma_store_matrix
//       CHECK: else
//   CHECK-NOT:   vector.transfer
