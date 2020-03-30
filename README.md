# Making-MLIR-interact-with-existing-polyhedral-tools
Making MLIR interact with existing polyhedral tools
it supports the following dialects:
Affine ,LLVM IR Dialect.


1.Nested Regions

%2 = xla.fusion (%0 : tensor<f32>, %1 : tensor<f32>) : tensor<f32> {
 ^bb0(%a0 : tensor<f32>, %a1 : tensor<f32>):
 %x0 = xla.add %a0, %a1 : tensor<f32>
 %x1 = xla.relu %x0 : tensor<f32>
 return %x1
 }
