# Making-MLIR-interact-with-existing-polyhedral-tools
Making MLIR interact with existing polyhedral tools
it supports the following dialects:
Affine ,LLVM IR Dialect.


<h2>Nested Regions</h2>

%2 = xla.fusion (%0 : tensor<f32>, %1 : tensor<f32>) : tensor<f32> {
 ^bb0(%a0 : tensor<f32>, %a1 : tensor<f32>):
 %x0 = xla.add %a0, %a1 : tensor<f32>
 %x1 = xla.relu %x0 : tensor<f32>
 return %x1
 }
 
<h2>Context: Traditional Polyhedral Form</h2>
We started by discussing a representation that uses the traditional polyhedral schedule set + domain representation, e.g. consider C-like code like:


  void simple_example(...) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
         float tmp = X[i,j]    // S1
         A[i,j] = tmp + 1      // S2
         B[i,j] = tmp * 42     // S3
       }
    }
  }
  <h2>Proposal: Simplified Polyhedral Form</h2>
mlfunc @simple_example(... %N) {
    affine.for %i = 0 ... %N step 1 {
      affine.for %j = 0 ... %N step 1 {
        // identity noop in this case, but can exist in general.
        %0,%1 = affine.apply #57(%i, %j)

        %tmp = call @S1(%X, %0, %1)

        call @S2(%tmp, %A, %0, %1)

        call @S3(%tmp, %B, %0, %1)
      }
    }
  }

<h2>The example with the reduced domain would be represented with an if instruction:</h2>

  mlfunc @reduced_domain_example(... %N) {
    affine.for %i = 0 ... %N step 1 {
      affine.for %j = 0 ... %N step 1 {
        // identity noop in this case, but can exist in general.
        %0,%1 = affinecall #57(%i, %j)

        %tmp = call @S1(%X, %0, %1)

        if (10 <= %i < %N-10), (10 <= %j < %N-10) {

          %2,%3 = affine.apply(%i, %j)    // identity noop in this case

          call @S2(%tmp, %A, %2, %3)
        }
      }
    }
  }
  <h2>Evaluation</h2>
Both of these forms are capable of expressing the same class of computation: multidimensional loop nests with affine loop bounds and affine memory references
