.. _slep_023:

============================================
SLEP023: Using Highway for SIMD Acceleration
============================================

:Author: Meekail Zain
:Status: Draft
:Type: Standards Track
:Created: 2023-9-16

Abstract
--------

This SLEP proposes the inclusion of C++ based SIMD acceleration.
The initial scope is limited to accelerating ``DistanceMetric`` implementations,
however there may be other areas which could benefit from similar acceleration.


Detailed description
--------------------

For many estimators (see `Implementation`_), a dominant bottleneck for
computation over many samples is distance computation. Currently, any distance
calculated using the ``DistanceMetric`` back-end is completed through repeated
calls to ``DistanceMetric.dist`` which operates on a pair of sample vectors. These
distance calculations is a series of parallel arithmetic operations with some
reduction which are currently implemented as scalar loops in Cython/C. These parallel
arithmetic operations occur on data contiguous in memory, and hence are well-suited
to SIMD optimization. For modern hardware, leveraging SIMD instructions can speed
up distance computation by **a factor of 4x-16x**, depending on data type and
specific SIMD architecture. For estimators where distance computation is a
significant bottleneck, such as ``KNeighborsRegressor``, this translates to a
**2.5x-4x speed up**.

While SIMD may be leveraged anywhere that there are data-parallel scalar operations
operating on contiguous memory, this SLEP focuses on the use case of accelerating
``DistanceMetric`` implementations.

SIMD optimization relies in CPU-specific features, which may not be present at runtime
if compiled and run on different machines. This increases the complexity of ensuring
portability of code, however modern SIMD libraries help mitigate this complexity by
providing helpful abstractions with allow for writing architecture-agnostic
implementations.

This SLEP proposes to use Google Highway, a C++ library which provides coverage for
a wide variety of SIMD architectures through a simple architecture-agnostic API.
In addition, Highway is actively maintained, tightly tested, and proactive in
supporting downstream libraries. See `Libraries`_ for other libraries considered.

The adoption of SIMD-accelerated computation would pose no change to user-experience
aside from reduced runtimes. It is important to note that the implementation requires
C++ which may introduce some complexity in terms of maintainence, however the core
computation implementations are simple, with the majority of required code being
boiler-plate. The only notable change to the build system would be the compilation
and use of a runtime library to isolate the C++ necessary for the SIMD routines.
Note that the ability to isolate the C++ code means that this SLEP **does not**
require formal acceptance of C++ as an extension language in the broader context
of the entire library. This limits the scope of C++ used in scikit-learn to only
the SIMD-accelerated runtime library.


Implementation
--------------

Dispatching
^^^^^^^^^^^

In order to leverage SIMD instructions, even with the aforementioned libraries, a
choice must be made regarding how to dispatch to an appropriate implementation.
Generally, one must use either static (single-target) or dynamic (multi-target)
dispatching. Static dispatching compiles against a single SIMD architecture per
platform and falls back to a scalar implementation if that architecture is not
present at runtime.

A more complex approach is to compile implementations of architecture-agnostic
code multiple times, targeting different SIMD architectures (e.g. ``SSE3, AVX, AVX2``
on ``x86_64``). Then, at runtime, computational calls are dispatched to the
*best available* SIMD architecture. This runtime dispatch strategy incurs some
overhead, however it is negligible for sufficiently-many features (~>12). For
vectors with too-few features, a simple conditional check on the numver of features
results in implementations with no major regressions. The use of dynamic dispatch
ensures the use of the best-possible runtime architecture while preserving greater
general availability of SIMD acceleration for users with older SIMD architectures.

Note that highway provides fallback universal scalar implementations (denoted
``EMU128, SCALAR``) which can simplify scikit-learn implementations by freeing us
from the obligation of providing our own scalar loops (i.e. our current implementations)
*however* it is observed that it is faster to avoid the highway dispatch mechanism
entirely for sufficiently short vectors, and thus maintaining our current scalar
implementations *in addition to* new SIMD optimized implementations is wortwhile.

An example implementation of dynamic dispatch for accelerating ``ManhattanDistance.dist``
can be found `here <https://github.com/Micky774/scikit-learn/pull/12/files>`__. The
core computation is defined in `simd.cpp <https://github.com/Micky774/scikit-learn/blob/75bda031f665c1978917ae3a05b94153036c62fa/sklearn/metrics/_simd/simd.cpp>`__.
The crux of the computation is simple, however the use of unrolling lengthens the
implementation without much additional complexity. The implementation includes some
Highway boiler-plate code to ensure proper dispatching.

For illustration purposes, here is an example of computing the manhattan distance
between two vectors with Highway without loop unrolling::

      template <typename Type>
      inline Type manhattan_dist(
         const Type* x,
         const Type* y,
         const size_t size
      ) {
         const hn::ScalableTag<Type> d;
         using batch_type = decltype(hn::Zero(d));
         batch_type simd_sum = hn::Zero(d);
         batch_type simd_x;
         batch_type simd_y;

         size_t lane_size = hn::Lanes(d);
         size_t vec_size = size - size % lane_size;
         for (size_t i = 0; i < vec_size; i += lane_size) {
               simd_x = hn::LoadU(d, x + i);
               simd_y = hn::LoadU(d, y + i);
               simd_sum += hn::AbsDiff(simd_x, simd_y);
         }
         Type scalar_sum = hn::ReduceSum(d, simd_sum);
         for (size_t i = vec_size; i < size; i += 1) {
               scalar_sum += fabs(x[i] - y[i]);
         }
         return scalar_sum;


Affected Estimators
^^^^^^^^^^^^^^^^^^^

Roughly speaking, any estimator which leverages ``DistanceMetric`` objects for
distance computation or any member of the ``PairwiseDistancesReduction`` family
will stand to benefit from SIMD acceleration. Those that are more computation-bound
will obtain speedups closer to the ideal 4x-16x. A non-exhaustive list of such
estimators can be found `here <https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_1_0.html#performance-improvements>`__.

Supported Architectures
^^^^^^^^^^^^^^^^^^^^^^^

Highway supports:

- Arm: NEON (Armv7+), SVE, SVE2, SVE_256, SVE2_128;
- POWER: PPC8 (v2.07), PPC9 (v3.0), PPC10 (v3.1B, not yet supported due to compiler bugs);
- RISC-V: RVV (1.0)
- WebAssembly: WASM, WASM_EMU256
- x86: SSE2, SSSE3, SSE4, AVX2, AVX3, AVX3_DL, AVX3_ZEN4, AVX3_SPR


Backward compatibility
----------------------

No concerns.


Future Work
-----------

The introduction of SIMD acceleration opens the door to several potential future
optimizations. Currently, any data-parallel scalar operations operating on contiguous
memory warrant further investigation on whether explicit SIMD vectorization may be
beneficial. Consequently, the availability of SIMD also encourages us to prioritize
reframing computation in terms of data-parallel operations on contiguous memory for
potential gains. For example, adding a method for batched distance computation would
lead to even greater speedups for ``metrics.pairwise_distances``.

Alternatives
------------

Libraries
^^^^^^^^^
- `Experimental C++ SIMD library <https://en.cppreference.com/w/cpp/experimental/simd>`_
   - Experimental
   - No WASM (?)
- `xsimd <https://github.com/xtensor-stack/xsimd>`_
   - Complex dynamic dispatch which requires code generation and using compiler flags
   - Limited supported architectures
   - No WASM
- `libsimdpp <https://github.com/p12tic/libsimdpp>`_
   - Inactive, most recent release Dec 14th, 2017
   - No WASM
- `SIMD everywhere <https://github.com/simd-everywhere/simde>`_
   - No WASM
- `SLEEF <https://github.com/shibatch/sleef>`_
   - Inactive, most recent release Sep 14th, 2020
   - No WASM

Static Dispatching
^^^^^^^^^^^^^^^^^^

The simplest implementation of SIMD acceleration is to compile for an explicitly
chosen baseline SIMD architecture for each supported platform (e.g. ``AVX`` on the
``x86_64`` platform) and fallback to scalar loops. This approach is relatively simple
to implement with minimal overhead, however potentially sacrifices coverage since
machines without support for their platform's corresponding static-dispatch target
*must* use the scalar fallback. Therefore, a tradeoff must be made between using
*high-throughput* architectures (e.g. ``AVX3``), or more *common* architectures
(e.g. ``SSE3``).


Discussion
----------

This section may just be a bullet list including links to any discussions
regarding the SLEP:

- This includes links to mailing list threads or relevant GitHub issues.


References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

Copyright
---------

This document has been placed in the public domain.
