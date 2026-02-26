/***************************************************************************
 *   OpenACC GPU Likelihood Computation for IQ-TREE — Test Declarations   *
 *   Test & verification functions for OpenACC implementation steps        *
 ***************************************************************************/

#ifndef PHYLOKERNEL_OPENACC_TEST_H
#define PHYLOKERNEL_OPENACC_TEST_H

#ifdef USE_OPENACC

// Forward declarations
class PhyloTree;

// ==========================================================================
// Step 2: JC Transition Matrix — test & verification
// ==========================================================================

// ==========================================================================
// Step 3: Tip one-hot vectors — test & verification
// ==========================================================================

/**
 * Verify that tip one-hot encoding produces correct [1,0,0,0], [0,1,0,0], etc.
 * for each DNA state, and [1,1,1,1] for unknown/gap.
 *
 * @return true if all tests pass
 */
bool testTipOneHot();

// ==========================================================================
// Step 4: TIP-TIP internal node — test & verification
// ==========================================================================

/**
 * Standalone test of TIP-TIP cherry computation (no IQ-TREE tree required).
 * Computes partial likelihood at an internal node with two leaf children
 * using Felsenstein's pruning formula, and verifies log-likelihood.
 *
 * Tests:
 *   4a: Single mismatch (A vs C) at t=0.1 → lnL = -4.224717
 *   4b: Single match (A vs A) at t=0.1 → lnL ≈ -1.579
 *   4c: Multi-pattern 2-taxon alignment → total lnL valid
 *   4d: Asymmetric branch lengths → correct range
 *
 * @return true if all tests pass
 */
bool testTipTipInternal();

// ==========================================================================
// Step 5: TIP-INTERNAL + INTERNAL-INTERNAL — test & verification
// ==========================================================================

/**
 * Standalone test of TIP-INTERNAL and INTERNAL-INTERNAL computation
 * (no IQ-TREE tree required). Exercises:
 *   - 3-taxon tree: TIP-TIP at cherry → TIP-INTERNAL at root
 *   - 4-taxon tree: TIP-TIP at two cherries → INTERNAL-INTERNAL at root
 *
 * Tests:
 *   5a: Reference 3-taxon ((A:0.1,B:0.2):0.05,C:0.15) → lnL = -6.465999
 *   5b: Intermediate partial likelihood verification against plan values
 *   5c: 4-taxon INTERNAL-INTERNAL with swap symmetry check
 *   5d: Multi-pattern 3-taxon alignment → weighted total lnL
 *
 * Loop structure matches production kernels (OpenACC-ready):
 *   for s (output state): for x (inner dot product): vleft += P[s*K+x]*L[x]
 *
 * @return true if all tests pass
 */
bool testTipInternalInternal();

// ==========================================================================
// Full kernel tests (require initialized PhyloTree)
// ==========================================================================

/**
 * Test the OpenACC scalar likelihood kernels by computing the log-likelihood
 * of the loaded tree and verifying results are valid.
 * Tests: (1) log-likelihood is finite & negative, (2) per-pattern likelihoods
 * are valid, (3) recomputation is deterministic, (4) partial likelihoods at
 * internal nodes are sane.
 *
 * @param tree  the PhyloTree (must be fully initialized with model, alignment,
 *              and partial likelihoods)
 * @return true if all tests pass
 */
bool testLikelihoodKernel(PhyloTree *tree);

/**
 * Verify JC transition matrix values against known reference values.
 * Tests computeTransMatrixEqualRate() from modelsubst.h
 * Prints PASS/FAIL for each test case.
 *
 * @return true if all tests pass
 */
bool testJCTransMatrix();

/**
 * Verify that ModelSubst::computeTransMatrix() (virtual, CPU)
 * produces identical results to computeTransMatrixEqualRate() (standalone, GPU-ready)
 * for all branches in the loaded tree.
 *
 * @param tree  the PhyloTree (must have model initialized)
 * @return true if all matrices match within tolerance
 */
bool testPrecomputedMatrices(PhyloTree *tree);

#endif // USE_OPENACC
#endif // PHYLOKERNEL_OPENACC_TEST_H
