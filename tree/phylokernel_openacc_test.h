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
// Step 3: Scalar Likelihood Kernel — test & verification
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
