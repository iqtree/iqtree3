/***************************************************************************
 *   OpenACC GPU Likelihood Computation for IQ-TREE                       *
 *   Scalar (plain C) likelihood kernels — state-space P(t) computation   *
 *   Used for all models (both reversible and non-reversible)             *
 *                                                                        *
 *   Reuses IQ-TREE's existing data structures:                           *
 *     - partial_lh arrays (flat double*, pattern-state layout)           *
 *     - PhyloNeighbor/PhyloNode tree structure                           *
 *     - computeTransMatrixEqualRate() for JC P(t) (from modelsubst.h)   *
 *     - SCALING_THRESHOLD / LOG_SCALING_THRESHOLD (from phylotree.h)     *
 *     - computeTraversalInfo() for post-order traversal                  *
 ***************************************************************************/

#ifndef PHYLOKERNEL_OPENACC_H
#define PHYLOKERNEL_OPENACC_H

#ifdef USE_OPENACC

// All constants (SCALING_THRESHOLD, LOG_SCALING_THRESHOLD, etc.)
// come from phylotree.h — no redefinitions here.
//
// The standalone GPU-compatible function computeTransMatrixEqualRate()
// lives in model/modelsubst.h — callable from both CPU and OpenACC kernels.
//
// Kernel member functions (computePartialLikelihoodGenericOpenACC,
// computeLikelihoodBranchGenericOpenACC) are declared in phylotree.h
// under #ifdef USE_OPENACC — they are PhyloTree members.
//
// Test declarations live in phylokernel_openacc_test.h

#endif // USE_OPENACC
#endif // PHYLOKERNEL_OPENACC_H
