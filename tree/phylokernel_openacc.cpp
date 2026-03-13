/***************************************************************************
 *   OpenACC GPU Likelihood Computation for IQ-TREE                       *
 *   Step 4: GPU-ready kernels with explicit indexing                     *
 *   Phase A: Refactored for OpenACC (no pragmas yet — CPU verifiable)    *
 *   Phase B: OpenACC pragmas + GPU data management                       *
 *     Step 9:  TIP-TIP kernel offloaded to GPU                          *
 *     Step 10: TIP-INTERNAL + INTERNAL-INTERNAL                          *
 *     Step 11: Log-likelihood reduction                                  *
 *     Step 12: Persistent GPU data                                       *
 *     Step 13: Hook into IQ-TREE kernel machinery                       *
 ***************************************************************************/

#ifdef USE_OPENACC

#include "phylokernel_openacc.h"
#include "phylotree.h"           // SCALING_THRESHOLD, PhyloTree, PhyloNeighbor, etc.
#include "phylokernelnew.h"      // computeTraversalInfo<Vec1d>, computeBounds<Vec1d>, computePartialInfo<Vec1d>
#include "vectorclass/vectorf64.h" // Vec1d (pure C++ scalar wrapper, no x86 intrinsics)
#include "model/modelsubst.h"    // computeTransMatrixEqualRate(), ModelSubst
#include "utils/tools.h"         // Params (for kernel_nonrev flag)
#ifdef USE_OPENACC_PROFILE
#include "utils/timeutil.h"      // getRealTime() for profiling
#endif

#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <openacc.h>            // Step 12: acc_is_present (future dynamic checks)

using namespace std;

#ifdef USE_OPENACC_PROFILE
// ==========================================================================
// Profiling instrumentation — accumulates timing across all calls.
// Compile with: cmake -DUSE_OPENACC=ON -DUSE_OPENACC_PROFILE=ON ..
// Runtime:      IQTREE_OPENACC_PROFILE=1 to activate timing
// Interval:     IQTREE_OPENACC_PROFILE_INTERVAL (default: 10 calls)
// ==========================================================================
static struct OpenACCProfile {
    bool enabled = false;
    bool initialized = false;
    int  call_count = 0;

    // Phase accumulators (seconds)
    double t_traversal_info = 0.0;
    double t_buffer_upload = 0.0;
    double t_persistent_upload = 0.0;
    double t_trans_mat_setup = 0.0;
    double t_offset_build = 0.0;
    double t_kernel_tip_tip = 0.0;
    double t_kernel_tip_int = 0.0;
    double t_kernel_int_int = 0.0;
    double t_reduction = 0.0;
    double t_d2h_pattern_lh = 0.0;
    double t_buffer_delete = 0.0;
    double t_host_postproc = 0.0;
    double t_total = 0.0;

    // Kernel launch counts
    int n_kernel_tip_tip = 0;
    int n_kernel_tip_int = 0;
    int n_kernel_int_int = 0;
    int n_reduction = 0;
    int total_levels = 0;
    int total_nodes_tip_tip = 0;
    int total_nodes_tip_int = 0;
    int total_nodes_int_int = 0;

    // D7: Derivative kernel profiling
    int    deriv_call_count = 0;
    double t_deriv_total = 0.0;
    double t_deriv_traversal = 0.0;     // computeTraversalInfo + stale partial recomp
    double t_deriv_trans_mat = 0.0;     // host-side P(t)/dP/d²P computation + freq scaling
    double t_deriv_trans_upload = 0.0;  // D4 update device() to GPU
    double t_deriv_tip_setup = 0.0;     // tip lookup table build + copyin (TIP-INT only)
    double t_deriv_kernel = 0.0;        // GPU derivative kernel execution
    double t_deriv_postproc = 0.0;      // ASC bias correction on host
    int    n_deriv_tip_int = 0;         // calls that took TIP-INT path
    int    n_deriv_int_int = 0;         // calls that took INT-INT path
    int    n_deriv_stale_recomp = 0;    // calls with non-empty traversal_info

    void print_summary() {
        if (call_count == 0 || t_total == 0.0) return;
        double t_kernels = t_kernel_tip_tip + t_kernel_tip_int + t_kernel_int_int;
        double t_accounted = t_traversal_info + t_buffer_upload +
            t_persistent_upload + t_trans_mat_setup + t_offset_build +
            t_kernels + t_reduction + t_d2h_pattern_lh + t_buffer_delete +
            t_host_postproc;
        double t_other = t_total - t_accounted;
        auto pct = [&](double t) { return 100.0 * t / t_total; };

        cout << "\n===== OpenACC Profiling Summary (" << call_count << " calls) =====" << endl;
        cout << fixed;
        cout << "  Total wall time:      " << t_total << " s" << endl;
        cout << "  ---" << endl;
        cout << "  computeTraversalInfo: " << t_traversal_info << " s (" << pct(t_traversal_info) << "%)" << endl;
        cout << "  buffer_plh upload:    " << t_buffer_upload << " s (" << pct(t_buffer_upload) << "%)" << endl;
        cout << "  persistent upload:    " << t_persistent_upload << " s (" << pct(t_persistent_upload) << "%)" << endl;
        cout << "  trans_mat setup:      " << t_trans_mat_setup << " s (" << pct(t_trans_mat_setup) << "%)" << endl;
        cout << "  offset build:         " << t_offset_build << " s (" << pct(t_offset_build) << "%)" << endl;
        cout << "  --- GPU Kernels ---" << endl;
        cout << "  TIP-TIP kernels:      " << t_kernel_tip_tip << " s (" << pct(t_kernel_tip_tip) << "%) ["
             << n_kernel_tip_tip << " launches, " << total_nodes_tip_tip << " nodes]" << endl;
        cout << "  TIP-INT kernels:      " << t_kernel_tip_int << " s (" << pct(t_kernel_tip_int) << "%) ["
             << n_kernel_tip_int << " launches, " << total_nodes_tip_int << " nodes]" << endl;
        cout << "  INT-INT kernels:      " << t_kernel_int_int << " s (" << pct(t_kernel_int_int) << "%) ["
             << n_kernel_int_int << " launches, " << total_nodes_int_int << " nodes]" << endl;
        cout << "  Reduction kernel:     " << t_reduction << " s (" << pct(t_reduction) << "%) ["
             << n_reduction << " launches]" << endl;
        cout << "  ---" << endl;
        cout << "  D->H pattern_lh:      " << t_d2h_pattern_lh << " s (" << pct(t_d2h_pattern_lh) << "%)" << endl;
        cout << "  buffer_plh delete:    " << t_buffer_delete << " s (" << pct(t_buffer_delete) << "%)" << endl;
        cout << "  Host post-proc:       " << t_host_postproc << " s (" << pct(t_host_postproc) << "%)" << endl;
        cout << "  Unaccounted:          " << t_other << " s (" << pct(t_other) << "%)" << endl;
        cout << "  Avg levels/call:      " << (double)total_levels / call_count << endl;

        // D7: Derivative kernel profiling summary
        if (deriv_call_count > 0) {
            double t_deriv_accounted = t_deriv_traversal + t_deriv_trans_mat +
                t_deriv_trans_upload + t_deriv_tip_setup + t_deriv_kernel +
                t_deriv_postproc;
            double t_deriv_other = t_deriv_total - t_deriv_accounted;
            auto dpct = [&](double t) { return 100.0 * t / t_deriv_total; };

            cout << "\n--- Derivative Kernel (" << deriv_call_count << " calls) ---" << endl;
            cout << "  Total deriv time:     " << t_deriv_total << " s" << endl;
            cout << "  Avg per call:         " << (t_deriv_total / deriv_call_count * 1e6) << " us" << endl;
            cout << "  ---" << endl;
            cout << "  Traversal+stale:      " << t_deriv_traversal << " s (" << dpct(t_deriv_traversal) << "%) ["
                 << n_deriv_stale_recomp << " stale recomps]" << endl;
            cout << "  Trans mat compute:    " << t_deriv_trans_mat << " s (" << dpct(t_deriv_trans_mat) << "%)" << endl;
            cout << "  Trans mat upload:     " << t_deriv_trans_upload << " s (" << dpct(t_deriv_trans_upload) << "%)" << endl;
            cout << "  Tip table setup:      " << t_deriv_tip_setup << " s (" << dpct(t_deriv_tip_setup) << "%) ["
                 << n_deriv_tip_int << " TIP-INT calls]" << endl;
            cout << "  GPU deriv kernel:     " << t_deriv_kernel << " s (" << dpct(t_deriv_kernel) << "%) ["
                 << n_deriv_tip_int << " TIP-INT, " << n_deriv_int_int << " INT-INT]" << endl;
            cout << "  Host post-proc:       " << t_deriv_postproc << " s (" << dpct(t_deriv_postproc) << "%)" << endl;
            cout << "  Unaccounted:          " << t_deriv_other << " s (" << dpct(t_deriv_other) << "%)" << endl;
        }

        cout << "=============================================" << endl;
    }
} acc_profile;
#endif // USE_OPENACC_PROFILE

// ==========================================================================
// P2: Level-based node batching — helper structures and batched kernels
// Groups independent tree nodes by dependency level and processes all nodes
// at the same level in a single GPU kernel launch using collapse(2) over
// (node, pattern). Reduces ~196 kernel launches to ~15 per evaluation.
// ==========================================================================

// Kernel type classification for batching
enum BatchKernelType { BATCH_TIP_TIP = 0, BATCH_TIP_INTERNAL = 1, BATCH_INTERNAL_INTERNAL = 2 };

// Per-level grouping of TraversalInfo pointers by kernel type
struct LevelBatch {
    vector<TraversalInfo*> tip_tip;
    vector<TraversalInfo*> tip_internal;
    vector<TraversalInfo*> internal_internal;
};

// Group traversal_info by dependency level and kernel type.
// Returns vector of LevelBatch (index = level), and sets max_level.
static vector<LevelBatch> groupByLevelAndType(
    vector<TraversalInfo> &traversal_info,
    int &max_level)
{
    // Compute dependency level for each node in post-order.
    // Leaf: level = -1. Internal: level = max(children_levels) + 1.
    // Post-order guarantees children appear before parents in traversal_info.
    unordered_map<int, int> node_level;
    max_level = 0;
    vector<int> info_levels(traversal_info.size());

    for (size_t ti = 0; ti < traversal_info.size(); ti++) {
        TraversalInfo &info = traversal_info[ti];
        PhyloNode *node = (PhyloNode*)info.dad_branch->node;
        PhyloNode *dad = info.dad;

        int max_child_level = -1;
        FOR_NEIGHBOR_IT(node, dad, it) {
            PhyloNode *child = (PhyloNode*)(*it)->node;
            if (!child->isLeaf()) {
                auto found = node_level.find(child->id);
                if (found != node_level.end()) {
                    if (found->second > max_child_level)
                        max_child_level = found->second;
                }
            }
        }
        int level = max_child_level + 1;
        node_level[node->id] = level;
        info_levels[ti] = level;
        if (level > max_level) max_level = level;
    }

    // Group by (level, kernel type)
    vector<LevelBatch> levels(max_level + 1);
    for (size_t ti = 0; ti < traversal_info.size(); ti++) {
        TraversalInfo &info = traversal_info[ti];
        int level = info_levels[ti];
        PhyloNode *node = (PhyloNode*)info.dad_branch->node;
        PhyloNode *dad = info.dad;

        // Determine kernel type from children
        PhyloNeighbor *left = NULL, *right = NULL;
        FOR_NEIGHBOR_IT(node, dad, it2) {
            if (!left) left = (PhyloNeighbor*)(*it2);
            else right = (PhyloNeighbor*)(*it2);
        }

        bool left_leaf = left->node->isLeaf();
        bool right_leaf = right->node->isLeaf();
        if (left_leaf && right_leaf) {
            levels[level].tip_tip.push_back(&info);
        } else if (left_leaf || right_leaf) {
            levels[level].tip_internal.push_back(&info);
        } else {
            levels[level].internal_internal.push_back(&info);
        }
    }

    return levels;
}

// ==========================================================================
// P2: Batched TIP-TIP kernel
// Processes all TIP-TIP nodes in a single kernel launch.
// No scaling needed (cherry nodes always have scale_num = 0).
//
// O4+O5 optimization: collapse(3) over (op, p, s) for coalesced writes.
// Single kernel (no scaling split needed — scale_num is always 0).
// ==========================================================================
static void batchedTipTip(
    size_t *offsets, int num_nodes,
    double *central_plh_base, UBYTE *central_scl_base,
    double *buffer_plh_base, int *tip_states_base,
    size_t total_lh, size_t total_scl, size_t buffer_size, size_t tip_states_size,
    int ptn_lower_int, int ptn_upper_int, int block_int, size_t nptn_stride)
{
    #pragma acc data \
        copyin(offsets[0:num_nodes*6]) \
        present(central_plh_base[0:total_lh], \
                central_scl_base[0:total_scl], \
                buffer_plh_base[0:buffer_size], \
                tip_states_base[0:tip_states_size])
    {
        #pragma acc parallel loop gang vector collapse(3) vector_length(128)
        for (int op = 0; op < num_nodes; op++) {
            for (int p = ptn_lower_int; p < ptn_upper_int; p++) {
                for (int s = 0; s < block_int; s++) {
                    size_t dad_off       = offsets[op * 6 + 0];
                    size_t dad_scl_off   = offsets[op * 6 + 1];
                    size_t tlh_left_off  = offsets[op * 6 + 2];
                    size_t tlh_right_off = offsets[op * 6 + 3];
                    int left_nid         = (int)offsets[op * 6 + 4];
                    int right_nid        = (int)offsets[op * 6 + 5];

                    int sl = tip_states_base[(size_t)left_nid * nptn_stride + p];
                    int sr = tip_states_base[(size_t)right_nid * nptn_stride + p];

                    // Only first thread per (op,p) writes scale_num (always 0 for cherry nodes)
                    if (s == 0)
                        central_scl_base[dad_scl_off + p] = 0;

                    central_plh_base[dad_off + (size_t)p * block_int + s] =
                        buffer_plh_base[tlh_left_off + (size_t)sl * block_int + s] *
                        buffer_plh_base[tlh_right_off + (size_t)sr * block_int + s];
                }
            }
        }
    } // end acc data
}

// ==========================================================================
// P2: Batched TIP-INTERNAL kernel
// Left child is a leaf (tip lookup), right child has partial likelihoods.
//
// O4+O5 optimization: Split into two kernels (same as INT-INT):
//   Kernel 1 (collapse(3)): Compute — right child dot product + left tip lookup.
//   Kernel 2 (collapse(2)): Scale propagation + scaling check.
// ==========================================================================
static void batchedTipInternal(
    size_t *offsets, int num_nodes,
    double *central_plh_base, UBYTE *central_scl_base,
    double *buffer_plh_base, int *tip_states_base,
    double *local_tip_plh,
    size_t total_lh, size_t total_scl, size_t buffer_size, size_t tip_states_size,
    size_t tip_unknown_size,
    int ptn_lower_int, int ptn_upper_int,
    int block_int, int nstates_int, int nstatesqr_int,
    size_t nptn_stride, size_t state_unknown)
{
    // offsets layout per node [8]: dad_plh, dad_scl, right_plh, right_scl,
    //                              eright (buffer_plh), tip_lh_left (buffer_plh),
    //                              left_node_id, (pad)
    #pragma acc data \
        copyin(offsets[0:num_nodes*8]) \
        present(central_plh_base[0:total_lh], \
                central_scl_base[0:total_scl], \
                buffer_plh_base[0:buffer_size], \
                tip_states_base[0:tip_states_size], \
                local_tip_plh[0:tip_unknown_size])
    {
        // Kernel 1: Compute partial likelihoods (collapse(3) — coalesced, low registers)
        #pragma acc parallel loop gang vector collapse(3) vector_length(128)
        for (int op = 0; op < num_nodes; op++) {
            for (int p = ptn_lower_int; p < ptn_upper_int; p++) {
                for (int s = 0; s < block_int; s++) {
                    size_t dad_off       = offsets[op * 8 + 0];
                    size_t right_plh_off = offsets[op * 8 + 2];
                    size_t eright_off    = offsets[op * 8 + 4];
                    size_t tlh_left_off  = offsets[op * 8 + 5];
                    int left_nid         = (int)offsets[op * 8 + 6];

                    int state_left = tip_states_base[(size_t)left_nid * nptn_stride + p];

                    int cat = s / nstates_int;
                    int state = s % nstates_int;
                    int emat_base = cat * nstatesqr_int + state * nstates_int;
                    int plh_base = p * block_int + cat * nstates_int;

                    // Right child: dot product P(t) × partial_lh
                    double vright = 0.0;
                    #pragma acc loop seq
                    for (int k = 0; k < nstates_int; k++) {
                        vright += buffer_plh_base[eright_off + emat_base + k]
                                * central_plh_base[right_plh_off + plh_base + k];
                    }
                    // Left child: precomputed tip lookup
                    double vleft_val = buffer_plh_base[tlh_left_off + state_left * block_int + s];
                    central_plh_base[dad_off + (size_t)p * block_int + s] = vleft_val * vright;
                }
            }
        }

        // Kernel 2: Scale propagation + scaling check (collapse(2) — lightweight, per-pattern)
        #pragma acc parallel loop gang vector collapse(2) vector_length(128)
        for (int op = 0; op < num_nodes; op++) {
            for (int p = ptn_lower_int; p < ptn_upper_int; p++) {
                size_t dad_off       = offsets[op * 8 + 0];
                size_t dad_scl_off   = offsets[op * 8 + 1];
                size_t right_scl_off = offsets[op * 8 + 3];

                // Scale_num from right child only (left is a leaf, no scaling)
                central_scl_base[dad_scl_off + p] = central_scl_base[right_scl_off + p];

                // Scaling check
                double lh_max = 0.0;
                for (int s = 0; s < block_int; s++) {
                    double v = central_plh_base[dad_off + (size_t)p * block_int + s];
                    if (v > lh_max) lh_max = v;
                }
                if (lh_max == 0.0) {
                    for (int s = 0; s < block_int; s++)
                        central_plh_base[dad_off + (size_t)p * block_int + s] =
                            local_tip_plh[state_unknown * nstates_int + (s % nstates_int)];
                    central_scl_base[dad_scl_off + p] += 4;
                } else if (lh_max < SCALING_THRESHOLD) {
                    for (int s = 0; s < block_int; s++)
                        central_plh_base[dad_off + (size_t)p * block_int + s] =
                            ldexp(central_plh_base[dad_off + (size_t)p * block_int + s], SCALING_THRESHOLD_EXP);
                    central_scl_base[dad_scl_off + p] += 1;
                }
            }
        }
    } // end acc data
}

// ==========================================================================
// P2: Batched INTERNAL-INTERNAL kernel
// Both children are internal nodes with partial likelihoods.
// HOT PATH — two matrix-vector products + Hadamard product.
//
// O4+O5 optimization: Split into two kernels:
//   Kernel 1 (collapse(3) over op,p,s): Compute partial likelihoods.
//     - Each thread computes ONE output element dad_plh[p*block+s].
//     - Consecutive threads (consecutive s) write to consecutive memory → coalesced.
//     - All threads with same (op,p) read same child partial_lh values → broadcast.
//     - Only 2 FP64 accumulators per thread → ~20-25 registers → high occupancy.
//   Kernel 2 (collapse(2) over op,p): Scale propagation + scaling check.
//     - Per-pattern: propagate child scale counts, check for underflow.
//     - Lightweight (no dot products), so old coalescing pattern is acceptable.
//
// Before O4+O5: collapse(2) over (op,p), sequential for(s) → 73% memory waste,
//               132 registers, 18.75% occupancy, scheduler idle 84%.
// ==========================================================================
static void batchedInternalInternal(
    size_t *offsets, int num_nodes,
    double *central_plh_base, UBYTE *central_scl_base,
    double *buffer_plh_base,
    double *local_tip_plh,
    size_t total_lh, size_t total_scl, size_t buffer_size,
    size_t tip_unknown_size,
    int ptn_lower_int, int ptn_upper_int,
    int block_int, int nstates_int, int nstatesqr_int,
    size_t state_unknown)
{
    // offsets layout per node [8]: dad_plh, dad_scl, left_plh, right_plh,
    //                              left_scl, right_scl, eleft (buffer_plh), eright (buffer_plh)
    #pragma acc data \
        copyin(offsets[0:num_nodes*8]) \
        present(central_plh_base[0:total_lh], \
                central_scl_base[0:total_scl], \
                buffer_plh_base[0:buffer_size], \
                local_tip_plh[0:tip_unknown_size])
    {
        // Kernel 1: Compute partial likelihoods (collapse(3) — coalesced, low registers)
        // Each thread computes ONE output value: dad_plh[p*block+s] = vleft * vright
        // where vleft = dot(P_left[s,:], left_plh[p,:]), vright = dot(P_right[s,:], right_plh[p,:])
        #pragma acc parallel loop gang vector collapse(3) vector_length(128)
        for (int op = 0; op < num_nodes; op++) {
            for (int p = ptn_lower_int; p < ptn_upper_int; p++) {
                for (int s = 0; s < block_int; s++) {
                    size_t dad_off       = offsets[op * 8 + 0];
                    size_t left_plh_off  = offsets[op * 8 + 2];
                    size_t right_plh_off = offsets[op * 8 + 3];
                    size_t eleft_off     = offsets[op * 8 + 6];
                    size_t eright_off    = offsets[op * 8 + 7];

                    int cat = s / nstates_int;
                    int state = s % nstates_int;
                    int emat_base = cat * nstatesqr_int + state * nstates_int;
                    int plh_base = p * block_int + cat * nstates_int;

                    double vleft = 0.0, vright = 0.0;
                    #pragma acc loop seq
                    for (int k = 0; k < nstates_int; k++) {
                        vleft  += buffer_plh_base[eleft_off + emat_base + k]
                                * central_plh_base[left_plh_off + plh_base + k];
                        vright += buffer_plh_base[eright_off + emat_base + k]
                                * central_plh_base[right_plh_off + plh_base + k];
                    }
                    central_plh_base[dad_off + (size_t)p * block_int + s] = vleft * vright;
                }
            }
        }

        // Kernel 2: Scale propagation + scaling check (collapse(2) — lightweight, per-pattern)
        #pragma acc parallel loop gang vector collapse(2) vector_length(128)
        for (int op = 0; op < num_nodes; op++) {
            for (int p = ptn_lower_int; p < ptn_upper_int; p++) {
                size_t dad_off       = offsets[op * 8 + 0];
                size_t dad_scl_off   = offsets[op * 8 + 1];
                size_t left_scl_off  = offsets[op * 8 + 4];
                size_t right_scl_off = offsets[op * 8 + 5];

                // Propagate scale counts from both children
                central_scl_base[dad_scl_off + p] =
                    central_scl_base[left_scl_off + p] + central_scl_base[right_scl_off + p];

                // Scaling check
                double lh_max = 0.0;
                for (int s = 0; s < block_int; s++) {
                    double v = central_plh_base[dad_off + (size_t)p * block_int + s];
                    if (v > lh_max) lh_max = v;
                }
                if (lh_max == 0.0) {
                    for (int s = 0; s < block_int; s++)
                        central_plh_base[dad_off + (size_t)p * block_int + s] =
                            local_tip_plh[state_unknown * nstates_int + (s % nstates_int)];
                    central_scl_base[dad_scl_off + p] += 4;
                } else if (lh_max < SCALING_THRESHOLD) {
                    for (int s = 0; s < block_int; s++)
                        central_plh_base[dad_off + (size_t)p * block_int + s] =
                            ldexp(central_plh_base[dad_off + (size_t)p * block_int + s], SCALING_THRESHOLD_EXP);
                    central_scl_base[dad_scl_off + p] += 1;
                }
            }
        }
    } // end acc data
}

// ==========================================================================
// Step 4 Phase A: GPU-ready Partial Likelihood Kernel
// Refactored from Step 3 scalar kernels:
//   - Raw pointers extracted before loops (no struct dereference in kernels)
//   - Running pointer arithmetic → explicit index calculations
//   - aln->at(ptn) state lookups → precomputed flat arrays
//   - memset/memcpy → plain loops
//   - ASSERT removed from kernel regions
// ==========================================================================

void PhyloTree::computePartialLikelihoodGenericOpenACC(TraversalInfo &info, size_t ptn_lower, size_t ptn_upper, int packet_id) {
    // NOTE: packet_id is not used yet in the OpenACC kernel. It is kept to match the
    // ComputePartialLikelihoodType function pointer signature (phylotree.h).
    (void)packet_id;  // suppress unused-variable warning

    PhyloNeighbor *dad_branch = info.dad_branch;
    PhyloNode *dad = info.dad;

    ASSERT(dad);
    PhyloNode *node = (PhyloNode*)(dad_branch->node);

    ASSERT(dad_branch->direction != UNDEFINED_DIRECTION);

    size_t nstates = aln->num_states;
    size_t nstatesqr = nstates * nstates;

    if (node->isLeaf()) {
        return;
    }

    ASSERT(node->degree() >= 3);

    size_t ptn, c;
    size_t orig_ntn = aln->size();
    size_t ncat = site_rate->getNRate();
    size_t i, x;
    size_t block = nstates * ncat;

    // internal node
    // left, right: PhyloNeighbor* pointers to the two child branches.
    // These hold partial_lh (child partial likelihoods), scale_num,
    // branch length, and node identity.
    PhyloNeighbor *left = NULL, *right = NULL;
    FOR_NEIGHBOR_IT(node, dad, it) {
        if (!left) left = (PhyloNeighbor*)(*it); else right = (PhyloNeighbor*)(*it);
    }

    // echildren: flat array of precomputed transition probability matrices P(t).
    // Filled by IQ-TREE's model machinery (via eigendecomposition of the rate
    // matrix) BEFORE this kernel is called.  By the time we access it here,
    // it is just a plain double array — no Eigen library calls at runtime.
    //
    // eleft / eright: pointers into echildren for the left and right branch
    // P matrices respectively.  Each is a K×K block (16 doubles for DNA).
    //   eleft[s*block + x] = P(x → s | t_left)
    //
    // NOTE: The "e" prefix is IQ-TREE convention (historical, from "eigen").
    // Our OpenACC JC kernels use computeTransMatrixEqualRate() (Step 2) which
    // computes P(t) via the direct JC closed-form formula — no eigendecomposition.
    // However, we keep the same variable names for consistency with the rest of
    // the IQ-TREE codebase since the data layout is identical.
    double *echildren = info.echildren;
    double *partial_lh_leaves = info.partial_lh_leaves;

    double *eleft = echildren, *eright = echildren + block*nstates;

    if ((!left->node->isLeaf() && right->node->isLeaf())) {
        PhyloNeighbor *tmp = left;
        left = right;
        right = tmp;
        double *etmp = eleft;
        eleft = eright;
        eright = etmp;
    }

    // ====== A1: Extract raw pointers from structs (before any kernel loops) ======
    // __restrict__ tells the compiler these arrays don't alias each other,
    // enabling better register allocation and load/store reordering on GPU.
    double * __restrict__ dad_partial_lh  = dad_branch->partial_lh;
    UBYTE  * __restrict__ dad_scale_num   = dad_branch->scale_num;

    if (node->degree() > 3) {

        /*--------------------- multifurcating node ------------------*/
        // Note: Multifurcating nodes are rare. Keep mostly sequential for now.
        // GPU parallelization of multifurcating case deferred to later.

        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            // A2: explicit indexing for output
            for (i = 0; i < block; i++)
                dad_partial_lh[ptn*block + i] = 1.0;
            dad_scale_num[ptn] = 0;

            double *partial_lh_leaf = partial_lh_leaves;
            double *echild = echildren;
            int child_idx = 0;

            FOR_NEIGHBOR_IT(node, dad, it) {
                PhyloNeighbor *child = (PhyloNeighbor*)*it;
                if (child->node->isLeaf()) {
                    // external node — A3: precompute state
                    int state_child;
                    if (isRootLeaf(child->node))
                        state_child = 0;
                    else
                        state_child = (ptn < orig_ntn) ? (aln->at(ptn))[child->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn][child->node->id];
                    double *child_lh = partial_lh_leaf + state_child*block;
                    for (c = 0; c < block; c++) {
                        dad_partial_lh[ptn*block + c] *= child_lh[c];
                    }
                    partial_lh_leaf += (aln->STATE_UNKNOWN+1)*block;
                } else {
                    // internal node — A2: explicit indexing
                    double *child_partial_lh = child->partial_lh;
                    UBYTE  *child_scale_num  = child->scale_num;
                    dad_scale_num[ptn] += child_scale_num[ptn];

                    for (c = 0; c < ncat; c++) {
                        for (x = 0; x < nstates; x++) {
                            double vchild = 0.0;
                            for (i = 0; i < nstates; i++) {
                                vchild += echild[c*nstatesqr + x*nstates + i] * child_partial_lh[ptn*block + c*nstates + i];
                            }
                            dad_partial_lh[ptn*block + c*nstates + x] *= vchild;
                        }
                    }
                } // if
                echild += block*nstates;
            } // FOR_NEIGHBOR

            // Find max for scaling
            double lh_max = dad_partial_lh[ptn*block];
            for (i = 1; i < block; i++) {
                double v = dad_partial_lh[ptn*block + i];
                if (v > lh_max) lh_max = v;
            }

            // A4: replace memcpy with plain loop, A5: remove ASSERT
            if (lh_max == 0.0) {
                // for very shitty data
                for (c = 0; c < ncat; c++)
                    for (i = 0; i < nstates; i++)
                        dad_partial_lh[ptn*block + c*nstates + i] = tip_partial_lh[aln->STATE_UNKNOWN*nstates + i];
                dad_scale_num[ptn] += 4;
            } else if (lh_max < SCALING_THRESHOLD) {
                for (i = 0; i < block; i++) {
                    dad_partial_lh[ptn*block + i] = ldexp(dad_partial_lh[ptn*block + i], SCALING_THRESHOLD_EXP);
                }
                dad_scale_num[ptn] += 1;
            }

        } // for ptn

        // end multifurcating treatment
    } else if (left->node->isLeaf() && right->node->isLeaf()) {

        /*--------------------- TIP-TIP (cherry) case ------------------*/

        double * __restrict__ tip_lh_left = partial_lh_leaves;
        double * __restrict__ tip_lh_right = partial_lh_leaves + (aln->STATE_UNKNOWN+1)*block;

        if (isRootLeaf(right->node)) {
            // swap so that left node is the root
            PhyloNeighbor *tmp = left;
            left = right;
            right = tmp;
            double *etmp = eleft;
            eleft = eright;
            eright = etmp;
            etmp = tip_lh_left;
            tip_lh_left = tip_lh_right;
            tip_lh_right = etmp;
        }

        // A3: Precompute per-pattern tip states into flat arrays (CPU side)
        size_t nptn_range = ptn_upper - ptn_lower;
        int *states_left  = new int[nptn_range];
        int *states_right = new int[nptn_range];
        bool left_is_root = isRootLeaf(left->node);
        int left_node_id  = left->node->id;
        int right_node_id = right->node->id;

        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            size_t idx = ptn - ptn_lower;
            if (left_is_root)
                states_left[idx] = 0;
            else
                states_left[idx] = (ptn < orig_ntn) ? (aln->at(ptn))[left_node_id] : model_factory->unobserved_ptns[ptn-orig_ntn][left_node_id];
            states_right[idx] = (ptn < orig_ntn) ? (aln->at(ptn))[right_node_id] : model_factory->unobserved_ptns[ptn-orig_ntn][right_node_id];
        }

        // Step 9: TIP-TIP kernel offloaded to GPU via OpenACC
        // Data flow: copyin states + tip lookup tables; partial_lh and
        // scale_num are persistent on GPU (Step 12: present instead of copyout).
        // Parallelism: gang over patterns, vector over states (matches PoC)
        {
            size_t tip_lh_size = (aln->STATE_UNKNOWN + 1) * block;
            size_t plh_offset = ptn_lower * block;
            size_t plh_count  = (ptn_upper - ptn_lower) * block;
            size_t scl_offset = ptn_lower;
            size_t scl_count  = ptn_upper - ptn_lower;

            #pragma acc data \
                copyin(states_left[0:nptn_range], states_right[0:nptn_range], \
                       tip_lh_left[0:tip_lh_size], tip_lh_right[0:tip_lh_size]) \
                present(dad_partial_lh[plh_offset:plh_count], \
                        dad_scale_num[scl_offset:scl_count])
            {
                // Zero scale_num for TIP-TIP (no scaling at cherry nodes)
                int block_int = (int)block;
                #pragma acc parallel loop gang vector
                for (int p = (int)ptn_lower; p < (int)ptn_upper; p++)
                    dad_scale_num[p] = 0;

                // Main TIP-TIP kernel: element-wise product of precomputed tip lookups
                #pragma acc parallel loop gang
                for (int p = (int)ptn_lower; p < (int)ptn_upper; p++) {
                    int idx = p - (int)ptn_lower;
                    int sl = states_left[idx];
                    int sr = states_right[idx];
                    #pragma acc loop vector
                    for (int s = 0; s < block_int; s++) {
                        dad_partial_lh[p * block_int + s] =
                            tip_lh_left[sl * block_int + s] * tip_lh_right[sr * block_int + s];
                    }
                 }
            } // end acc data
        }

        delete [] states_left;
        delete [] states_right;

    } else if (left->node->isLeaf() && !right->node->isLeaf()) {

        /*--------------------- TIP-INTERNAL NODE case ------------------*/
        // Step 10: Offloaded to GPU via OpenACC
        // Left child is a leaf (tip lookup), right child has partial likelihoods.
        // For each pattern: dad[s] = tip_lh_left[state][s] * (P_right × right_plh)[s]
        // Parallelism: gang over patterns, vector over output states,
        //              sequential inner dot product (matches PoC compositehadamard)

        // A1: Extract raw pointers (__restrict__ for no-alias optimization)
        double * __restrict__ right_partial_lh = right->partial_lh;
        UBYTE  * __restrict__ right_scale_num  = right->scale_num;
        double * __restrict__ tip_lh_left = partial_lh_leaves;

        // A3: Precompute per-pattern tip states (CPU side, before GPU region)
        size_t nptn_range = ptn_upper - ptn_lower;
        int *states_left = new int[nptn_range];
        bool left_is_root = isRootLeaf(left->node);
        int left_node_id = left->node->id;

        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            size_t idx = ptn - ptn_lower;
            if (left_is_root)
                states_left[idx] = 0;
            else
                states_left[idx] = (ptn < orig_ntn) ? (aln->at(ptn))[left_node_id] : model_factory->unobserved_ptns[ptn-orig_ntn][left_node_id];
        }

        // Data sizes for OpenACC transfers
        {
            size_t tip_lh_size = (aln->STATE_UNKNOWN + 1) * block;
            size_t eright_size = block * nstates;            // K×K P matrix (ncat=1)
            size_t plh_offset  = ptn_lower * block;
            size_t plh_count   = (ptn_upper - ptn_lower) * block;
            size_t scl_offset  = ptn_lower;
            size_t scl_count   = ptn_upper - ptn_lower;
            size_t tip_unknown_size = (aln->STATE_UNKNOWN + 1) * nstates;
            size_t nstatesqr_local = nstates * nstates;
            // Capture host-side class members into locals for GPU access.
            // Class member pointers (this->tip_partial_lh, this->aln, etc.)
            // cannot be dereferenced on GPU — the compiler would try to map
            // 'this' to the device, causing a PRESENT clause failure.
            size_t state_unknown = aln->STATE_UNKNOWN;
            double *local_tip_plh = tip_partial_lh;

            // Step 12: persistent data uses present(); per-node data uses copyin.
            #pragma acc data \
                copyin(states_left[0:nptn_range], \
                       tip_lh_left[0:tip_lh_size], \
                       eright[0:eright_size]) \
                present(right_partial_lh[plh_offset:plh_count], \
                        right_scale_num[scl_offset:scl_count], \
                        local_tip_plh[0:tip_unknown_size], \
                        dad_partial_lh[plh_offset:plh_count], \
                        dad_scale_num[scl_offset:scl_count])
            {
                // Kernel 1: Compute partial likelihoods
                // gang over patterns, vector over output states, sequential dot product
                // int loop vars for faster 32-bit GPU math; division hoisted out of inner loop
                int block_int = (int)block;
                int nstates_int = (int)nstates;
                int nstatesqr_int = (int)nstatesqr_local;
                #pragma acc parallel loop gang
                for (int p = (int)ptn_lower; p < (int)ptn_upper; p++) {
                    int idx = p - (int)ptn_lower;
                    int state_left = states_left[idx];

                    // Copy scale_num from right child
                    dad_scale_num[p] = right_scale_num[p];

                    #pragma acc loop vector
                    for (int s = 0; s < block_int; s++) {
                        int cat = s / nstates_int;
                        int state = s % nstates_int;
                        int eright_base = cat * nstatesqr_int + state * nstates_int;
                        int plh_base = p * block_int + cat * nstates_int;
                        // Right child: dot product P_right × right_partial_lh
                        double vright = 0.0;
                        for (int k = 0; k < nstates_int; k++) {
                            vright += eright[eright_base + k]
                                    * right_partial_lh[plh_base + k];
                        }
                        // Left child: precomputed tip lookup (no matrix multiply)
                        double vleft_val = tip_lh_left[state_left * block_int + s];
                        dad_partial_lh[p * block_int + s] = vleft_val * vright;
                    }
                }

                // Kernel 2: Scaling check (separate pass, matches PoC pattern)
                // Uses gang+vector with reduction(max:) per pattern
                #pragma acc parallel loop gang
                for (int p = (int)ptn_lower; p < (int)ptn_upper; p++) {
                    double lh_max = 0.0;
                    #pragma acc loop vector reduction(max:lh_max)
                    for (int s = 0; s < block_int; s++) {
                        double v = dad_partial_lh[p * block_int + s];
                        if (v > lh_max) lh_max = v;
                    }
                    if (lh_max == 0.0) {
                        // All-zero: replace with unknown-state partial
                        #pragma acc loop seq
                        for (int s = 0; s < block_int; s++)
                            dad_partial_lh[p * block_int + s] = local_tip_plh[state_unknown * nstates_int + (s % nstates_int)];
                        dad_scale_num[p] += 4;
                    } else if (lh_max < SCALING_THRESHOLD) {
                        // Underflow: scale up by 2^256
                        #pragma acc loop seq
                        for (int s = 0; s < block_int; s++)
                            dad_partial_lh[p * block_int + s] = ldexp(dad_partial_lh[p * block_int + s], SCALING_THRESHOLD_EXP);
                        dad_scale_num[p] += 1;
                    }
                }
            } // end acc data
        }

        delete [] states_left;

    } else {

        /*--------------------- INTERNAL-INTERNAL NODE case ------------------*/
        // Step 10: Offloaded to GPU via OpenACC
        // HOT PATH — both children are internal nodes with partial likelihoods.
        // For each pattern: dad[s] = (P_left × left_plh)[s] * (P_right × right_plh)[s]
        // Matches PoC compositehadamard parallelism exactly:
        //   gang over patterns, vector over states, sequential dot product
        // Scaling done in a separate kernel pass (matches PoC pattern).

        // A1: Extract raw pointers (__restrict__ for no-alias optimization)
        double * __restrict__ left_partial_lh  = left->partial_lh;
        double * __restrict__ right_partial_lh = right->partial_lh;
        UBYTE  * __restrict__ left_scale_num   = left->scale_num;
        UBYTE  * __restrict__ right_scale_num  = right->scale_num;

        // Data sizes for OpenACC transfers
        {
            size_t eleft_size  = block * nstates;            // K×K P matrix
            size_t eright_size = block * nstates;
            size_t plh_offset  = ptn_lower * block;
            size_t plh_count   = (ptn_upper - ptn_lower) * block;
            size_t scl_offset  = ptn_lower;
            size_t scl_count   = ptn_upper - ptn_lower;
            size_t tip_unknown_size = (aln->STATE_UNKNOWN + 1) * nstates;
            size_t nstatesqr_local = nstates * nstates;
            // Capture host-side class members into locals for GPU access.
            // Class member pointers (this->tip_partial_lh, this->aln, etc.)
            // cannot be dereferenced on GPU — the compiler would try to map
            // 'this' to the device, causing a PRESENT clause failure.
            size_t state_unknown = aln->STATE_UNKNOWN;
            double *local_tip_plh = tip_partial_lh;

            // Step 12: persistent data uses present(); per-node data uses copyin.
            #pragma acc data \
                copyin(eleft[0:eleft_size], eright[0:eright_size]) \
                present(left_partial_lh[plh_offset:plh_count], \
                        right_partial_lh[plh_offset:plh_count], \
                        left_scale_num[scl_offset:scl_count], \
                        right_scale_num[scl_offset:scl_count], \
                        local_tip_plh[0:tip_unknown_size], \
                        dad_partial_lh[plh_offset:plh_count], \
                        dad_scale_num[scl_offset:scl_count])
            {
                // Kernel 1: Compute partial likelihoods (two matrix-vector products + Hadamard)
                // gang over patterns, vector over output states, sequential dot product
                // int loop vars for faster 32-bit GPU math; division hoisted out of inner loop
                int block_int = (int)block;
                int nstates_int = (int)nstates;
                int nstatesqr_int = (int)nstatesqr_local;
                #pragma acc parallel loop gang
                for (int p = (int)ptn_lower; p < (int)ptn_upper; p++) {
                    // Propagate scale counts from both children
                    dad_scale_num[p] = left_scale_num[p] + right_scale_num[p];

                    #pragma acc loop vector
                    for (int s = 0; s < block_int; s++) {
                        int cat = s / nstates_int;
                        int state = s % nstates_int;
                        int emat_base = cat * nstatesqr_int + state * nstates_int;
                        int plh_base = p * block_int + cat * nstates_int;
                        double vleft = 0.0, vright = 0.0;
                        for (int k = 0; k < nstates_int; k++) {
                            vleft  += eleft[emat_base + k]
                                    * left_partial_lh[plh_base + k];
                            vright += eright[emat_base + k]
                                    * right_partial_lh[plh_base + k];
                        }
                        dad_partial_lh[p * block_int + s] = vleft * vright;
                    }
                }

                // Kernel 2: Scaling check (separate pass, matches PoC pattern)
                // Uses gang+vector with reduction(max:) per pattern
                #pragma acc parallel loop gang
                for (int p = (int)ptn_lower; p < (int)ptn_upper; p++) {
                    double lh_max = 0.0;
                    #pragma acc loop vector reduction(max:lh_max)
                    for (int s = 0; s < block_int; s++) {
                        double v = dad_partial_lh[p * block_int + s];
                        if (v > lh_max) lh_max = v;
                    }
                    if (lh_max == 0.0) {
                        // All-zero: replace with unknown-state partial
                        #pragma acc loop seq
                        for (int s = 0; s < block_int; s++)
                            dad_partial_lh[p * block_int + s] = local_tip_plh[state_unknown * nstates_int + (s % nstates_int)];
                        dad_scale_num[p] += 4;
                    } else if (lh_max < SCALING_THRESHOLD) {
                        // Underflow: scale up by 2^256
                        #pragma acc loop seq
                        for (int s = 0; s < block_int; s++)
                            dad_partial_lh[p * block_int + s] = ldexp(dad_partial_lh[p * block_int + s], SCALING_THRESHOLD_EXP);
                        dad_scale_num[p] += 1;
                    }
                }
            } // end acc data
        }

    }
}

// ==========================================================================
// Step 4 Phase A: GPU-ready Branch Likelihood Kernel
// Refactored from Step 3:
//   - State lookups precomputed into flat arrays
//   - Running pointers → explicit indexing in reduction loops
//   - memset → plain loops
//   - ASSERT removed from kernel regions
// ==========================================================================

double PhyloTree::computeLikelihoodBranchGenericOpenACC(PhyloNeighbor *dad_branch, PhyloNode *dad, bool save_log_value) {

    // One-time verification message
    static bool openacc_kernel_printed = false;
    if (!openacc_kernel_printed) {
        cout << "OpenACC: Using GPU-ready (explicit indexing) likelihood kernel "
             << "(computePartialLikelihoodGenericOpenACC + "
             << "computeLikelihoodBranchGenericOpenACC)" << endl;
        openacc_kernel_printed = true;
    }

#ifdef USE_OPENACC_PROFILE
    // Profiling: initialize from env var on first call, start total timer
    double prof_t0 = 0.0, prof_t1 = 0.0;
    if (!acc_profile.initialized) {
        const char *env = getenv("IQTREE_OPENACC_PROFILE");
        acc_profile.enabled = (env && env[0] != '0');
        acc_profile.initialized = true;
        if (acc_profile.enabled)
            cout << "OpenACC: Profiling ENABLED (IQTREE_OPENACC_PROFILE=1)" << endl;
    }
    bool profiling = acc_profile.enabled;
    if (profiling) {
        prof_t0 = getRealTime();
        acc_profile.call_count++;
    }
#endif

    // Supports both rooted and unrooted trees

    PhyloNode *node = (PhyloNode*) dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*) node->findNeighbor(dad);
    if (!central_partial_lh)
        initializeAllPartialLh();
    if (node->isLeaf() || (dad_branch->direction == AWAYFROM_ROOT && !isRootLeaf(dad))) {
        PhyloNode *tmp_node = dad;
        dad = node;
        node = tmp_node;
        PhyloNeighbor *tmp_nei = dad_branch;
        dad_branch = node_branch;
        node_branch = tmp_nei;
    }

    // Build traversal order and precompute P(t) / tip lookup tables.
#ifdef USE_OPENACC_PROFILE
    if (profiling) prof_t1 = getRealTime();
#endif
    Params::getInstance().kernel_nonrev = true;
    computeTraversalInfo<Vec1d>(node, dad, false);
#ifdef USE_OPENACC_PROFILE
    if (profiling) acc_profile.t_traversal_info += getRealTime() - prof_t1;
#endif

    // P2: Batch upload buffer_partial_lh — contains all per-node P(t) matrices
    // and tip lookup tables, already filled by computeTraversalInfo().
    // One upload replaces ~98 per-node copyin calls.
    // D3: If buffer is already resident from a previous call, delete first
    // (host may have refilled it with new traversal data).
#ifdef USE_OPENACC_PROFILE
    if (profiling) prof_t1 = getRealTime();
#endif
    if (gpu_buffer_plh_resident && gpu_buffer_plh_ptr) {
        #pragma acc exit data delete(gpu_buffer_plh_ptr[0:gpu_buffer_plh_size])
        gpu_buffer_plh_resident = false;
        gpu_buffer_plh_ptr = nullptr;
    }
    gpu_buffer_plh_size = getBufferPartialLhSize();
    double *local_buffer_plh = buffer_partial_lh;
    #pragma acc enter data copyin(local_buffer_plh[0:gpu_buffer_plh_size])
#ifdef USE_OPENACC_PROFILE
    if (profiling) {
        #pragma acc wait
        acc_profile.t_buffer_upload += getRealTime() - prof_t1;
    }
#endif

    double tree_lh = 0.0;
    size_t nstates = aln->num_states;
    size_t nstatesqr = nstates*nstates;
    size_t ncat = site_rate->getNRate();

    size_t block = ncat * nstates;
    size_t ptn;
    size_t c, i, x;
    size_t orig_nptn = aln->size();
    size_t nptn = aln->size()+model_factory->unobserved_ptns.size();

    vector<size_t> limits;
    computeBounds<Vec1d>(num_threads, num_packets, nptn, limits);

#ifdef USE_OPENACC_PROFILE
    if (profiling) prof_t1 = getRealTime();
#endif
    double *trans_mat = new double[block*nstates];
    for (c = 0; c < ncat; c++) {
        double len = site_rate->getRate(c)*dad_branch->length;
        double prop = site_rate->getProp(c);
        double *this_trans_mat = &trans_mat[c*nstatesqr];
        model->computeTransMatrix(len, this_trans_mat);
        for (i = 0; i < nstatesqr; i++)
            this_trans_mat[i] *= prop;
    }
    if (!rooted) {
        // For unrooted trees, pre-multiply state frequencies into transition matrix
        // at the virtual root edge (matches CPU pattern in phylokernelnonrev.h:1121-1130)
        double state_freq[64]; // max states: codon=61, AA=20, DNA=4
        model->getStateFrequency(state_freq);
        for (c = 0; c < ncat; c++) {
            double *this_trans_mat = &trans_mat[c*nstatesqr];
            for (i = 0; i < nstates; i++) {
                for (x = 0; x < nstates; x++)
                    this_trans_mat[x] *= state_freq[i];
                this_trans_mat += nstates;
            }
        }
    }

    // ====== Persistent GPU data management ======
    // Upload central_partial_lh, central_scale_num, ptn_freq, ptn_invar to GPU
    // ONCE on the first call. Subsequent calls find data already present().
    // Only _pattern_lh is copied back to host per call (needed by prob_const
    // correction and IQ-TREE callers). This eliminates ~6.5 GB of PCIe
    // round-trips per evaluation for large datasets.
    //
    // Size computation replicates initializeAllPartialLh() allocation logic:
    //   central_partial_lh: max_lh_slots * lh_block + 4 + tip_alloc_size
    //   central_scale_num:  max_lh_slots * scale_block
    size_t nptn_ncat = nptn * ncat;  // for _pattern_lh_cat

    // Capture class member pointers for OpenACC directives.
    // Class member pointers (this->central_partial_lh, etc.) cause the compiler
    // to map 'this' to the device — capture into locals to avoid that.
    double *local_central_plh = central_partial_lh;
    UBYTE  *local_central_scl = central_scale_num;
    double *local_ptn_freq = ptn_freq;
    double *local_ptn_invar = ptn_invar;
    double *local_pattern_lh = _pattern_lh;
    double *local_pattern_lh_cat = _pattern_lh_cat;

    if (!gpu_data_resident) {
#ifdef USE_OPENACC_PROFILE
        // Profiling: end trans_mat segment 1 (trans_mat computation + local captures)
        // before timing the persistent upload separately
        if (profiling) {
            acc_profile.t_trans_mat_setup += getRealTime() - prof_t1;
            prof_t1 = getRealTime();
        }
#endif
        size_t nptn_safe = get_safe_upper_limit(aln->size())
            + max(get_safe_upper_limit(nstates),
                  get_safe_upper_limit(model_factory->unobserved_ptns.size()));
        size_t nmix = (model_factory->fused_mix_rate) ? (size_t)1 : (size_t)model->getNMixtures();
        size_t scale_block_total = nptn_safe * ncat * nmix;
        size_t lh_block_total = scale_block_total * nstates;
        size_t tip_alloc_size = get_safe_upper_limit(
            nstates * (aln->STATE_UNKNOWN + 1) * model->getNMixtures());
        size_t total_lh_entries = (size_t)max_lh_slots * lh_block_total + 4 + tip_alloc_size;
        size_t total_scale_entries = (size_t)max_lh_slots * scale_block_total;

        // O1 optimization: Use create (allocate-only, no upload) for central_partial_lh
        // and central_scale_num. The GPU computes ALL internal node partial likelihoods
        // and scale counts during traversal — uploading host values (zeros) is wasteful.
        // Only tip_partial_lh (tiny, at end of central_partial_lh) needs host→device
        // transfer because it contains pre-computed one-hot/ambiguity state vectors.
        //
        // Before O1: copyin uploaded 3.2 GB (DNA) / 15.8 GB (AA) — 99.99% wasted.
        // After O1:  create + selective update uploads only ~608 B (DNA) / ~3.7 KB (AA).
        size_t tip_offset = (size_t)max_lh_slots * lh_block_total + 4;

        #pragma acc enter data \
            create(local_central_plh[0:total_lh_entries], \
                   local_central_scl[0:total_scale_entries]) \
            copyin(local_ptn_freq[0:nptn], \
                   local_ptn_invar[0:nptn]) \
            create(local_pattern_lh[0:nptn], \
                   local_pattern_lh_cat[0:nptn_ncat])

        // Upload ONLY the tip_partial_lh section (few KB at end of central_partial_lh).
        // tip_partial_lh = central_partial_lh + max_lh_slots * block_size (see phylotree.cpp:1146)
        // It stores one-hot vectors and ambiguity state mappings used by TIP-TIP/TIP-INT kernels.
        #pragma acc update device(local_central_plh[tip_offset:tip_alloc_size])

        // Save sizes and pointers for freeOpenACCData()
        gpu_total_lh_entries = total_lh_entries;
        gpu_total_scale_entries = total_scale_entries;
        gpu_nptn = nptn;
        gpu_nptn_ncat = nptn_ncat;
        gpu_central_plh_ptr = local_central_plh;
        gpu_central_scl_ptr = local_central_scl;
        gpu_ptn_freq_ptr = local_ptn_freq;
        gpu_ptn_invar_ptr = local_ptn_invar;
        gpu_pattern_lh_ptr = local_pattern_lh;
        gpu_pattern_lh_cat_ptr = local_pattern_lh_cat;
        gpu_data_resident = true;

        // P2: Build tip_states_flat — alignment state for each leaf × pattern
        // Used by batched kernels to avoid per-node states_left/states_right copyin.
        // O7: Flatten alignment on CPU (pattern-major, 1M pointer chases vs 50M),
        // upload flat array to GPU, transpose to leaf-major via GPU kernel.
        // Original sequential (nid, p) loop took ~1,456ms (79-93% of eval time).
        tip_states_flat = new int[(size_t)leafNum * nptn];
        int root_id = root->id;
        bool local_rooted = rooted;  // for GPU: only treat root specially when tree is rooted
        int local_leafNum = (int)leafNum;
        size_t local_nptn = nptn;
        size_t local_orig_nptn = orig_nptn;

        // Step 1: Flatten alignment into contiguous pattern-major array on CPU.
        // Each pattern vector is accessed once (1M pointer chases, not 50M).
        size_t aln_flat_size = (size_t)orig_nptn * leafNum;
        int *aln_flat = new int[aln_flat_size];
        for (size_t p = 0; p < orig_nptn; p++) {
            const auto &pattern = aln->at(p);
            int *dst = &aln_flat[p * leafNum];
            for (int nid = 0; nid < local_leafNum; nid++)
                dst[nid] = pattern[nid];
        }

        // Step 2: Allocate tip_states_flat on GPU (create = no upload needed)
        int *local_tip_states = tip_states_flat;
        size_t tip_states_total = (size_t)leafNum * nptn;
        #pragma acc enter data create(local_tip_states[0:tip_states_total])

        // Step 3: Upload flat alignment, transpose on GPU, then free flat array
        #pragma acc enter data copyin(aln_flat[0:aln_flat_size])

        #pragma acc parallel loop collapse(2) gang vector \
            present(local_tip_states, aln_flat)
        for (int nid = 0; nid < local_leafNum; nid++) {
            for (size_t p = 0; p < local_orig_nptn; p++) {
                if (local_rooted && nid == root_id)
                    local_tip_states[(size_t)nid * local_nptn + p] = 0;
                else
                    local_tip_states[(size_t)nid * local_nptn + p] =
                        aln_flat[p * local_leafNum + nid];
            }
        }

        #pragma acc exit data delete(aln_flat[0:aln_flat_size])
        delete[] aln_flat;

        // Handle unobserved patterns (usually very few, e.g. <100)
        size_t num_unobs = nptn - orig_nptn;
        if (num_unobs > 0) {
            size_t unobs_flat_size = num_unobs * leafNum;
            int *unobs_flat = new int[unobs_flat_size];
            for (size_t p = 0; p < num_unobs; p++) {
                const auto &unobs = model_factory->unobserved_ptns[p];
                int *dst = &unobs_flat[p * leafNum];
                for (int nid = 0; nid < local_leafNum; nid++)
                    dst[nid] = unobs[nid];
            }

            #pragma acc enter data copyin(unobs_flat[0:unobs_flat_size])

            #pragma acc parallel loop collapse(2) gang vector \
                present(local_tip_states, unobs_flat)
            for (int nid = 0; nid < local_leafNum; nid++) {
                for (size_t p = 0; p < num_unobs; p++) {
                    size_t dst_p = local_orig_nptn + p;
                    if (local_rooted && nid == root_id)
                        local_tip_states[(size_t)nid * local_nptn + dst_p] = 0;
                    else
                        local_tip_states[(size_t)nid * local_nptn + dst_p] =
                            unobs_flat[p * local_leafNum + nid];
                }
            }

            #pragma acc exit data delete(unobs_flat[0:unobs_flat_size])
            delete[] unobs_flat;
        }

        // Zero out padding beyond nptn for root row (root gets state 0 everywhere)
        // Already handled: both kernels above set root row to 0 for all pattern positions.

        gpu_tip_states_ptr = local_tip_states;
        gpu_tip_states_size = tip_states_total;

        if (verbose_mode >= VB_MED) {
            size_t uploaded_bytes = tip_alloc_size * sizeof(double)
                + nptn * sizeof(double) * 2;  // ptn_freq + ptn_invar
            size_t allocated_bytes = total_lh_entries * sizeof(double)
                + total_scale_entries * sizeof(UBYTE)
                + tip_states_total * sizeof(int);  // tip_states built on GPU
            size_t aln_upload = aln_flat_size * sizeof(int);
            cout << "OpenACC: GPU persistent data — allocated "
                 << allocated_bytes / (1024*1024) << " MB (create), "
                 << "uploaded " << uploaded_bytes / (1024*1024) << " MB, "
                 << "tip_states built on GPU via transpose kernel "
                 << "(" << tip_states_total * sizeof(int) / (1024*1024) << " MB, "
                 << "from " << aln_upload / (1024*1024) << " MB flat upload)"
                 << endl;
        }
#ifdef USE_OPENACC_PROFILE
        if (profiling) {
            #pragma acc wait
            acc_profile.t_persistent_upload += getRealTime() - prof_t1;
            prof_t1 = getRealTime(); // restart for trans_mat segment 2
        }
#endif
    }

    double prob_const = 0.0;

    // A1: Extract raw pointers for the branch (__restrict__ for no-alias optimization)
    double * __restrict__ dad_partial_lh_base = dad_branch->partial_lh;
    UBYTE  * __restrict__ dad_scale_num_base  = dad_branch->scale_num;

    if (dad->isLeaf()) {
        // special treatment for TIP-INTERNAL NODE case
        double *partial_lh_node = new double[(aln->STATE_UNKNOWN+1)*block];
        if (isRootLeaf(dad)) {
            // Rooted tree: apply state frequencies at root (unrooted never enters here)
            for (c = 0; c < ncat; c++) {
                double *lh_node = partial_lh_node + c*nstates;
                model->getStateFrequency(lh_node);
                double prop = site_rate->getProp(c);
                for (i = 0; i < nstates; i++)
                    lh_node[i] *= prop;
            }
        } else {
            // precompute information from one tip — A2: explicit indexing
            double *local_tip_plh = tip_partial_lh;
            for (int state = 0; state <= aln->STATE_UNKNOWN; state++) {
                double *lh_tip = local_tip_plh + state*nstates;
                for (c = 0; c < ncat; c++) {
                    for (i = 0; i < nstates; i++) {
                        double val = 0.0;
                        for (x = 0; x < nstates; x++)
                            val += trans_mat[c*nstatesqr + i*nstates + x] * lh_tip[x];
                        partial_lh_node[state*block + c*nstates + i] = val;
                    }
                }
            }
        }
#ifdef USE_OPENACC_PROFILE
        if (profiling) acc_profile.t_trans_mat_setup += getRealTime() - prof_t1;
#endif

        // now do the real computation
        for (int packet_id = 0; packet_id < num_packets; packet_id++) {
            size_t ptn_lower = limits[packet_id];
            size_t ptn_upper = limits[packet_id+1];

            // P2: Level-based batched partial likelihood computation
            // Group independent nodes by level, launch single kernel per (level, type).
            // Check for multifurcating nodes — fall back to per-node approach if any exist.
            {
                bool has_multifurcating = false;
                for (auto &info_check : traversal_info) {
                    PhyloNode *node_check = (PhyloNode*)info_check.dad_branch->node;
                    if (!node_check->isLeaf() && node_check->degree() > 3) {
                        has_multifurcating = true;
                        break;
                    }
                }

                if (has_multifurcating) {
                    // Fallback: per-node sequential computation (handles multifurcating nodes)
                    for (vector<TraversalInfo>::iterator it = traversal_info.begin(); it != traversal_info.end(); it++)
                        computePartialLikelihood(*it, ptn_lower, ptn_upper, packet_id);
                } else {

#ifdef USE_OPENACC_PROFILE
                if (profiling) prof_t1 = getRealTime();
#endif
                int max_level = 0;
                vector<LevelBatch> level_batches = groupByLevelAndType(traversal_info, max_level);
#ifdef USE_OPENACC_PROFILE
                if (profiling) acc_profile.total_levels += max_level + 1;
#endif

                // Precompute values needed by all batched kernels
                int ptn_lower_int = (int)ptn_lower;
                int ptn_upper_int = (int)ptn_upper;
                int block_int_b = (int)block;
                int nstates_int_b = (int)nstates;
                int nstatesqr_int_b = (int)nstatesqr;
                size_t nptn_stride = nptn;
                size_t state_unknown_b = aln->STATE_UNKNOWN;
                double *local_tip_plh_b = tip_partial_lh;
                int *local_tip_states_b = tip_states_flat;
                size_t tip_unknown_size_b = (aln->STATE_UNKNOWN + 1) * nstates;

                for (int lev = 0; lev <= max_level; lev++) {
                    // TIP-TIP batch
                    if (!level_batches[lev].tip_tip.empty()) {
                        auto &batch = level_batches[lev].tip_tip;
                        int num_nodes = (int)batch.size();
                        size_t *offsets = new size_t[num_nodes * 6];
                        for (int bi = 0; bi < num_nodes; bi++) {
                            TraversalInfo &info_b = *batch[bi];
                            PhyloNode *node_b = (PhyloNode*)info_b.dad_branch->node;
                            PhyloNode *dad_b = info_b.dad;
                            PhyloNeighbor *left_b = NULL, *right_b = NULL;
                            FOR_NEIGHBOR_IT(node_b, dad_b, it3) {
                                if (!left_b) left_b = (PhyloNeighbor*)(*it3);
                                else right_b = (PhyloNeighbor*)(*it3);
                            }
                            // Swap logic: if right child is root leaf, swap so left is root
                            // (matches original TIP-TIP kernel behavior; only for rooted trees)
                            bool swapped_tt = false;
                            if (isRootLeaf(right_b->node)) {
                                PhyloNeighbor *tmp = left_b; left_b = right_b; right_b = tmp;
                                swapped_tt = true;
                            }
                            double *tip_lh_left_b = info_b.partial_lh_leaves;
                            double *tip_lh_right_b = info_b.partial_lh_leaves + (aln->STATE_UNKNOWN + 1) * block;
                            if (swapped_tt) {
                                double *tmp_tlh = tip_lh_left_b;
                                tip_lh_left_b = tip_lh_right_b;
                                tip_lh_right_b = tmp_tlh;
                            }
                            offsets[bi*6 + 0] = (size_t)(info_b.dad_branch->partial_lh - central_partial_lh);
                            offsets[bi*6 + 1] = (size_t)(info_b.dad_branch->scale_num - central_scale_num);
                            offsets[bi*6 + 2] = (size_t)(tip_lh_left_b - buffer_partial_lh);
                            offsets[bi*6 + 3] = (size_t)(tip_lh_right_b - buffer_partial_lh);
                            offsets[bi*6 + 4] = (size_t)left_b->node->id;
                            offsets[bi*6 + 5] = (size_t)right_b->node->id;
                        }
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            acc_profile.t_offset_build += getRealTime() - prof_t1;
                            #pragma acc wait
                            prof_t1 = getRealTime();
                        }
#endif
                        batchedTipTip(offsets, num_nodes,
                            local_central_plh, local_central_scl,
                            local_buffer_plh, local_tip_states_b,
                            gpu_total_lh_entries, gpu_total_scale_entries,
                            gpu_buffer_plh_size, gpu_tip_states_size,
                            ptn_lower_int, ptn_upper_int, block_int_b, nptn_stride);
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            #pragma acc wait
                            acc_profile.t_kernel_tip_tip += getRealTime() - prof_t1;
                            acc_profile.n_kernel_tip_tip++;
                            acc_profile.total_nodes_tip_tip += num_nodes;
                            prof_t1 = getRealTime();
                        }
#endif
                        delete[] offsets;
                    }

                    // TIP-INTERNAL batch
                    if (!level_batches[lev].tip_internal.empty()) {
                        auto &batch = level_batches[lev].tip_internal;
                        int num_nodes = (int)batch.size();
                        size_t *offsets = new size_t[num_nodes * 8];
                        for (int bi = 0; bi < num_nodes; bi++) {
                            TraversalInfo &info_b = *batch[bi];
                            PhyloNode *node_b = (PhyloNode*)info_b.dad_branch->node;
                            PhyloNode *dad_b = info_b.dad;
                            PhyloNeighbor *left_b = NULL, *right_b = NULL;
                            FOR_NEIGHBOR_IT(node_b, dad_b, it3) {
                                if (!left_b) left_b = (PhyloNeighbor*)(*it3);
                                else right_b = (PhyloNeighbor*)(*it3);
                            }
                            // Ensure left is the leaf, right is internal
                            // Must swap both neighbors AND P(t) matrices (matches original kernel)
                            double *eleft_b = info_b.echildren;
                            double *eright_b = info_b.echildren + block * nstates;
                            if (!left_b->node->isLeaf() && right_b->node->isLeaf()) {
                                PhyloNeighbor *tmp = left_b; left_b = right_b; right_b = tmp;
                                double *etmp = eleft_b; eleft_b = eright_b; eright_b = etmp;
                            }
                            double *tip_lh_left_b = info_b.partial_lh_leaves;

                            offsets[bi*8 + 0] = (size_t)(info_b.dad_branch->partial_lh - central_partial_lh);
                            offsets[bi*8 + 1] = (size_t)(info_b.dad_branch->scale_num - central_scale_num);
                            offsets[bi*8 + 2] = (size_t)(right_b->partial_lh - central_partial_lh);
                            offsets[bi*8 + 3] = (size_t)(right_b->scale_num - central_scale_num);
                            offsets[bi*8 + 4] = (size_t)(eright_b - buffer_partial_lh);
                            offsets[bi*8 + 5] = (size_t)(tip_lh_left_b - buffer_partial_lh);
                            offsets[bi*8 + 6] = (size_t)left_b->node->id;
                            offsets[bi*8 + 7] = 0; // padding
                        }
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            acc_profile.t_offset_build += getRealTime() - prof_t1;
                            #pragma acc wait
                            prof_t1 = getRealTime();
                        }
#endif
                        batchedTipInternal(offsets, num_nodes,
                            local_central_plh, local_central_scl,
                            local_buffer_plh, local_tip_states_b,
                            local_tip_plh_b,
                            gpu_total_lh_entries, gpu_total_scale_entries,
                            gpu_buffer_plh_size, gpu_tip_states_size,
                            tip_unknown_size_b,
                            ptn_lower_int, ptn_upper_int,
                            block_int_b, nstates_int_b, nstatesqr_int_b,
                            nptn_stride, state_unknown_b);
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            #pragma acc wait
                            acc_profile.t_kernel_tip_int += getRealTime() - prof_t1;
                            acc_profile.n_kernel_tip_int++;
                            acc_profile.total_nodes_tip_int += num_nodes;
                            prof_t1 = getRealTime();
                        }
#endif
                        delete[] offsets;
                    }

                    // INTERNAL-INTERNAL batch
                    if (!level_batches[lev].internal_internal.empty()) {
                        auto &batch = level_batches[lev].internal_internal;
                        int num_nodes = (int)batch.size();
                        size_t *offsets = new size_t[num_nodes * 8];
                        for (int bi = 0; bi < num_nodes; bi++) {
                            TraversalInfo &info_b = *batch[bi];
                            PhyloNode *node_b = (PhyloNode*)info_b.dad_branch->node;
                            PhyloNode *dad_b = info_b.dad;
                            PhyloNeighbor *left_b = NULL, *right_b = NULL;
                            FOR_NEIGHBOR_IT(node_b, dad_b, it3) {
                                if (!left_b) left_b = (PhyloNeighbor*)(*it3);
                                else right_b = (PhyloNeighbor*)(*it3);
                            }
                            double *eleft_b = info_b.echildren;
                            double *eright_b = info_b.echildren + block * nstates;

                            offsets[bi*8 + 0] = (size_t)(info_b.dad_branch->partial_lh - central_partial_lh);
                            offsets[bi*8 + 1] = (size_t)(info_b.dad_branch->scale_num - central_scale_num);
                            offsets[bi*8 + 2] = (size_t)(left_b->partial_lh - central_partial_lh);
                            offsets[bi*8 + 3] = (size_t)(right_b->partial_lh - central_partial_lh);
                            offsets[bi*8 + 4] = (size_t)(left_b->scale_num - central_scale_num);
                            offsets[bi*8 + 5] = (size_t)(right_b->scale_num - central_scale_num);
                            offsets[bi*8 + 6] = (size_t)(eleft_b - buffer_partial_lh);
                            offsets[bi*8 + 7] = (size_t)(eright_b - buffer_partial_lh);
                        }
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            acc_profile.t_offset_build += getRealTime() - prof_t1;
                            #pragma acc wait
                            prof_t1 = getRealTime();
                        }
#endif
                        batchedInternalInternal(offsets, num_nodes,
                            local_central_plh, local_central_scl,
                            local_buffer_plh,
                            local_tip_plh_b,
                            gpu_total_lh_entries, gpu_total_scale_entries,
                            gpu_buffer_plh_size,
                            tip_unknown_size_b,
                            ptn_lower_int, ptn_upper_int,
                            block_int_b, nstates_int_b, nstatesqr_int_b,
                            state_unknown_b);
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            #pragma acc wait
                            acc_profile.t_kernel_int_int += getRealTime() - prof_t1;
                            acc_profile.n_kernel_int_int++;
                            acc_profile.total_nodes_int_int += num_nodes;
                            prof_t1 = getRealTime();
                        }
#endif
                        delete[] offsets;
                    }
                } // for lev
#ifdef USE_OPENACC_PROFILE
                // Profiling: end offset_build for remaining CPU work after last kernel
                if (profiling) acc_profile.t_offset_build += getRealTime() - prof_t1;
#endif

                } // end else (bifurcating batched path)
            } // end P2 batched computation

            // A3: Precompute per-pattern dad states (CPU side, before GPU region)
            size_t nptn_range = ptn_upper - ptn_lower;
            int *states_dad = new int[nptn_range];
            bool dad_is_root = isRootLeaf(dad);
            int dad_node_id = dad->id;
            for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
                size_t idx = ptn - ptn_lower;
                if (dad_is_root)
                    states_dad[idx] = 0;
                else
                    states_dad[idx] = (ptn < orig_nptn) ? (aln->at(ptn))[dad_node_id] : model_factory->unobserved_ptns[ptn-orig_nptn][dad_node_id];
            }

            // Step 11: Log-likelihood reduction offloaded to GPU via OpenACC
            // TIP-INTERNAL case: dad is a leaf, node is internal.
            // partial_lh_node[state*block+s] is precomputed (trans_mat × tip_lh).
            // Reduction: gang over patterns, reduction(+:) for total lnL.
#ifdef USE_OPENACC_PROFILE
            if (profiling) prof_t1 = getRealTime();
#endif
            {
                size_t plh_node_size = (aln->STATE_UNKNOWN + 1) * block;
                size_t plh_offset = ptn_lower * block;
                size_t plh_count  = (ptn_upper - ptn_lower) * block;
                size_t scl_offset = ptn_lower;
                size_t scl_count  = ptn_upper - ptn_lower;
                // Step 12: class member pointers captured in outer scope for
                // enter data; persistent data uses present() instead of copyin/copyout.

                #pragma acc data \
                    copyin(states_dad[0:nptn_range], \
                           partial_lh_node[0:plh_node_size]) \
                    present(dad_partial_lh_base[plh_offset:plh_count], \
                            dad_scale_num_base[scl_offset:scl_count], \
                            local_ptn_invar[ptn_lower:scl_count], \
                            local_ptn_freq[ptn_lower:scl_count], \
                            local_pattern_lh[ptn_lower:scl_count], \
                            local_pattern_lh_cat[ptn_lower:scl_count])
                {
                    int block_int = (int)block;
                    int nstates_int = (int)nstates;
                    int ncat_int = (int)ncat;
                    int orig_nptn_int = (int)orig_nptn;
                    #pragma acc parallel loop gang reduction(+:tree_lh, prob_const)
                    for (int p = (int)ptn_lower; p < (int)ptn_upper; p++) {
                        int idx = p - (int)ptn_lower;
                        int state_dad = states_dad[idx];
                        double lh_ptn = local_ptn_invar[p];

                        // Dot product: partial_lh_node[state] · dad_partial_lh[p]
                        // For ncat=1: single category, block == nstates
                        for (int cc = 0; cc < ncat_int; cc++) {
                            double lh_cat = 0.0;
                            for (int ii = 0; ii < nstates_int; ii++) {
                                lh_cat += partial_lh_node[state_dad*block_int + cc*nstates_int + ii]
                                        * dad_partial_lh_base[p*block_int + cc*nstates_int + ii];
                            }
                            local_pattern_lh_cat[p*ncat_int + cc] = lh_cat;
                            lh_ptn += lh_cat;
                        }

                        // Log-likelihood + scaling correction
                        if (p < orig_nptn_int) {
                            lh_ptn = log(fabs(lh_ptn)) + dad_scale_num_base[p] * LOG_SCALING_THRESHOLD;
                            local_pattern_lh[p] = lh_ptn;
                            tree_lh += lh_ptn * local_ptn_freq[p];
                        } else {
                            if (dad_scale_num_base[p] >= 1)
                                lh_ptn *= SCALING_THRESHOLD;
                            prob_const += lh_ptn;
                        }
                    } // FOR p
                } // end acc data
            }
#ifdef USE_OPENACC_PROFILE
            if (profiling) {
                #pragma acc wait
                acc_profile.t_reduction += getRealTime() - prof_t1;
                acc_profile.n_reduction++;
            }
#endif

            delete [] states_dad;
        } // FOR packet_id
        delete [] partial_lh_node;
    } else {
#ifdef USE_OPENACC_PROFILE
        // Profiling: end trans_mat segment 2 (Path B: no partial_lh_node precomp)
        if (profiling) acc_profile.t_trans_mat_setup += getRealTime() - prof_t1;
#endif

        // both dad and node are internal nodes
        // A1: Extract raw pointers (__restrict__ for no-alias optimization)
        double * __restrict__ node_partial_lh_base = node_branch->partial_lh;
        UBYTE  * __restrict__ node_scale_num_base  = node_branch->scale_num;

        for (int packet_id = 0; packet_id < num_packets; packet_id++) {
            size_t ptn_lower = limits[packet_id];
            size_t ptn_upper = limits[packet_id+1];

            // P2: Level-based batched partial likelihood computation
            {
                bool has_multifurcating = false;
                for (auto &info_check : traversal_info) {
                    PhyloNode *node_check = (PhyloNode*)info_check.dad_branch->node;
                    if (!node_check->isLeaf() && node_check->degree() > 3) {
                        has_multifurcating = true;
                        break;
                    }
                }

                if (has_multifurcating) {
                    for (vector<TraversalInfo>::iterator it = traversal_info.begin(); it != traversal_info.end(); it++)
                        computePartialLikelihood(*it, ptn_lower, ptn_upper, packet_id);
                } else {

#ifdef USE_OPENACC_PROFILE
                if (profiling) prof_t1 = getRealTime();
#endif
                int max_level = 0;
                vector<LevelBatch> level_batches = groupByLevelAndType(traversal_info, max_level);
#ifdef USE_OPENACC_PROFILE
                if (profiling) acc_profile.total_levels += max_level + 1;
#endif

                int ptn_lower_int = (int)ptn_lower;
                int ptn_upper_int = (int)ptn_upper;
                int block_int_b = (int)block;
                int nstates_int_b = (int)nstates;
                int nstatesqr_int_b = (int)nstatesqr;
                size_t nptn_stride = nptn;
                size_t state_unknown_b = aln->STATE_UNKNOWN;
                double *local_tip_plh_b = tip_partial_lh;
                int *local_tip_states_b = tip_states_flat;
                size_t tip_unknown_size_b = (aln->STATE_UNKNOWN + 1) * nstates;

                for (int lev = 0; lev <= max_level; lev++) {
                    if (!level_batches[lev].tip_tip.empty()) {
                        auto &batch = level_batches[lev].tip_tip;
                        int num_nodes = (int)batch.size();
                        size_t *offsets = new size_t[num_nodes * 6];
                        for (int bi = 0; bi < num_nodes; bi++) {
                            TraversalInfo &info_b = *batch[bi];
                            PhyloNode *node_b = (PhyloNode*)info_b.dad_branch->node;
                            PhyloNode *dad_b = info_b.dad;
                            PhyloNeighbor *left_b = NULL, *right_b = NULL;
                            FOR_NEIGHBOR_IT(node_b, dad_b, it3) {
                                if (!left_b) left_b = (PhyloNeighbor*)(*it3);
                                else right_b = (PhyloNeighbor*)(*it3);
                            }
                            bool swapped_tt = false;
                            if (isRootLeaf(right_b->node)) {
                                PhyloNeighbor *tmp = left_b; left_b = right_b; right_b = tmp;
                                swapped_tt = true;
                            }
                            double *tip_lh_left_b = info_b.partial_lh_leaves;
                            double *tip_lh_right_b = info_b.partial_lh_leaves + (aln->STATE_UNKNOWN + 1) * block;
                            if (swapped_tt) {
                                double *tmp_tlh = tip_lh_left_b;
                                tip_lh_left_b = tip_lh_right_b;
                                tip_lh_right_b = tmp_tlh;
                            }
                            offsets[bi*6 + 0] = (size_t)(info_b.dad_branch->partial_lh - central_partial_lh);
                            offsets[bi*6 + 1] = (size_t)(info_b.dad_branch->scale_num - central_scale_num);
                            offsets[bi*6 + 2] = (size_t)(tip_lh_left_b - buffer_partial_lh);
                            offsets[bi*6 + 3] = (size_t)(tip_lh_right_b - buffer_partial_lh);
                            offsets[bi*6 + 4] = (size_t)left_b->node->id;
                            offsets[bi*6 + 5] = (size_t)right_b->node->id;
                        }
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            acc_profile.t_offset_build += getRealTime() - prof_t1;
                            #pragma acc wait
                            prof_t1 = getRealTime();
                        }
#endif
                        batchedTipTip(offsets, num_nodes,
                            local_central_plh, local_central_scl,
                            local_buffer_plh, local_tip_states_b,
                            gpu_total_lh_entries, gpu_total_scale_entries,
                            gpu_buffer_plh_size, gpu_tip_states_size,
                            ptn_lower_int, ptn_upper_int, block_int_b, nptn_stride);
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            #pragma acc wait
                            acc_profile.t_kernel_tip_tip += getRealTime() - prof_t1;
                            acc_profile.n_kernel_tip_tip++;
                            acc_profile.total_nodes_tip_tip += num_nodes;
                            prof_t1 = getRealTime();
                        }
#endif
                        delete[] offsets;
                    }
                    if (!level_batches[lev].tip_internal.empty()) {
                        auto &batch = level_batches[lev].tip_internal;
                        int num_nodes = (int)batch.size();
                        size_t *offsets = new size_t[num_nodes * 8];
                        for (int bi = 0; bi < num_nodes; bi++) {
                            TraversalInfo &info_b = *batch[bi];
                            PhyloNode *node_b = (PhyloNode*)info_b.dad_branch->node;
                            PhyloNode *dad_b = info_b.dad;
                            PhyloNeighbor *left_b = NULL, *right_b = NULL;
                            FOR_NEIGHBOR_IT(node_b, dad_b, it3) {
                                if (!left_b) left_b = (PhyloNeighbor*)(*it3);
                                else right_b = (PhyloNeighbor*)(*it3);
                            }
                            double *eleft_b = info_b.echildren;
                            double *eright_b = info_b.echildren + block * nstates;
                            if (!left_b->node->isLeaf() && right_b->node->isLeaf()) {
                                PhyloNeighbor *tmp = left_b; left_b = right_b; right_b = tmp;
                                double *etmp = eleft_b; eleft_b = eright_b; eright_b = etmp;
                            }
                            double *tip_lh_left_b = info_b.partial_lh_leaves;
                            offsets[bi*8 + 0] = (size_t)(info_b.dad_branch->partial_lh - central_partial_lh);
                            offsets[bi*8 + 1] = (size_t)(info_b.dad_branch->scale_num - central_scale_num);
                            offsets[bi*8 + 2] = (size_t)(right_b->partial_lh - central_partial_lh);
                            offsets[bi*8 + 3] = (size_t)(right_b->scale_num - central_scale_num);
                            offsets[bi*8 + 4] = (size_t)(eright_b - buffer_partial_lh);
                            offsets[bi*8 + 5] = (size_t)(tip_lh_left_b - buffer_partial_lh);
                            offsets[bi*8 + 6] = (size_t)left_b->node->id;
                            offsets[bi*8 + 7] = 0;
                        }
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            acc_profile.t_offset_build += getRealTime() - prof_t1;
                            #pragma acc wait
                            prof_t1 = getRealTime();
                        }
#endif
                        batchedTipInternal(offsets, num_nodes,
                            local_central_plh, local_central_scl,
                            local_buffer_plh, local_tip_states_b,
                            local_tip_plh_b,
                            gpu_total_lh_entries, gpu_total_scale_entries,
                            gpu_buffer_plh_size, gpu_tip_states_size,
                            tip_unknown_size_b,
                            ptn_lower_int, ptn_upper_int,
                            block_int_b, nstates_int_b, nstatesqr_int_b,
                            nptn_stride, state_unknown_b);
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            #pragma acc wait
                            acc_profile.t_kernel_tip_int += getRealTime() - prof_t1;
                            acc_profile.n_kernel_tip_int++;
                            acc_profile.total_nodes_tip_int += num_nodes;
                            prof_t1 = getRealTime();
                        }
#endif
                        delete[] offsets;
                    }
                    if (!level_batches[lev].internal_internal.empty()) {
                        auto &batch = level_batches[lev].internal_internal;
                        int num_nodes = (int)batch.size();
                        size_t *offsets = new size_t[num_nodes * 8];
                        for (int bi = 0; bi < num_nodes; bi++) {
                            TraversalInfo &info_b = *batch[bi];
                            PhyloNode *node_b = (PhyloNode*)info_b.dad_branch->node;
                            PhyloNode *dad_b = info_b.dad;
                            PhyloNeighbor *left_b = NULL, *right_b = NULL;
                            FOR_NEIGHBOR_IT(node_b, dad_b, it3) {
                                if (!left_b) left_b = (PhyloNeighbor*)(*it3);
                                else right_b = (PhyloNeighbor*)(*it3);
                            }
                            double *eleft_b = info_b.echildren;
                            double *eright_b = info_b.echildren + block * nstates;
                            offsets[bi*8 + 0] = (size_t)(info_b.dad_branch->partial_lh - central_partial_lh);
                            offsets[bi*8 + 1] = (size_t)(info_b.dad_branch->scale_num - central_scale_num);
                            offsets[bi*8 + 2] = (size_t)(left_b->partial_lh - central_partial_lh);
                            offsets[bi*8 + 3] = (size_t)(right_b->partial_lh - central_partial_lh);
                            offsets[bi*8 + 4] = (size_t)(left_b->scale_num - central_scale_num);
                            offsets[bi*8 + 5] = (size_t)(right_b->scale_num - central_scale_num);
                            offsets[bi*8 + 6] = (size_t)(eleft_b - buffer_partial_lh);
                            offsets[bi*8 + 7] = (size_t)(eright_b - buffer_partial_lh);
                        }
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            acc_profile.t_offset_build += getRealTime() - prof_t1;
                            #pragma acc wait
                            prof_t1 = getRealTime();
                        }
#endif
                        batchedInternalInternal(offsets, num_nodes,
                            local_central_plh, local_central_scl,
                            local_buffer_plh,
                            local_tip_plh_b,
                            gpu_total_lh_entries, gpu_total_scale_entries,
                            gpu_buffer_plh_size,
                            tip_unknown_size_b,
                            ptn_lower_int, ptn_upper_int,
                            block_int_b, nstates_int_b, nstatesqr_int_b,
                            state_unknown_b);
#ifdef USE_OPENACC_PROFILE
                        if (profiling) {
                            #pragma acc wait
                            acc_profile.t_kernel_int_int += getRealTime() - prof_t1;
                            acc_profile.n_kernel_int_int++;
                            acc_profile.total_nodes_int_int += num_nodes;
                            prof_t1 = getRealTime();
                        }
#endif
                        delete[] offsets;
                    }
                } // for lev
#ifdef USE_OPENACC_PROFILE
                // Profiling: end offset_build for remaining CPU work after last kernel
                if (profiling) acc_profile.t_offset_build += getRealTime() - prof_t1;
#endif

                } // end else (bifurcating batched path)
            } // end P2 batched computation

            // Step 11: Log-likelihood reduction offloaded to GPU via OpenACC
            // INTERNAL-INTERNAL case: both dad and node are internal nodes.
            // For each pattern: compute trans_mat × node_partial_lh, dot with
            // dad_partial_lh, take log + scaling correction, accumulate weighted sum.
            // Matches PoC: gang over patterns, reduction(+:logLikelihood).
#ifdef USE_OPENACC_PROFILE
            if (profiling) prof_t1 = getRealTime();
#endif
            {
                size_t trans_mat_size = block * nstates;
                size_t plh_offset = ptn_lower * block;
                size_t plh_count  = (ptn_upper - ptn_lower) * block;
                size_t scl_offset = ptn_lower;
                size_t scl_count  = ptn_upper - ptn_lower;
                // Step 12: class member pointers captured in outer scope for
                // enter data; persistent data uses present() instead of copyin/copyout.

                #pragma acc data \
                    copyin(trans_mat[0:trans_mat_size]) \
                    present(dad_partial_lh_base[plh_offset:plh_count], \
                            node_partial_lh_base[plh_offset:plh_count], \
                            dad_scale_num_base[scl_offset:scl_count], \
                            node_scale_num_base[scl_offset:scl_count], \
                            local_ptn_invar[ptn_lower:scl_count], \
                            local_ptn_freq[ptn_lower:scl_count], \
                            local_pattern_lh[ptn_lower:scl_count], \
                            local_pattern_lh_cat[ptn_lower:scl_count])
                {
                    int block_int = (int)block;
                    int nstates_int = (int)nstates;
                    int nstatesqr_int = (int)nstatesqr;
                    int ncat_int = (int)ncat;
                    int orig_nptn_int = (int)orig_nptn;
                    #pragma acc parallel loop gang reduction(+:tree_lh, prob_const)
                    for (int p = (int)ptn_lower; p < (int)ptn_upper; p++) {
                        double lh_ptn = local_ptn_invar[p];

                        for (int cc = 0; cc < ncat_int; cc++) {
                            double lh_cat = 0.0;
                            for (int ii = 0; ii < nstates_int; ii++) {
                                // Matrix-vector product: trans_mat × node_partial_lh
                                double lh_state = 0.0;
                                for (int xx = 0; xx < nstates_int; xx++)
                                    lh_state += trans_mat[cc*nstatesqr_int + ii*nstates_int + xx]
                                              * node_partial_lh_base[p*block_int + cc*nstates_int + xx];
                                lh_cat += dad_partial_lh_base[p*block_int + cc*nstates_int + ii] * lh_state;
                            }
                            local_pattern_lh_cat[p*ncat_int + cc] = lh_cat;
                            lh_ptn += lh_cat;
                        }

                        // Log-likelihood + scaling correction
                        if (p < orig_nptn_int) {
                            lh_ptn = log(fabs(lh_ptn)) + (dad_scale_num_base[p] + node_scale_num_base[p]) * LOG_SCALING_THRESHOLD;
                            local_pattern_lh[p] = lh_ptn;
                            tree_lh += lh_ptn * local_ptn_freq[p];
                        } else {
                            if (dad_scale_num_base[p] + node_scale_num_base[p] >= 1)
                                lh_ptn *= SCALING_THRESHOLD;
                            prob_const += lh_ptn;
                        }
                    } // FOR p
                } // end acc data
            }
#ifdef USE_OPENACC_PROFILE
            if (profiling) {
                #pragma acc wait
                acc_profile.t_reduction += getRealTime() - prof_t1;
                acc_profile.n_reduction++;
            }
#endif
        } // FOR packet_id
    }

    // ====== Copy _pattern_lh back to host ======
    // Only _pattern_lh is needed on the host (for prob_const correction and
    // IQ-TREE callers). All other GPU data stays resident — no download of
    // central_partial_lh or central_scale_num needed since they remain
    // present() for subsequent calls.
#ifdef USE_OPENACC_PROFILE
    if (profiling) prof_t1 = getRealTime();
#endif
    #pragma acc update self(local_pattern_lh[0:nptn])
#ifdef USE_OPENACC_PROFILE
    if (profiling) {
        #pragma acc wait
        acc_profile.t_d2h_pattern_lh += getRealTime() - prof_t1;
    }

    if (profiling) prof_t1 = getRealTime();
#endif
    if (std::isnan(tree_lh) || std::isinf(tree_lh)) {
        cout << "WARNING: Numerical underflow caused by alignment sites";
        i = aln->getNSite();
        int j;
        for (j = 0, c = 0; j < i; j++) {
            ptn = aln->getPatternID(j);
            if (std::isnan(_pattern_lh[ptn]) || std::isinf(_pattern_lh[ptn])) {
                cout << " " << j+1;
                c++;
                if (c >= 10) {
                    cout << " ...";
                    break;
                }
            }
        }
        cout << endl;
        tree_lh = 0.0;
        for (ptn = 0; ptn < orig_nptn; ptn++) {
            if (std::isnan(_pattern_lh[ptn]) || std::isinf(_pattern_lh[ptn])) {
                _pattern_lh[ptn] = LOG_SCALING_THRESHOLD*4; // log(2^(-1024))
            }
            tree_lh += _pattern_lh[ptn] * ptn_freq[ptn];
        }
    }

    if (orig_nptn < nptn) {
        // ascertainment bias correction
        if (prob_const >= 1.0 || prob_const < 0.0) {
            printTree(cout, WT_TAXON_ID + WT_BR_LEN + WT_NEWLINE);
            model->writeInfo(cout);
        }
        ASSERT(prob_const < 1.0 && prob_const >= 0.0);

        prob_const = log(1.0 - prob_const);
        for (ptn = 0; ptn < orig_nptn; ptn++)
            _pattern_lh[ptn] -= prob_const;
        tree_lh -= aln->getNSite()*prob_const;
        ASSERT(!std::isnan(tree_lh) && !std::isinf(tree_lh));
    }

    ASSERT(!std::isnan(tree_lh) && !std::isinf(tree_lh));
#ifdef USE_OPENACC_PROFILE
    if (profiling) acc_profile.t_host_postproc += getRealTime() - prof_t1;
#endif

    // D3: Keep buffer_partial_lh on GPU for derivative kernel reuse.
    // During branch length optimization, the derivative kernel needs this buffer
    // for stale partial recomputation. Keeping it resident avoids redundant
    // re-uploads (~7-95 MB per derivative call).
    gpu_buffer_plh_resident = true;
    gpu_buffer_plh_ptr = local_buffer_plh;
#ifdef USE_OPENACC_PROFILE
    if (profiling) {
        acc_profile.t_buffer_delete += 0.0;  // no delete, kept resident
    }
#endif

#ifdef USE_OPENACC_PROFILE
    // Profiling: accumulate total time and print summary at configured interval
    if (profiling) {
        acc_profile.t_total += getRealTime() - prof_t0;
        const char *interval_env = getenv("IQTREE_OPENACC_PROFILE_INTERVAL");
        int interval = interval_env ? atoi(interval_env) : 10;
        if (interval <= 0) interval = 10;
        if (acc_profile.call_count % interval == 0) {
            acc_profile.print_summary();
        }
    }
#endif

    delete [] trans_mat;
    return tree_lh;
}

// ==========================================================================
// OpenACC Derivative Kernel for Branch Length Optimization
//
// Computes first and second derivatives of log-likelihood w.r.t. branch
// length t, using partial likelihoods already resident on GPU.
//
// Math (per pattern p, summed over rate categories c and states i):
//   ℓ_p   = Σ_c Σ_i L^dad[i,p] · Σ_x P̃_c[i,x]   · L^node[x,p]
//   dℓ_p  = Σ_c Σ_i L^dad[i,p] · Σ_x P̃'_c[i,x]  · L^node[x,p]
//   d²ℓ_p = Σ_c Σ_i L^dad[i,p] · Σ_x P̃''_c[i,x] · L^node[x,p]
//
// Normalization (quotient rule on log ℓ):
//   g_p  = dℓ_p / ℓ_p
//   h_p  = d²ℓ_p / ℓ_p − g_p²
//   df  += f_p · g_p,    ddf += f_p · h_p
// ==========================================================================

void PhyloTree::computeLikelihoodDervGenericOpenACC(
    PhyloNeighbor *dad_branch, PhyloNode *dad, double *df, double *ddf)
{
    PhyloNode *node = (PhyloNode*) dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*) node->findNeighbor(dad);
    if (!central_partial_lh)
        initializeAllPartialLh();
    if (node->isLeaf() || (dad_branch->direction == AWAYFROM_ROOT && !isRootLeaf(dad))) {
        PhyloNode *tmp_node = dad;
        dad = node;
        node = tmp_node;
        PhyloNeighbor *tmp_nei = dad_branch;
        dad_branch = node_branch;
        node_branch = tmp_nei;
    }

    // Force state-space kernel (same as likelihood kernel)
    Params::getInstance().kernel_nonrev = true;

#ifdef USE_OPENACC_PROFILE
    bool deriv_profiling = acc_profile.enabled;
    double deriv_prof_t0 = 0.0, deriv_prof_t1 = 0.0;
    if (deriv_profiling) {
        deriv_prof_t0 = getRealTime();
        deriv_prof_t1 = deriv_prof_t0;
        acc_profile.deriv_call_count++;
    }
#endif

    computeTraversalInfo<Vec1d>(node, dad, false);

    // Recompute any stale partial likelihoods on GPU.
    // During optimizeAllBranches, changing one branch length invalidates
    // partials at adjacent nodes (clearReversePartialLh). computeTraversalInfo
    // populates traversal_info with the stale entries. The CPU nonrev kernel
    // iterates over these and calls computePartialLikelihood for each; we
    // must do the same, otherwise the derivative uses stale GPU data.
    //
    // D3: buffer_partial_lh may already be GPU-resident from the likelihood
    // kernel. If so, sync new content with update device() (avoids alloc/dealloc
    // overhead) instead of full copyin. If not resident, do full copyin.
    // When traversal_info is empty, no upload needed at all — partials are fresh.
    if (!traversal_info.empty()) {
        size_t buf_plh_size = getBufferPartialLhSize();
        double *local_buffer_plh = buffer_partial_lh;
        if (gpu_buffer_plh_resident) {
            // Buffer allocation exists on GPU — just sync the new host content
            #pragma acc update device(local_buffer_plh[0:buf_plh_size])
        } else {
            // First time — allocate and upload
            #pragma acc enter data copyin(local_buffer_plh[0:buf_plh_size])
            gpu_buffer_plh_resident = true;
            gpu_buffer_plh_ptr = local_buffer_plh;
            gpu_buffer_plh_size = buf_plh_size;
        }
        for (auto it = traversal_info.begin(); it != traversal_info.end(); it++)
            computePartialLikelihood(*it, 0, aln->size() + model_factory->unobserved_ptns.size(), 0);
        // D3: Keep buffer on GPU — don't delete here
    }

#ifdef USE_OPENACC_PROFILE
    if (deriv_profiling) {
        #pragma acc wait
        acc_profile.t_deriv_traversal += getRealTime() - deriv_prof_t1;
        if (!traversal_info.empty()) acc_profile.n_deriv_stale_recomp++;
        deriv_prof_t1 = getRealTime();
    }
#endif

    size_t nstates = aln->num_states;
    size_t nstatesqr = nstates * nstates;
    size_t ncat = site_rate->getNRate();
    size_t block = ncat * nstates;
    size_t orig_nptn = aln->size();
    size_t nptn = aln->size() + model_factory->unobserved_ptns.size();

    // D4: Persistent transition matrices — allocate once, reuse across calls.
    // Only the content changes (via update device), not the allocation.
    size_t mat_size = block * nstates;
    if (!gpu_trans_mat || gpu_trans_mat_size < mat_size) {
        // First call or size changed (e.g., model switch) — (re)allocate
        if (gpu_trans_mat_resident) {
            #pragma acc exit data delete(gpu_trans_mat[0:gpu_trans_mat_size], \
                                         gpu_trans_derv1[0:gpu_trans_mat_size], \
                                         gpu_trans_derv2[0:gpu_trans_mat_size])
            gpu_trans_mat_resident = false;
        }
        delete[] gpu_trans_mat;
        delete[] gpu_trans_derv1;
        delete[] gpu_trans_derv2;
        gpu_trans_mat   = new double[mat_size];
        gpu_trans_derv1 = new double[mat_size];
        gpu_trans_derv2 = new double[mat_size];
        gpu_trans_mat_size = mat_size;
    }
    double *trans_mat   = gpu_trans_mat;
    double *trans_derv1 = gpu_trans_derv1;
    double *trans_derv2 = gpu_trans_derv2;

    for (size_t c = 0; c < ncat; c++) {
        double cat_rate = site_rate->getRate(c);
        double len = cat_rate * dad_branch->length;
        double prop = site_rate->getProp(c);
        double *this_trans_mat   = &trans_mat[c * nstatesqr];
        double *this_trans_derv1 = &trans_derv1[c * nstatesqr];
        double *this_trans_derv2 = &trans_derv2[c * nstatesqr];

        model->computeTransDerv(len, this_trans_mat, this_trans_derv1, this_trans_derv2);

        double prop_rate   = prop * cat_rate;       // w_c · r_c
        double prop_rate_2 = prop_rate * cat_rate;   // w_c · r_c²
        for (size_t i = 0; i < nstatesqr; i++) {
            this_trans_mat[i]   *= prop;
            this_trans_derv1[i] *= prop_rate;
            this_trans_derv2[i] *= prop_rate_2;
        }
    }

    if (!rooted) {
        // Pre-multiply state frequencies into all 3 matrices (unrooted virtual root)
        double state_freq[64];
        model->getStateFrequency(state_freq);
        for (size_t c = 0; c < ncat; c++) {
            double *tm  = &trans_mat[c * nstatesqr];
            double *td1 = &trans_derv1[c * nstatesqr];
            double *td2 = &trans_derv2[c * nstatesqr];
            for (size_t i = 0; i < nstates; i++) {
                for (size_t x = 0; x < nstates; x++) {
                    tm[x]  *= state_freq[i];
                    td1[x] *= state_freq[i];
                    td2[x] *= state_freq[i];
                }
                tm  += nstates;
                td1 += nstates;
                td2 += nstates;
            }
        }
    }

#ifdef USE_OPENACC_PROFILE
    if (deriv_profiling) {
        acc_profile.t_deriv_trans_mat += getRealTime() - deriv_prof_t1;
        deriv_prof_t1 = getRealTime();
    }
#endif

    // GPU data: partial likelihoods already resident (gpu_data_resident == true).
    ASSERT(gpu_data_resident && "Derivative kernel requires GPU data from prior likelihood call");

    // D4: Upload transition matrices — create GPU allocation once, then update content
    if (!gpu_trans_mat_resident) {
        #pragma acc enter data create(trans_mat[0:mat_size], \
                                      trans_derv1[0:mat_size], \
                                      trans_derv2[0:mat_size])
        gpu_trans_mat_resident = true;
    }
    #pragma acc update device(trans_mat[0:mat_size], \
                              trans_derv1[0:mat_size], \
                              trans_derv2[0:mat_size])

#ifdef USE_OPENACC_PROFILE
    if (deriv_profiling) {
        #pragma acc wait
        acc_profile.t_deriv_trans_upload += getRealTime() - deriv_prof_t1;
        deriv_prof_t1 = getRealTime();
    }
#endif

    // Capture class member pointers for OpenACC (avoid mapping 'this')
    double *local_ptn_freq  = ptn_freq;
    double *local_ptn_invar = ptn_invar;

    double my_df = 0.0, my_ddf = 0.0;
    double prob_const = 0.0, df_const = 0.0, ddf_const = 0.0;

    // Extract raw pointers for the branch
    double * __restrict__ dad_partial_lh_base = dad_branch->partial_lh;
    UBYTE  * __restrict__ dad_scale_num_base  = dad_branch->scale_num;

    if (dad->isLeaf()) {
        // ---- TIP-INTERNAL path ----
        // Precompute lookup tables for all 3 matrices × all possible tip states
        size_t plh_tip_size = (aln->STATE_UNKNOWN + 1) * block;
        double *partial_lh_node  = new double[plh_tip_size];
        double *partial_lh_derv1 = new double[plh_tip_size];
        double *partial_lh_derv2 = new double[plh_tip_size];

        if (isRootLeaf(dad)) {
            for (size_t c = 0; c < ncat; c++) {
                double *lh_node  = partial_lh_node  + c * nstates;
                double *lh_derv1 = partial_lh_derv1 + c * nstates;
                double *lh_derv2 = partial_lh_derv2 + c * nstates;
                double prop = site_rate->getProp(c);
                model->getStateFrequency(lh_node);
                for (size_t i = 0; i < nstates; i++) {
                    lh_node[i]  *= prop;
                    lh_derv1[i] = 0.0;
                    lh_derv2[i] = 0.0;
                }
            }
        } else {
            double *local_tip_plh = tip_partial_lh;
            for (int state = 0; state <= aln->STATE_UNKNOWN; state++) {
                double *lh_tip = local_tip_plh + state * nstates;
                for (size_t c = 0; c < ncat; c++) {
                    for (size_t i = 0; i < nstates; i++) {
                        double val  = 0.0;
                        double val1 = 0.0;
                        double val2 = 0.0;
                        for (size_t x = 0; x < nstates; x++) {
                            val  += trans_mat[c * nstatesqr + i * nstates + x]   * lh_tip[x];
                            val1 += trans_derv1[c * nstatesqr + i * nstates + x] * lh_tip[x];
                            val2 += trans_derv2[c * nstatesqr + i * nstates + x] * lh_tip[x];
                        }
                        partial_lh_node [state * block + c * nstates + i] = val;
                        partial_lh_derv1[state * block + c * nstates + i] = val1;
                        partial_lh_derv2[state * block + c * nstates + i] = val2;
                    }
                }
            }
        }

#ifdef USE_OPENACC_PROFILE
        if (deriv_profiling) {
            acc_profile.t_deriv_tip_setup += getRealTime() - deriv_prof_t1;
            deriv_prof_t1 = getRealTime();
            acc_profile.n_deriv_tip_int++;
        }
#endif

        // GPU kernel: TIP-INTERNAL derivative reduction
        {
            size_t plh_offset = 0;  // ptn_lower * block
            size_t plh_count  = nptn * block;
            size_t scl_count  = nptn;

            // D2: Reuse tip_states_flat (already GPU-resident from O7)
            // Layout: tip_states_flat[node_id * nptn + ptn]
            int dad_node_id = dad->id;
            int *local_tip_states = tip_states_flat;
            size_t tip_dad_offset = (size_t)dad_node_id * nptn;

            #pragma acc data \
                copyin(partial_lh_node[0:plh_tip_size], \
                       partial_lh_derv1[0:plh_tip_size], \
                       partial_lh_derv2[0:plh_tip_size]) \
                present(local_tip_states[tip_dad_offset:nptn], \
                        dad_partial_lh_base[plh_offset:plh_count], \
                        dad_scale_num_base[0:scl_count], \
                        local_ptn_invar[0:scl_count], \
                        local_ptn_freq[0:scl_count])
            {
                int block_int = (int)block;
                int nstates_int = (int)nstates;
                int ncat_int = (int)ncat;
                int orig_nptn_int = (int)orig_nptn;

                // D6: Vector parallelism on TIP-INTERNAL derivative kernel.
                // Same pattern as D5 (INT-INT): flatten (cc, ii) → single s loop,
                // distribute across vector threads with vector reduction.
                //
                // Simpler than D5: no inner xx dot-product loop — the tip lookup
                // tables (partial_lh_node/derv1/derv2) were pre-computed on CPU
                // with the matrix-vector product already baked in. Each vector
                // thread does just 3 multiplies + 3 adds.
                //
                // tip_idx = state_dad * block + s (coalesced for fixed state_dad)
                // dad_val = dad_partial_lh_base[p * block + s] (coalesced across
                // vector threads)
                //
                // vector_length(32) = 1 warp: reduction uses fast warp shuffles
                // (no cross-warp shared memory barrier). DNA block=16 → 16/32
                // active; Protein block=80 → each thread handles ~3 iterations.
                #pragma acc parallel loop gang vector_length(32) default(present) \
                    reduction(+:my_df, my_ddf, prob_const, df_const, ddf_const)
                for (int p = 0; p < (int)nptn; p++) {
                    int state_dad = local_tip_states[tip_dad_offset + p];
                    double lh_ptn  = 0.0;
                    double df_ptn  = 0.0;
                    double ddf_ptn = 0.0;

                    #pragma acc loop vector reduction(+:lh_ptn, df_ptn, ddf_ptn)
                    for (int s = 0; s < block_int; s++) {
                        int tip_idx = state_dad * block_int + s;
                        double dad_val = dad_partial_lh_base[p * block_int + s];
                        lh_ptn  += partial_lh_node[tip_idx]  * dad_val;
                        df_ptn  += partial_lh_derv1[tip_idx] * dad_val;
                        ddf_ptn += partial_lh_derv2[tip_idx] * dad_val;
                    }

                    // Add invariant site contribution AFTER vector reduction
                    lh_ptn += local_ptn_invar[p];

                    // Normalize and accumulate (gang-level reduction)
                    if (p < orig_nptn_int) {
                        double inv_lh = 1.0 / fabs(lh_ptn);
                        double df_frac  = df_ptn * inv_lh;
                        double ddf_frac = ddf_ptn * inv_lh;
                        my_df  += df_frac * local_ptn_freq[p];
                        my_ddf += (ddf_frac - df_frac * df_frac) * local_ptn_freq[p];
                    } else {
                        if (dad_scale_num_base[p] >= 1) {
                            lh_ptn  *= SCALING_THRESHOLD;
                            df_ptn  *= SCALING_THRESHOLD;
                            ddf_ptn *= SCALING_THRESHOLD;
                        }
                        prob_const += lh_ptn;
                        df_const   += df_ptn;
                        ddf_const  += ddf_ptn;
                    }
                } // FOR p
            } // end acc data

        }

#ifdef USE_OPENACC_PROFILE
        if (deriv_profiling) {
            #pragma acc wait
            acc_profile.t_deriv_kernel += getRealTime() - deriv_prof_t1;
            deriv_prof_t1 = getRealTime();
        }
#endif

        delete[] partial_lh_node;
        delete[] partial_lh_derv1;
        delete[] partial_lh_derv2;

    } else {
        // ---- INTERNAL-INTERNAL path ----
        double * __restrict__ node_partial_lh_base = node_branch->partial_lh;
        UBYTE  * __restrict__ node_scale_num_base  = node_branch->scale_num;

{
#ifdef USE_OPENACC_PROFILE
            if (deriv_profiling) {
                acc_profile.n_deriv_int_int++;
                deriv_prof_t1 = getRealTime();
            }
#endif
            size_t plh_offset = 0;
            size_t plh_count  = nptn * block;
            size_t scl_count  = nptn;

            // D4: trans matrices already on GPU via update device above
            #pragma acc data \
                present(trans_mat[0:mat_size], \
                        trans_derv1[0:mat_size], \
                        trans_derv2[0:mat_size], \
                        dad_partial_lh_base[plh_offset:plh_count], \
                        node_partial_lh_base[plh_offset:plh_count], \
                        dad_scale_num_base[0:scl_count], \
                        node_scale_num_base[0:scl_count], \
                        local_ptn_invar[0:scl_count], \
                        local_ptn_freq[0:scl_count])
            {
                int block_int = (int)block;
                int nstates_int = (int)nstates;
                int nstatesqr_int = (int)nstatesqr;
                int ncat_int = (int)ncat;
                int orig_nptn_int = (int)orig_nptn;

                // D5: Vector parallelism — distribute (cc, ii) work across
                // vector threads within each gang.
                //
                // Flatten (cc, ii) → single index s = cc*nstates + ii (= block
                // elements). Each vector thread computes one row of the matrix-
                // vector product: lh_state = Σ_xx P[cc,ii,xx] · node_plh[p,cc,xx],
                // then multiplies by dad_plh[p,s] and accumulates via vector
                // reduction into lh_ptn/df_ptn/ddf_ptn.
                //
                // Gang: distributes patterns p across warps (gang reduction on
                //        my_df, my_ddf, prob_const, df_const, ddf_const)
                // Vector: distributes s=0..block-1 across threads in each warp
                //         (vector reduction on lh_ptn, df_ptn, ddf_ptn)
                //
                // DNA (4 states, 4 cats):  block=16,  16/32 threads active
                // Protein (20 states, 4 cats): block=80, ~3 iters/thread, all 32 active
                //
                // The inner xx loop (nstates FMAs × 3 matrices) remains sequential
                // per vector thread — each thread does one dot-product row.
                //
                // vector_length(32) = 1 warp: reduction uses fast warp shuffles
                // (no cross-warp shared memory barrier). vector_length(128) caused
                // ~1s regression due to 4-warp shared-memory reduction overhead.
                //
                // IMPORTANT: lh_ptn must be initialized to 0 (not ptn_invar[p])
                // inside the vector reduction. The invariant contribution is added
                // AFTER the reduction to avoid summing it across all vector threads.
                #pragma acc parallel loop gang vector_length(32) default(present) \
                    reduction(+:my_df, my_ddf, prob_const, df_const, ddf_const)
                for (int p = 0; p < (int)nptn; p++) {
                    double lh_ptn  = 0.0;
                    double df_ptn  = 0.0;
                    double ddf_ptn = 0.0;

                    #pragma acc loop vector reduction(+:lh_ptn, df_ptn, ddf_ptn)
                    for (int s = 0; s < block_int; s++) {
                        int cc = s / nstates_int;
                        int ii = s % nstates_int;

                        double lh_state  = 0.0;
                        double df_state  = 0.0;
                        double ddf_state = 0.0;
                        #pragma acc loop seq
                        for (int xx = 0; xx < nstates_int; xx++) {
                            int mat_idx = cc * nstatesqr_int + ii * nstates_int + xx;
                            double plh = node_partial_lh_base[p * block_int + cc * nstates_int + xx];
                            lh_state  += trans_mat[mat_idx]   * plh;
                            df_state  += trans_derv1[mat_idx] * plh;
                            ddf_state += trans_derv2[mat_idx] * plh;
                        }
                        double dad_val = dad_partial_lh_base[p * block_int + s];
                        lh_ptn  += dad_val * lh_state;
                        df_ptn  += dad_val * df_state;
                        ddf_ptn += dad_val * ddf_state;
                    }

                    // Add invariant site contribution AFTER vector reduction
                    lh_ptn += local_ptn_invar[p];

                    // Normalize and accumulate (gang-level reduction)
                    if (p < orig_nptn_int) {
                        double inv_lh = 1.0 / fabs(lh_ptn);
                        double df_frac  = df_ptn * inv_lh;
                        double ddf_frac = ddf_ptn * inv_lh;
                        my_df  += df_frac * local_ptn_freq[p];
                        my_ddf += (ddf_frac - df_frac * df_frac) * local_ptn_freq[p];
                    } else {
                        if ((dad_scale_num_base[p] + node_scale_num_base[p]) >= 1) {
                            lh_ptn  *= SCALING_THRESHOLD;
                            df_ptn  *= SCALING_THRESHOLD;
                            ddf_ptn *= SCALING_THRESHOLD;
                        }
                        prob_const += lh_ptn;
                        df_const   += df_ptn;
                        ddf_const  += ddf_ptn;
                    }
                } // FOR p
            } // end acc data
        }
#ifdef USE_OPENACC_PROFILE
        if (deriv_profiling) {
            #pragma acc wait
            acc_profile.t_deriv_kernel += getRealTime() - deriv_prof_t1;
            deriv_prof_t1 = getRealTime();
        }
#endif
    }

    // Ascertainment bias correction (on host)
    if (orig_nptn < nptn) {
        prob_const = 1.0 - prob_const;
        double df_frac  = df_const / prob_const;
        double ddf_frac = ddf_const / prob_const;
        size_t nsites = aln->getNSite();
        my_df  += nsites * df_frac;
        my_ddf += nsites * (ddf_frac + df_frac * df_frac);
    }

    *df  = my_df;
    *ddf = my_ddf;

    if (!std::isfinite(*df) || !std::isfinite(*ddf)) {
        cout << "WARNING: Numerical underflow for OpenACC lh-derivative" << endl;
        *df = *ddf = 0.0;
    }

    // D4: trans matrices are persistent — don't delete here

#ifdef USE_OPENACC_PROFILE
    if (deriv_profiling) {
        acc_profile.t_deriv_postproc += getRealTime() - deriv_prof_t1;
        acc_profile.t_deriv_total += getRealTime() - deriv_prof_t0;
    }
#endif
}

// ==========================================================================
// Reversible OpenACC kernels REMOVED.
// OpenACC forces kernel_nonrev = true (state-space P(t) path) for all models,
// so the eigenspace reversible kernels (computeRevPartialLikelihoodOpenACC,
// computeRevLikelihoodBranchOpenACC) are never dispatched. Removed to avoid
// maintaining dead code with known bugs (e.g. tmp_state[4] overflow for
// protein models with nstates=20).
// ==========================================================================

// ==========================================================================
// Persistent GPU data cleanup
// Called from destructor and initializeAllPartialLh (before host realloc).
// ==========================================================================

void PhyloTree::freeOpenACCData() {
    if (!gpu_data_resident) return;

    if (verbose_mode >= VB_MED)
        cout << "OpenACC: Freeing persistent GPU data" << endl;

    // Use the saved pointers and sizes from the enter data create/copyin call.
    // These must match exactly — OpenACC tracks device data by host address.
    // Note: central_partial_lh and central_scale_num were allocated with 'create'
    // (O1 optimization — no host-to-device upload). exit data delete works the
    // same for 'create' and 'copyin' data.
    #pragma acc exit data \
        delete(gpu_central_plh_ptr[0:gpu_total_lh_entries], \
               gpu_central_scl_ptr[0:gpu_total_scale_entries], \
               gpu_ptn_freq_ptr[0:gpu_nptn], \
               gpu_ptn_invar_ptr[0:gpu_nptn], \
               gpu_pattern_lh_ptr[0:gpu_nptn], \
               gpu_pattern_lh_cat_ptr[0:gpu_nptn_ncat])

    // D3: Free buffer_partial_lh from GPU if still resident
    if (gpu_buffer_plh_resident && gpu_buffer_plh_ptr) {
        #pragma acc exit data delete(gpu_buffer_plh_ptr[0:gpu_buffer_plh_size])
        gpu_buffer_plh_resident = false;
        gpu_buffer_plh_ptr = nullptr;
    }

    // D4: Free persistent transition matrices from GPU and host
    if (gpu_trans_mat_resident && gpu_trans_mat) {
        #pragma acc exit data delete(gpu_trans_mat[0:gpu_trans_mat_size], \
                                     gpu_trans_derv1[0:gpu_trans_mat_size], \
                                     gpu_trans_derv2[0:gpu_trans_mat_size])
        gpu_trans_mat_resident = false;
    }
    delete[] gpu_trans_mat;   gpu_trans_mat = nullptr;
    delete[] gpu_trans_derv1; gpu_trans_derv1 = nullptr;
    delete[] gpu_trans_derv2; gpu_trans_derv2 = nullptr;
    gpu_trans_mat_size = 0;

    // P2: Free tip_states_flat from GPU and host
    if (gpu_tip_states_ptr) {
        #pragma acc exit data delete(gpu_tip_states_ptr[0:gpu_tip_states_size])
        delete[] tip_states_flat;
        tip_states_flat = nullptr;
        gpu_tip_states_ptr = nullptr;
        gpu_tip_states_size = 0;
    }

    gpu_data_resident = false;
    gpu_central_plh_ptr = nullptr;
    gpu_central_scl_ptr = nullptr;
    gpu_ptn_freq_ptr = nullptr;
    gpu_ptn_invar_ptr = nullptr;
    gpu_pattern_lh_ptr = nullptr;
    gpu_pattern_lh_cat_ptr = nullptr;
}

#endif // USE_OPENACC
