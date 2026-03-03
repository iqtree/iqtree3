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

#include <cmath>
#include <iostream>
#include <vector>
#include <openacc.h>            // Step 12: acc_is_present (future dynamic checks)

using namespace std;

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
    double *dad_partial_lh  = dad_branch->partial_lh;
    UBYTE  *dad_scale_num   = dad_branch->scale_num;

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
                    if (child->node == root)
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

        double *tip_lh_left = partial_lh_leaves;
        double *tip_lh_right = partial_lh_leaves + (aln->STATE_UNKNOWN+1)*block;

        if (right->node == root) {
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
        bool left_is_root = (left->node == root);
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
                #pragma acc parallel loop gang vector
                for (size_t p = ptn_lower; p < ptn_upper; p++)
                    dad_scale_num[p] = 0;

                // Main TIP-TIP kernel: element-wise product of precomputed tip lookups
                #pragma acc parallel loop gang
                for (size_t p = ptn_lower; p < ptn_upper; p++) {
                    size_t idx = p - ptn_lower;
                    int sl = states_left[idx];
                    int sr = states_right[idx];
                    #pragma acc loop vector
                    for (size_t s = 0; s < block; s++) {
                        dad_partial_lh[p * block + s] =
                            tip_lh_left[sl * block + s] * tip_lh_right[sr * block + s];
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

        // A1: Extract raw pointers
        double *right_partial_lh = right->partial_lh;
        UBYTE  *right_scale_num  = right->scale_num;
        double *tip_lh_left = partial_lh_leaves;

        // A3: Precompute per-pattern tip states (CPU side, before GPU region)
        size_t nptn_range = ptn_upper - ptn_lower;
        int *states_left = new int[nptn_range];
        bool left_is_root = (left->node == root);
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
                #pragma acc parallel loop gang
                for (size_t p = ptn_lower; p < ptn_upper; p++) {
                    size_t idx = p - ptn_lower;
                    int state_left = states_left[idx];

                    // Copy scale_num from right child
                    dad_scale_num[p] = right_scale_num[p];

                    #pragma acc loop vector
                    for (size_t s = 0; s < block; s++) {
                        size_t cx = s;  // for ncat=1, c*nstates+x == s
                        // Right child: dot product P_right × right_partial_lh
                        double vright = 0.0;
                        for (size_t k = 0; k < nstates; k++) {
                            vright += eright[(cx / nstates) * nstatesqr_local + (cx % nstates) * nstates + k]
                                    * right_partial_lh[p * block + (cx / nstates) * nstates + k];
                        }
                        // Left child: precomputed tip lookup (no matrix multiply)
                        double vleft_val = tip_lh_left[state_left * block + s];
                        dad_partial_lh[p * block + s] = vleft_val * vright;
                    }
                }

                // Kernel 2: Scaling check (separate pass, matches PoC pattern)
                // Uses gang+vector with reduction(max:) per pattern
                #pragma acc parallel loop gang
                for (size_t p = ptn_lower; p < ptn_upper; p++) {
                    double lh_max = 0.0;
                    #pragma acc loop vector reduction(max:lh_max)
                    for (size_t s = 0; s < block; s++) {
                        double v = dad_partial_lh[p * block + s];
                        if (v > lh_max) lh_max = v;
                    }
                    if (lh_max == 0.0) {
                        // All-zero: replace with unknown-state partial
                        #pragma acc loop seq
                        for (size_t s = 0; s < block; s++)
                            dad_partial_lh[p * block + s] = local_tip_plh[state_unknown * nstates + (s % nstates)];
                        dad_scale_num[p] += 4;
                    } else if (lh_max < SCALING_THRESHOLD) {
                        // Underflow: scale up by 2^256
                        #pragma acc loop seq
                        for (size_t s = 0; s < block; s++)
                            dad_partial_lh[p * block + s] = ldexp(dad_partial_lh[p * block + s], SCALING_THRESHOLD_EXP);
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

        // A1: Extract raw pointers
        double *left_partial_lh  = left->partial_lh;
        double *right_partial_lh = right->partial_lh;
        UBYTE  *left_scale_num   = left->scale_num;
        UBYTE  *right_scale_num  = right->scale_num;

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
                #pragma acc parallel loop gang
                for (size_t p = ptn_lower; p < ptn_upper; p++) {
                    // Propagate scale counts from both children
                    dad_scale_num[p] = left_scale_num[p] + right_scale_num[p];

                    #pragma acc loop vector
                    for (size_t s = 0; s < block; s++) {
                        size_t cx = s;  // for ncat=1, c*nstates+x == s
                        double vleft = 0.0, vright = 0.0;
                        for (size_t k = 0; k < nstates; k++) {
                            vleft  += eleft[(cx / nstates) * nstatesqr_local + (cx % nstates) * nstates + k]
                                    * left_partial_lh[p * block + (cx / nstates) * nstates + k];
                            vright += eright[(cx / nstates) * nstatesqr_local + (cx % nstates) * nstates + k]
                                    * right_partial_lh[p * block + (cx / nstates) * nstates + k];
                        }
                        dad_partial_lh[p * block + s] = vleft * vright;
                    }
                }

                // Kernel 2: Scaling check (separate pass, matches PoC pattern)
                // Uses gang+vector with reduction(max:) per pattern
                #pragma acc parallel loop gang
                for (size_t p = ptn_lower; p < ptn_upper; p++) {
                    double lh_max = 0.0;
                    #pragma acc loop vector reduction(max:lh_max)
                    for (size_t s = 0; s < block; s++) {
                        double v = dad_partial_lh[p * block + s];
                        if (v > lh_max) lh_max = v;
                    }
                    if (lh_max == 0.0) {
                        // All-zero: replace with unknown-state partial
                        #pragma acc loop seq
                        for (size_t s = 0; s < block; s++)
                            dad_partial_lh[p * block + s] = local_tip_plh[state_unknown * nstates + (s % nstates)];
                        dad_scale_num[p] += 4;
                    } else if (lh_max < SCALING_THRESHOLD) {
                        // Underflow: scale up by 2^256
                        #pragma acc loop seq
                        for (size_t s = 0; s < block; s++)
                            dad_partial_lh[p * block + s] = ldexp(dad_partial_lh[p * block + s], SCALING_THRESHOLD_EXP);
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

    ASSERT(rooted);

    PhyloNode *node = (PhyloNode*) dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*) node->findNeighbor(dad);
    if (!central_partial_lh)
        initializeAllPartialLh();
    if (node->isLeaf() || (dad_branch->direction == AWAYFROM_ROOT && dad != root)) {
        PhyloNode *tmp_node = dad;
        dad = node;
        node = tmp_node;
        PhyloNeighbor *tmp_nei = dad_branch;
        dad_branch = node_branch;
        node_branch = tmp_nei;
    }

    // Build traversal order and precompute P(t) / tip lookup tables.
    Params::getInstance().kernel_nonrev = true;
    computeTraversalInfo<Vec1d>(node, dad, false);

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

    double *trans_mat = new double[block*nstates];
    for (c = 0; c < ncat; c++) {
        double len = site_rate->getRate(c)*dad_branch->length;
        double prop = site_rate->getProp(c);
        double *this_trans_mat = &trans_mat[c*nstatesqr];
        model->computeTransMatrix(len, this_trans_mat);
        for (i = 0; i < nstatesqr; i++)
            this_trans_mat[i] *= prop;
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

        #pragma acc enter data \
            copyin(local_central_plh[0:total_lh_entries], \
                   local_central_scl[0:total_scale_entries], \
                   local_ptn_freq[0:nptn], \
                   local_ptn_invar[0:nptn]) \
            create(local_pattern_lh[0:nptn], \
                   local_pattern_lh_cat[0:nptn_ncat])

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

        if (verbose_mode >= VB_MED)
            cout << "OpenACC: Uploaded persistent GPU data ("
                 << (total_lh_entries * sizeof(double) + total_scale_entries * sizeof(UBYTE)) / (1024*1024)
                 << " MB)" << endl;
    }

    double prob_const = 0.0;

    // A1: Extract raw pointers for the branch
    double *dad_partial_lh_base = dad_branch->partial_lh;
    UBYTE  *dad_scale_num_base  = dad_branch->scale_num;

    if (dad->isLeaf()) {
        // special treatment for TIP-INTERNAL NODE case
        double *partial_lh_node = new double[(aln->STATE_UNKNOWN+1)*block];
        if (dad == root) {
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

        // now do the real computation
        for (int packet_id = 0; packet_id < num_packets; packet_id++) {
            size_t ptn_lower = limits[packet_id];
            size_t ptn_upper = limits[packet_id+1];
            // first compute partial_lh (on GPU via Steps 9-10)
            for (vector<TraversalInfo>::iterator it = traversal_info.begin(); it != traversal_info.end(); it++)
                computePartialLikelihood(*it, ptn_lower, ptn_upper, packet_id);

            // A3: Precompute per-pattern dad states (CPU side, before GPU region)
            size_t nptn_range = ptn_upper - ptn_lower;
            int *states_dad = new int[nptn_range];
            bool dad_is_root = (dad == root);
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
                    #pragma acc parallel loop gang reduction(+:tree_lh, prob_const)
                    for (size_t p = ptn_lower; p < ptn_upper; p++) {
                        size_t idx = p - ptn_lower;
                        int state_dad = states_dad[idx];
                        double lh_ptn = local_ptn_invar[p];

                        // Dot product: partial_lh_node[state] · dad_partial_lh[p]
                        // For ncat=1: single category, block == nstates
                        for (size_t cc = 0; cc < ncat; cc++) {
                            double lh_cat = 0.0;
                            for (size_t ii = 0; ii < nstates; ii++) {
                                lh_cat += partial_lh_node[state_dad*block + cc*nstates + ii]
                                        * dad_partial_lh_base[p*block + cc*nstates + ii];
                            }
                            local_pattern_lh_cat[p*ncat + cc] = lh_cat;
                            lh_ptn += lh_cat;
                        }

                        // Log-likelihood + scaling correction
                        if (p < orig_nptn) {
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

            delete [] states_dad;
        } // FOR packet_id
        delete [] partial_lh_node;
    } else {

        // both dad and node are internal nodes
        // A1: Extract raw pointers
        double *node_partial_lh_base = node_branch->partial_lh;
        UBYTE  *node_scale_num_base  = node_branch->scale_num;

        for (int packet_id = 0; packet_id < num_packets; packet_id++) {
            size_t ptn_lower = limits[packet_id];
            size_t ptn_upper = limits[packet_id+1];
            // first compute partial_lh (on GPU via Steps 9-10)
            for (vector<TraversalInfo>::iterator it = traversal_info.begin(); it != traversal_info.end(); it++)
                computePartialLikelihood(*it, ptn_lower, ptn_upper, packet_id);

            // Step 11: Log-likelihood reduction offloaded to GPU via OpenACC
            // INTERNAL-INTERNAL case: both dad and node are internal nodes.
            // For each pattern: compute trans_mat × node_partial_lh, dot with
            // dad_partial_lh, take log + scaling correction, accumulate weighted sum.
            // Matches PoC: gang over patterns, reduction(+:logLikelihood).
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
                    #pragma acc parallel loop gang reduction(+:tree_lh, prob_const)
                    for (size_t p = ptn_lower; p < ptn_upper; p++) {
                        double lh_ptn = local_ptn_invar[p];

                        for (size_t cc = 0; cc < ncat; cc++) {
                            double lh_cat = 0.0;
                            for (size_t ii = 0; ii < nstates; ii++) {
                                // Matrix-vector product: trans_mat × node_partial_lh
                                double lh_state = 0.0;
                                for (size_t xx = 0; xx < nstates; xx++)
                                    lh_state += trans_mat[cc*nstatesqr + ii*nstates + xx]
                                              * node_partial_lh_base[p*block + cc*nstates + xx];
                                lh_cat += dad_partial_lh_base[p*block + cc*nstates + ii] * lh_state;
                            }
                            local_pattern_lh_cat[p*ncat + cc] = lh_cat;
                            lh_ptn += lh_cat;
                        }

                        // Log-likelihood + scaling correction
                        if (p < orig_nptn) {
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
        } // FOR packet_id
    }

    // ====== Copy _pattern_lh back to host ======
    // Only _pattern_lh is needed on the host (for prob_const correction and
    // IQ-TREE callers). All other GPU data stays resident — no download of
    // central_partial_lh or central_scale_num needed since they remain
    // present() for subsequent calls.
    #pragma acc update self(local_pattern_lh[0:nptn])

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

    delete [] trans_mat;
    return tree_lh;
}

// ==========================================================================
// Step 13: Reversible Partial Likelihood Kernel (OpenACC)
//
// Same Felsenstein pruning as non-rev, but:
//   - echildren = U * diag(exp(lambda*t))  (half-factored P(t))
//   - partial_lh stored in eigenspace: U^{-1} * L_real
//   - After Hadamard product in state space, back-transform via inv_evec
//
// Math per pattern (INTERNAL-INTERNAL):
//   tmp[x] = (Σ_i eleft[x][i] * plh_left[i]) * (Σ_i eright[x][i] * plh_right[i])
//   dad_plh[i] = Σ_x inv_evec[i][x] * tmp[x]
// ==========================================================================

void PhyloTree::computeRevPartialLikelihoodOpenACC(TraversalInfo &info, size_t ptn_lower, size_t ptn_upper, int packet_id) {
    // NOTE: packet_id is not used yet — see comment in computePartialLikelihoodGenericOpenACC.
    (void)packet_id;

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

    PhyloNeighbor *left = NULL, *right = NULL;
    FOR_NEIGHBOR_IT(node, dad, it) {
        if (!left) left = (PhyloNeighbor*)(*it); else right = (PhyloNeighbor*)(*it);
    }

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

    double *dad_partial_lh  = dad_branch->partial_lh;
    UBYTE  *dad_scale_num   = dad_branch->scale_num;

    // Get inverse eigenvectors for back-transform (eigenspace storage)
    double *inv_evec = model->getInverseEigenvectors();
    ASSERT(inv_evec != NULL);

    if (node->degree() > 3) {

        /*--------------------- multifurcating node (rev) ------------------*/
        // Sequential for now. The Hadamard product happens in state space,
        // then the result is transformed back to eigenspace via inv_evec.

        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            // Initialize tmp in state space to 1.0
            double tmp_state[block];
            for (i = 0; i < block; i++)
                tmp_state[i] = 1.0;
            dad_scale_num[ptn] = 0;

            double *partial_lh_leaf = partial_lh_leaves;
            double *echild = echildren;

            FOR_NEIGHBOR_IT(node, dad, it) {
                PhyloNeighbor *child = (PhyloNeighbor*)*it;
                if (child->node->isLeaf()) {
                    int state_child;
                    if (child->node == root)
                        state_child = 0;
                    else
                        state_child = (ptn < orig_ntn) ? (aln->at(ptn))[child->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn][child->node->id];
                    double *child_lh = partial_lh_leaf + state_child*block;
                    for (c = 0; c < block; c++)
                        tmp_state[c] *= child_lh[c];
                    partial_lh_leaf += (aln->STATE_UNKNOWN+1)*block;
                } else {
                    double *child_partial_lh = child->partial_lh;
                    UBYTE  *child_scale_num  = child->scale_num;
                    dad_scale_num[ptn] += child_scale_num[ptn];

                    for (c = 0; c < ncat; c++) {
                        for (x = 0; x < nstates; x++) {
                            double vchild = 0.0;
                            for (i = 0; i < nstates; i++)
                                vchild += echild[c*nstatesqr + x*nstates + i] * child_partial_lh[ptn*block + c*nstates + i];
                            tmp_state[c*nstates + x] *= vchild;
                        }
                    }
                }
                echild += block*nstates;
            }

            // Back-transform from state space to eigenspace
            for (c = 0; c < ncat; c++) {
                for (i = 0; i < nstates; i++) {
                    double v = 0.0;
                    for (x = 0; x < nstates; x++)
                        v += inv_evec[i*nstates + x] * tmp_state[c*nstates + x];
                    dad_partial_lh[ptn*block + c*nstates + i] = v;
                }
            }

            // Scaling check
            double lh_max = fabs(dad_partial_lh[ptn*block]);
            for (i = 1; i < block; i++) {
                double v = fabs(dad_partial_lh[ptn*block + i]);
                if (v > lh_max) lh_max = v;
            }
            if (lh_max == 0.0) {
                for (c = 0; c < ncat; c++)
                    for (i = 0; i < nstates; i++)
                        dad_partial_lh[ptn*block + c*nstates + i] = tip_partial_lh[aln->STATE_UNKNOWN*nstates + i];
                dad_scale_num[ptn] += 4;
            } else if (lh_max < SCALING_THRESHOLD) {
                for (i = 0; i < block; i++)
                    dad_partial_lh[ptn*block + i] = ldexp(dad_partial_lh[ptn*block + i], SCALING_THRESHOLD_EXP);
                dad_scale_num[ptn] += 1;
            }
        }

    } else if (left->node->isLeaf() && right->node->isLeaf()) {

        /*--------------------- TIP-TIP (rev) ------------------*/
        // partial_lh_leaves are pre-contracted: already in state space.
        // Hadamard product in state space, then back-transform via inv_evec.

        double *tip_lh_left = partial_lh_leaves;
        double *tip_lh_right = partial_lh_leaves + (aln->STATE_UNKNOWN+1)*block;

        if (right->node == root) {
            PhyloNeighbor *tmp = left; left = right; right = tmp;
            double *etmp = eleft; eleft = eright; eright = etmp;
            etmp = tip_lh_left; tip_lh_left = tip_lh_right; tip_lh_right = etmp;
        }

        size_t nptn_range = ptn_upper - ptn_lower;
        int *states_left  = new int[nptn_range];
        int *states_right = new int[nptn_range];
        bool left_is_root = (left->node == root);
        int left_node_id  = left->node->id;
        int right_node_id = right->node->id;

        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            size_t idx = ptn - ptn_lower;
            if (left_is_root) states_left[idx] = 0;
            else states_left[idx] = (ptn < orig_ntn) ? (aln->at(ptn))[left_node_id] : model_factory->unobserved_ptns[ptn-orig_ntn][left_node_id];
            states_right[idx] = (ptn < orig_ntn) ? (aln->at(ptn))[right_node_id] : model_factory->unobserved_ptns[ptn-orig_ntn][right_node_id];
        }

        // GPU kernel: Hadamard + inv_evec back-transform
        {
            size_t tip_lh_size = (aln->STATE_UNKNOWN + 1) * block;
            size_t inv_evec_size = nstatesqr;
            size_t plh_offset = ptn_lower * block;
            size_t plh_count  = (ptn_upper - ptn_lower) * block;
            size_t scl_offset = ptn_lower;
            size_t scl_count  = ptn_upper - ptn_lower;

            #pragma acc data \
                copyin(states_left[0:nptn_range], states_right[0:nptn_range], \
                       tip_lh_left[0:tip_lh_size], tip_lh_right[0:tip_lh_size], \
                       inv_evec[0:inv_evec_size]) \
                present(dad_partial_lh[plh_offset:plh_count], \
                        dad_scale_num[scl_offset:scl_count])
            {
                #pragma acc parallel loop gang vector
                for (size_t p = ptn_lower; p < ptn_upper; p++)
                    dad_scale_num[p] = 0;

                #pragma acc parallel loop gang
                for (size_t p = ptn_lower; p < ptn_upper; p++) {
                    size_t idx = p - ptn_lower;
                    int sl = states_left[idx];
                    int sr = states_right[idx];
                    // Hadamard in state space, then inv_evec back-transform
                    for (size_t cc = 0; cc < ncat; cc++) {
                        double tmp_state[4]; // nstates max for DNA
                        for (size_t xx = 0; xx < nstates; xx++)
                            tmp_state[xx] = tip_lh_left[sl*block + cc*nstates + xx]
                                          * tip_lh_right[sr*block + cc*nstates + xx];
                        for (size_t ii = 0; ii < nstates; ii++) {
                            double v = 0.0;
                            for (size_t xx = 0; xx < nstates; xx++)
                                v += inv_evec[ii*nstates + xx] * tmp_state[xx];
                            dad_partial_lh[p*block + cc*nstates + ii] = v;
                        }
                    }
                }
            }
        }

        delete [] states_left;
        delete [] states_right;

    } else if (left->node->isLeaf() && !right->node->isLeaf()) {

        /*--------------------- TIP-INTERNAL (rev) ------------------*/
        // Left: pre-contracted tip (state space).
        // Right: eigenspace partial_lh, transformed via eright to state space.
        // Hadamard in state space, then inv_evec back-transform.

        double *right_partial_lh = right->partial_lh;
        UBYTE  *right_scale_num  = right->scale_num;
        double *tip_lh_left = partial_lh_leaves;

        size_t nptn_range = ptn_upper - ptn_lower;
        int *states_left = new int[nptn_range];
        bool left_is_root = (left->node == root);
        int left_node_id = left->node->id;

        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            size_t idx = ptn - ptn_lower;
            if (left_is_root) states_left[idx] = 0;
            else states_left[idx] = (ptn < orig_ntn) ? (aln->at(ptn))[left_node_id] : model_factory->unobserved_ptns[ptn-orig_ntn][left_node_id];
        }

        {
            size_t tip_lh_size = (aln->STATE_UNKNOWN + 1) * block;
            size_t eright_size = block * nstates;
            size_t inv_evec_size = nstatesqr;
            size_t plh_offset  = ptn_lower * block;
            size_t plh_count   = (ptn_upper - ptn_lower) * block;
            size_t scl_offset  = ptn_lower;
            size_t scl_count   = ptn_upper - ptn_lower;
            size_t tip_unknown_size = (aln->STATE_UNKNOWN + 1) * nstates;
            size_t state_unknown = aln->STATE_UNKNOWN;
            double *local_tip_plh = tip_partial_lh;

            #pragma acc data \
                copyin(states_left[0:nptn_range], \
                       tip_lh_left[0:tip_lh_size], \
                       eright[0:eright_size], \
                       inv_evec[0:inv_evec_size]) \
                present(right_partial_lh[plh_offset:plh_count], \
                        right_scale_num[scl_offset:scl_count], \
                        local_tip_plh[0:tip_unknown_size], \
                        dad_partial_lh[plh_offset:plh_count], \
                        dad_scale_num[scl_offset:scl_count])
            {
                // Kernel 1: Partial likelihoods with inv_evec back-transform
                #pragma acc parallel loop gang
                for (size_t p = ptn_lower; p < ptn_upper; p++) {
                    size_t idx = p - ptn_lower;
                    int state_left = states_left[idx];
                    dad_scale_num[p] = right_scale_num[p];

                    for (size_t cc = 0; cc < ncat; cc++) {
                        double tmp_state[4];
                        for (size_t xx = 0; xx < nstates; xx++) {
                            // Right child: eigenspace → state space via eright
                            double vright = 0.0;
                            for (size_t kk = 0; kk < nstates; kk++)
                                vright += eright[cc*nstatesqr + xx*nstates + kk]
                                        * right_partial_lh[p*block + cc*nstates + kk];
                            // Left child: already in state space from partial_lh_leaves
                            tmp_state[xx] = tip_lh_left[state_left*block + cc*nstates + xx] * vright;
                        }
                        // Back-transform to eigenspace
                        for (size_t ii = 0; ii < nstates; ii++) {
                            double v = 0.0;
                            for (size_t xx = 0; xx < nstates; xx++)
                                v += inv_evec[ii*nstates + xx] * tmp_state[xx];
                            dad_partial_lh[p*block + cc*nstates + ii] = v;
                        }
                    }
                }

                // Kernel 2: Scaling check
                #pragma acc parallel loop gang
                for (size_t p = ptn_lower; p < ptn_upper; p++) {
                    double lh_max = 0.0;
                    for (size_t s = 0; s < block; s++) {
                        double v = fabs(dad_partial_lh[p * block + s]);
                        if (v > lh_max) lh_max = v;
                    }
                    if (lh_max == 0.0) {
                        #pragma acc loop seq
                        for (size_t s = 0; s < block; s++)
                            dad_partial_lh[p*block + s] = local_tip_plh[state_unknown * nstates + (s % nstates)];
                        dad_scale_num[p] += 4;
                    } else if (lh_max < SCALING_THRESHOLD) {
                        #pragma acc loop seq
                        for (size_t s = 0; s < block; s++)
                            dad_partial_lh[p*block + s] = ldexp(dad_partial_lh[p*block + s], SCALING_THRESHOLD_EXP);
                        dad_scale_num[p] += 1;
                    }
                }
            }
        }

        delete [] states_left;

    } else {

        /*--------------------- INTERNAL-INTERNAL (rev) ------------------*/
        // Both children in eigenspace. Transform to state space via eleft/eright,
        // Hadamard product, then back-transform via inv_evec.

        double *left_partial_lh  = left->partial_lh;
        double *right_partial_lh = right->partial_lh;
        UBYTE  *left_scale_num   = left->scale_num;
        UBYTE  *right_scale_num  = right->scale_num;

        {
            size_t eleft_size  = block * nstates;
            size_t eright_size = block * nstates;
            size_t inv_evec_size = nstatesqr;
            size_t plh_offset  = ptn_lower * block;
            size_t plh_count   = (ptn_upper - ptn_lower) * block;
            size_t scl_offset  = ptn_lower;
            size_t scl_count   = ptn_upper - ptn_lower;
            size_t tip_unknown_size = (aln->STATE_UNKNOWN + 1) * nstates;
            size_t state_unknown = aln->STATE_UNKNOWN;
            double *local_tip_plh = tip_partial_lh;

            #pragma acc data \
                copyin(eleft[0:eleft_size], eright[0:eright_size], \
                       inv_evec[0:inv_evec_size]) \
                present(left_partial_lh[plh_offset:plh_count], \
                        right_partial_lh[plh_offset:plh_count], \
                        left_scale_num[scl_offset:scl_count], \
                        right_scale_num[scl_offset:scl_count], \
                        local_tip_plh[0:tip_unknown_size], \
                        dad_partial_lh[plh_offset:plh_count], \
                        dad_scale_num[scl_offset:scl_count])
            {
                // Kernel 1: Dual dot products + Hadamard + inv_evec back-transform
                #pragma acc parallel loop gang
                for (size_t p = ptn_lower; p < ptn_upper; p++) {
                    dad_scale_num[p] = left_scale_num[p] + right_scale_num[p];

                    for (size_t cc = 0; cc < ncat; cc++) {
                        double tmp_state[4];
                        for (size_t xx = 0; xx < nstates; xx++) {
                            double vleft = 0.0, vright = 0.0;
                            for (size_t kk = 0; kk < nstates; kk++) {
                                vleft  += eleft[cc*nstatesqr + xx*nstates + kk]
                                        * left_partial_lh[p*block + cc*nstates + kk];
                                vright += eright[cc*nstatesqr + xx*nstates + kk]
                                        * right_partial_lh[p*block + cc*nstates + kk];
                            }
                            tmp_state[xx] = vleft * vright;
                        }
                        for (size_t ii = 0; ii < nstates; ii++) {
                            double v = 0.0;
                            for (size_t xx = 0; xx < nstates; xx++)
                                v += inv_evec[ii*nstates + xx] * tmp_state[xx];
                            dad_partial_lh[p*block + cc*nstates + ii] = v;
                        }
                    }
                }

                // Kernel 2: Scaling check
                #pragma acc parallel loop gang
                for (size_t p = ptn_lower; p < ptn_upper; p++) {
                    double lh_max = 0.0;
                    for (size_t s = 0; s < block; s++) {
                        double v = fabs(dad_partial_lh[p * block + s]);
                        if (v > lh_max) lh_max = v;
                    }
                    if (lh_max == 0.0) {
                        #pragma acc loop seq
                        for (size_t s = 0; s < block; s++)
                            dad_partial_lh[p*block + s] = local_tip_plh[state_unknown * nstates + (s % nstates)];
                        dad_scale_num[p] += 4;
                    } else if (lh_max < SCALING_THRESHOLD) {
                        #pragma acc loop seq
                        for (size_t s = 0; s < block; s++)
                            dad_partial_lh[p*block + s] = ldexp(dad_partial_lh[p*block + s], SCALING_THRESHOLD_EXP);
                        dad_scale_num[p] += 1;
                    }
                }
            }
        }
    }
}

// ==========================================================================
// Step 13: Reversible Log-Likelihood Kernel (OpenACC)
//
// Reduction in eigenspace:
//   val[i] = exp(eval[i] * rate * t) * prop
//   TIP-INT:  lh = Σ_i (val[i] * tip_plh[state][i]) * dad_plh[i]
//   INT-INT:  lh = Σ_i val[i] * node_plh[i] * dad_plh[i]
//
// Simpler than non-rev (no matrix-vector product in reduction).
// ==========================================================================

double PhyloTree::computeRevLikelihoodBranchOpenACC(PhyloNeighbor *dad_branch, PhyloNode *dad, bool save_log_value) {

    static bool openacc_rev_kernel_printed = false;
    if (!openacc_rev_kernel_printed) {
        cout << "OpenACC: Using GPU-ready reversible likelihood kernel "
             << "(computeRevPartialLikelihoodOpenACC + "
             << "computeRevLikelihoodBranchOpenACC)" << endl;
        openacc_rev_kernel_printed = true;
    }

    ASSERT(rooted);

    PhyloNode *node = (PhyloNode*) dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*) node->findNeighbor(dad);
    if (!central_partial_lh)
        initializeAllPartialLh();
    if (node->isLeaf() || (dad_branch->direction == AWAYFROM_ROOT && dad != root)) {
        PhyloNode *tmp_node = dad;
        dad = node;
        node = tmp_node;
        PhyloNeighbor *tmp_nei = dad_branch;
        dad_branch = node_branch;
        node_branch = tmp_nei;
    }

    computeTraversalInfo<Vec1d>(node, dad, false);

    double tree_lh = 0.0;
    size_t nstates = aln->num_states;
    size_t nstatesqr = nstates*nstates;
    size_t ncat = site_rate->getNRate();

    size_t block = ncat * nstates;
    size_t ptn;
    size_t c, i;
    size_t orig_nptn = aln->size();
    size_t nptn = aln->size()+model_factory->unobserved_ptns.size();

    vector<size_t> limits;
    computeBounds<Vec1d>(num_threads, num_packets, nptn, limits);

    // Precompute val[i] = exp(eval[i] * rate * t) * prop
    // This replaces the full trans_mat used in non-rev.
    double *eval = model->getEigenvalues();
    ASSERT(eval != NULL);
    double *val = new double[block];
    for (c = 0; c < ncat; c++) {
        double len = site_rate->getRate(c) * dad_branch->length;
        double prop = site_rate->getProp(c);
        for (i = 0; i < nstates; i++)
            val[c*nstates + i] = exp(eval[i] * len) * prop;
    }

    // GPU data upload (same as non-rev Step 12)
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
    size_t nptn_ncat = nptn * ncat;

    double *local_central_plh = central_partial_lh;
    UBYTE  *local_central_scl = central_scale_num;
    double *local_ptn_freq = ptn_freq;
    double *local_ptn_invar = ptn_invar;
    double *local_pattern_lh = _pattern_lh;
    double *local_pattern_lh_cat = _pattern_lh_cat;

    #pragma acc enter data \
        copyin(local_central_plh[0:total_lh_entries], \
               local_central_scl[0:total_scale_entries], \
               local_ptn_freq[0:nptn], \
               local_ptn_invar[0:nptn]) \
        create(local_pattern_lh[0:nptn], \
               local_pattern_lh_cat[0:nptn_ncat])

    double prob_const = 0.0;

    double *dad_partial_lh_base = dad_branch->partial_lh;
    UBYTE  *dad_scale_num_base  = dad_branch->scale_num;

    if (dad->isLeaf()) {
        // TIP-INTERNAL: dad is a leaf
        // Precompute partial_lh_node[state*block + i] = val[i] * tip_partial_lh[state*nstates + i]
        // (tip_partial_lh is already in eigenspace: U^{-1} * e_state)
        size_t tip_block = nstates * nmix;
        double *partial_lh_node = new double[(aln->STATE_UNKNOWN+1)*block];
        double *local_tip_plh = tip_partial_lh;

        if (dad == root) {
            for (c = 0; c < ncat; c++) {
                double *lh_node = partial_lh_node + c*nstates;
                model->getStateFrequency(lh_node);
                double prop = site_rate->getProp(c);
                for (i = 0; i < nstates; i++)
                    lh_node[i] *= prop;
            }
        } else {
            for (int state = 0; state <= aln->STATE_UNKNOWN; state++) {
                for (c = 0; c < ncat; c++) {
                    for (i = 0; i < nstates; i++) {
                        partial_lh_node[state*block + c*nstates + i] =
                            val[c*nstates + i] * local_tip_plh[state*tip_block + c*nstates + i];
                    }
                }
            }
        }

        for (int packet_id = 0; packet_id < num_packets; packet_id++) {
            size_t ptn_lower = limits[packet_id];
            size_t ptn_upper = limits[packet_id+1];
            for (vector<TraversalInfo>::iterator it = traversal_info.begin(); it != traversal_info.end(); it++)
                computePartialLikelihood(*it, ptn_lower, ptn_upper, packet_id);

            size_t nptn_range = ptn_upper - ptn_lower;
            int *states_dad = new int[nptn_range];
            bool dad_is_root = (dad == root);
            int dad_node_id = dad->id;
            for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
                size_t idx = ptn - ptn_lower;
                if (dad_is_root) states_dad[idx] = 0;
                else states_dad[idx] = (ptn < orig_nptn) ? (aln->at(ptn))[dad_node_id] : model_factory->unobserved_ptns[ptn-orig_nptn][dad_node_id];
            }

            // Reduction: simple dot product (no matrix-vector multiply)
            // lh_cat = Σ_i partial_lh_node[state][i] * dad_partial_lh[i]
            {
                size_t plh_node_size = (aln->STATE_UNKNOWN + 1) * block;
                size_t plh_offset = ptn_lower * block;
                size_t plh_count  = (ptn_upper - ptn_lower) * block;
                size_t scl_offset = ptn_lower;
                size_t scl_count  = ptn_upper - ptn_lower;

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
                    #pragma acc parallel loop gang reduction(+:tree_lh, prob_const)
                    for (size_t p = ptn_lower; p < ptn_upper; p++) {
                        size_t idx = p - ptn_lower;
                        int state_dad = states_dad[idx];
                        double lh_ptn = local_ptn_invar[p];

                        for (size_t cc = 0; cc < ncat; cc++) {
                            double lh_cat = 0.0;
                            for (size_t ii = 0; ii < nstates; ii++)
                                lh_cat += partial_lh_node[state_dad*block + cc*nstates + ii]
                                        * dad_partial_lh_base[p*block + cc*nstates + ii];
                            local_pattern_lh_cat[p*ncat + cc] = lh_cat;
                            lh_ptn += lh_cat;
                        }

                        if (p < orig_nptn) {
                            lh_ptn = log(fabs(lh_ptn)) + dad_scale_num_base[p] * LOG_SCALING_THRESHOLD;
                            local_pattern_lh[p] = lh_ptn;
                            tree_lh += lh_ptn * local_ptn_freq[p];
                        } else {
                            if (dad_scale_num_base[p] >= 1)
                                lh_ptn *= SCALING_THRESHOLD;
                            prob_const += lh_ptn;
                        }
                    }
                }
            }

            delete [] states_dad;
        }
        delete [] partial_lh_node;
    } else {

        // INTERNAL-INTERNAL: both dad and node are internal
        // Reduction: lh = Σ_i val[i] * node_plh[i] * dad_plh[i]
        double *node_partial_lh_base = node_branch->partial_lh;
        UBYTE  *node_scale_num_base  = node_branch->scale_num;

        for (int packet_id = 0; packet_id < num_packets; packet_id++) {
            size_t ptn_lower = limits[packet_id];
            size_t ptn_upper = limits[packet_id+1];
            for (vector<TraversalInfo>::iterator it = traversal_info.begin(); it != traversal_info.end(); it++)
                computePartialLikelihood(*it, ptn_lower, ptn_upper, packet_id);

            {
                size_t plh_offset = ptn_lower * block;
                size_t plh_count  = (ptn_upper - ptn_lower) * block;
                size_t scl_offset = ptn_lower;
                size_t scl_count  = ptn_upper - ptn_lower;

                #pragma acc data \
                    copyin(val[0:block]) \
                    present(dad_partial_lh_base[plh_offset:plh_count], \
                            node_partial_lh_base[plh_offset:plh_count], \
                            dad_scale_num_base[scl_offset:scl_count], \
                            node_scale_num_base[scl_offset:scl_count], \
                            local_ptn_invar[ptn_lower:scl_count], \
                            local_ptn_freq[ptn_lower:scl_count], \
                            local_pattern_lh[ptn_lower:scl_count], \
                            local_pattern_lh_cat[ptn_lower:scl_count])
                {
                    #pragma acc parallel loop gang reduction(+:tree_lh, prob_const)
                    for (size_t p = ptn_lower; p < ptn_upper; p++) {
                        double lh_ptn = local_ptn_invar[p];

                        for (size_t cc = 0; cc < ncat; cc++) {
                            double lh_cat = 0.0;
                            // Simple three-way element-wise product (no matrix multiply!)
                            for (size_t ii = 0; ii < nstates; ii++)
                                lh_cat += val[cc*nstates + ii]
                                        * node_partial_lh_base[p*block + cc*nstates + ii]
                                        * dad_partial_lh_base[p*block + cc*nstates + ii];
                            local_pattern_lh_cat[p*ncat + cc] = lh_cat;
                            lh_ptn += lh_cat;
                        }

                        if (p < orig_nptn) {
                            lh_ptn = log(fabs(lh_ptn)) + (dad_scale_num_base[p] + node_scale_num_base[p]) * LOG_SCALING_THRESHOLD;
                            local_pattern_lh[p] = lh_ptn;
                            tree_lh += lh_ptn * local_ptn_freq[p];
                        } else {
                            if (dad_scale_num_base[p] + node_scale_num_base[p] >= 1)
                                lh_ptn *= SCALING_THRESHOLD;
                            prob_const += lh_ptn;
                        }
                    }
                }
            }
        }
    }

    // GPU cleanup (same as non-rev Step 12)
    #pragma acc update self(local_pattern_lh[0:nptn])
    #pragma acc update self(local_central_plh[0:total_lh_entries])
    #pragma acc update self(local_central_scl[0:total_scale_entries])
    #pragma acc exit data \
        delete(local_central_plh[0:total_lh_entries], \
               local_central_scl[0:total_scale_entries], \
               local_ptn_freq[0:nptn], \
               local_ptn_invar[0:nptn], \
               local_pattern_lh[0:nptn], \
               local_pattern_lh_cat[0:nptn_ncat])

    if (std::isnan(tree_lh) || std::isinf(tree_lh)) {
        cout << "WARNING: Numerical underflow caused by alignment sites";
        i = aln->getNSite();
        size_t j;
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
            if (std::isnan(_pattern_lh[ptn]) || std::isinf(_pattern_lh[ptn]))
                _pattern_lh[ptn] = LOG_SCALING_THRESHOLD*4;
            tree_lh += _pattern_lh[ptn] * ptn_freq[ptn];
        }
    }

    if (orig_nptn < nptn) {
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

    delete [] val;
    return tree_lh;
}

// ==========================================================================
// Persistent GPU data cleanup
// Called from destructor and initializeAllPartialLh (before host realloc).
// ==========================================================================

void PhyloTree::freeOpenACCData() {
    if (!gpu_data_resident) return;

    if (verbose_mode >= VB_MED)
        cout << "OpenACC: Freeing persistent GPU data" << endl;

    // Use the saved pointers and sizes from the enter data copyin call.
    // These must match exactly — OpenACC tracks device data by host address.
    #pragma acc exit data \
        delete(gpu_central_plh_ptr[0:gpu_total_lh_entries], \
               gpu_central_scl_ptr[0:gpu_total_scale_entries], \
               gpu_ptn_freq_ptr[0:gpu_nptn], \
               gpu_ptn_invar_ptr[0:gpu_nptn], \
               gpu_pattern_lh_ptr[0:gpu_nptn], \
               gpu_pattern_lh_cat_ptr[0:gpu_nptn_ncat])

    gpu_data_resident = false;
    gpu_central_plh_ptr = nullptr;
    gpu_central_scl_ptr = nullptr;
    gpu_ptn_freq_ptr = nullptr;
    gpu_ptn_invar_ptr = nullptr;
    gpu_pattern_lh_ptr = nullptr;
    gpu_pattern_lh_cat_ptr = nullptr;
}

#endif // USE_OPENACC
