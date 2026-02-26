/***************************************************************************
 *   OpenACC GPU Likelihood Computation for IQ-TREE                       *
 *   Step 4: GPU-ready kernels with explicit indexing                     *
 *   Phase A: Refactored for OpenACC (no pragmas yet — CPU verifiable)    *
 *   Phase B: OpenACC pragmas + GPU data management (TODO)                *
 ***************************************************************************/

#ifdef USE_OPENACC

#include "phylokernel_openacc.h"
#include "phylotree.h"           // SCALING_THRESHOLD, PhyloTree, PhyloNeighbor, etc.
#include "phylokernelnew.h"      // computeTraversalInfo<Vec1d>, computeBounds<Vec1d>, computePartialInfo<Vec1d>
#include "vectorclass/vectorf64.h" // Vec1d (pure C++ scalar wrapper, no x86 intrinsics)
#include "model/modelsubst.h"    // computeTransMatrixEqualRate(), ModelSubst

#include <cmath>
#include <iostream>
#include <vector>

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

void PhyloTree::computeNonrevPartialLikelihoodOpenACC(TraversalInfo &info, size_t ptn_lower, size_t ptn_upper, int thread_id) {

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

        // A3: Precompute per-pattern tip states into flat arrays
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

        // A4: replace memset with plain loop
        for (ptn = ptn_lower; ptn < ptn_upper; ptn++)
            dad_scale_num[ptn] = 0;

        // Main kernel loop — GPU-ready structure
        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            size_t idx = ptn - ptn_lower;
            int state_left  = states_left[idx];
            int state_right = states_right[idx];
            for (i = 0; i < block; i++) {
                dad_partial_lh[ptn*block + i] = tip_lh_left[state_left*block + i] * tip_lh_right[state_right*block + i];
            }
        }

        delete [] states_left;
        delete [] states_right;

    } else if (left->node->isLeaf() && !right->node->isLeaf()) {

        /*--------------------- TIP-INTERNAL NODE case ------------------*/

        // A1: Extract raw pointers
        double *right_partial_lh = right->partial_lh;
        UBYTE  *right_scale_num  = right->scale_num;
        double *tip_lh_left = partial_lh_leaves;

        // A4: replace memcpy with plain loop for scale_num copy
        for (ptn = ptn_lower; ptn < ptn_upper; ptn++)
            dad_scale_num[ptn] = right_scale_num[ptn];

        // A3: Precompute per-pattern tip states
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

        // Main kernel loop — A2: explicit indexing
        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            size_t idx = ptn - ptn_lower;
            int state_left = states_left[idx];
            double lh_max = 0.0;

            for (c = 0; c < ncat; c++) {
                for (x = 0; x < nstates; x++) {
                    // Right child: dot product with transition matrix
                    double vright = 0.0;
                    for (i = 0; i < nstates; i++) {
                        vright += eright[c*nstatesqr + x*nstates + i] * right_partial_lh[ptn*block + c*nstates + i];
                    }
                    // Left child: precomputed tip lookup
                    double vleft_val = tip_lh_left[state_left*block + c*nstates + x];
                    double val = vleft_val * vright;
                    dad_partial_lh[ptn*block + c*nstates + x] = val;
                    if (val > lh_max) lh_max = val;
                }
            }

            // A4: replace memcpy with plain loop, A5: remove ASSERT
            if (lh_max == 0.0) {
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
        }

        delete [] states_left;

    } else {

        /*--------------------- INTERNAL-INTERNAL NODE case ------------------*/
        // This is the HOT PATH — matches PoC compositehadamard parallelism:
        //   gang over patterns, vector over states, sequential dot product

        // A1: Extract raw pointers
        double *left_partial_lh  = left->partial_lh;
        double *right_partial_lh = right->partial_lh;
        UBYTE  *left_scale_num   = left->scale_num;
        UBYTE  *right_scale_num  = right->scale_num;

        // Main kernel loop — A2: fully explicit indexing (no running pointers)
        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            dad_scale_num[ptn] = left_scale_num[ptn] + right_scale_num[ptn];
            double lh_max = 0.0;

            for (c = 0; c < ncat; c++) {
                for (x = 0; x < nstates; x++) {
                    double vleft = 0.0, vright = 0.0;
                    for (i = 0; i < nstates; i++) {
                        vleft  += eleft[c*nstatesqr + x*nstates + i]  * left_partial_lh[ptn*block + c*nstates + i];
                        vright += eright[c*nstatesqr + x*nstates + i] * right_partial_lh[ptn*block + c*nstates + i];
                    }
                    double val = vleft * vright;
                    dad_partial_lh[ptn*block + c*nstates + x] = val;
                    if (val > lh_max) lh_max = val;
                }
            }

            // A4: replace memcpy with plain loop, A5: remove ASSERT from kernel region
            if (lh_max == 0.0) {
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

double PhyloTree::computeNonrevLikelihoodBranchOpenACC(PhyloNeighbor *dad_branch, PhyloNode *dad, bool save_log_value) {

    // One-time verification message
    static bool openacc_kernel_printed = false;
    if (!openacc_kernel_printed) {
        cout << "OpenACC: Using GPU-ready (explicit indexing) likelihood kernel "
             << "(computeNonrevPartialLikelihoodOpenACC + "
             << "computeNonrevLikelihoodBranchOpenACC)" << endl;
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

    // Build traversal order and precompute P(t) / tip lookup tables
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
            for (int state = 0; state <= aln->STATE_UNKNOWN; state++) {
                double *lh_tip = tip_partial_lh + state*nstates;
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
#ifdef _OPENMP
#pragma omp parallel for reduction(+: tree_lh, prob_const) private(ptn, i, c) schedule(static,1) num_threads(num_threads)
#endif
        for (int packet_id = 0; packet_id < num_packets; packet_id++) {
            size_t ptn_lower = limits[packet_id];
            size_t ptn_upper = limits[packet_id+1];
            // first compute partial_lh
            for (vector<TraversalInfo>::iterator it = traversal_info.begin(); it != traversal_info.end(); it++)
                computePartialLikelihood(*it, ptn_lower, ptn_upper, packet_id);

            // A3: Precompute per-pattern dad states
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

            // A4: replace memset with plain loop
            for (ptn = ptn_lower; ptn < ptn_upper; ptn++)
                for (c = 0; c < ncat; c++)
                    _pattern_lh_cat[ptn*ncat + c] = 0.0;

            // Reduction loop — A2: explicit indexing
            for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
                double lh_ptn = ptn_invar[ptn];
                size_t idx = ptn - ptn_lower;
                int state_dad = states_dad[idx];
                for (c = 0; c < ncat; c++) {
                    double lh_cat = 0.0;
                    for (i = 0; i < nstates; i++) {
                        lh_cat += partial_lh_node[state_dad*block + c*nstates + i] * dad_partial_lh_base[ptn*block + c*nstates + i];
                    }
                    _pattern_lh_cat[ptn*ncat + c] = lh_cat;
                    lh_ptn += lh_cat;
                }
                // A5: ASSERT removed from kernel region
                if (ptn < orig_nptn) {
                    lh_ptn = log(fabs(lh_ptn)) + dad_scale_num_base[ptn] * LOG_SCALING_THRESHOLD;
                    _pattern_lh[ptn] = lh_ptn;
                    tree_lh += lh_ptn * ptn_freq[ptn];
                } else {
                    if (dad_scale_num_base[ptn] >= 1)
                        lh_ptn *= SCALING_THRESHOLD;
                    prob_const += lh_ptn;
                }
            } // FOR ptn

            delete [] states_dad;
        } // FOR packet_id
        delete [] partial_lh_node;
    } else {

        // both dad and node are internal nodes
        // A1: Extract raw pointers
        double *node_partial_lh_base = node_branch->partial_lh;
        UBYTE  *node_scale_num_base  = node_branch->scale_num;

#ifdef _OPENMP
#pragma omp parallel for reduction(+: tree_lh, prob_const) private(ptn, i, c, x) schedule(static,1) num_threads(num_threads)
#endif
        for (int packet_id = 0; packet_id < num_packets; packet_id++) {
            size_t ptn_lower = limits[packet_id];
            size_t ptn_upper = limits[packet_id+1];
            // first compute partial_lh
            for (vector<TraversalInfo>::iterator it = traversal_info.begin(); it != traversal_info.end(); it++)
                computePartialLikelihood(*it, ptn_lower, ptn_upper, packet_id);

            // A4: replace memset with plain loop
            for (ptn = ptn_lower; ptn < ptn_upper; ptn++)
                for (c = 0; c < ncat; c++)
                    _pattern_lh_cat[ptn*ncat + c] = 0.0;

            // Reduction loop — A2: explicit indexing
            for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
                double lh_ptn = ptn_invar[ptn];
                for (c = 0; c < ncat; c++) {
                    double lh_cat = 0.0;
                    for (i = 0; i < nstates; i++) {
                        double lh_state = 0.0;
                        for (x = 0; x < nstates; x++)
                            lh_state += trans_mat[c*nstatesqr + i*nstates + x] * node_partial_lh_base[ptn*block + c*nstates + x];
                        lh_cat += dad_partial_lh_base[ptn*block + c*nstates + i] * lh_state;
                    }
                    _pattern_lh_cat[ptn*ncat + c] = lh_cat;
                    lh_ptn += lh_cat;
                }

                // A5: ASSERT removed from kernel region
                if (ptn < orig_nptn) {
                    lh_ptn = log(fabs(lh_ptn)) + (dad_scale_num_base[ptn] + node_scale_num_base[ptn])*LOG_SCALING_THRESHOLD;
                    _pattern_lh[ptn] = lh_ptn;
                    tree_lh += lh_ptn * ptn_freq[ptn];
                } else {
                    if (dad_scale_num_base[ptn] + node_scale_num_base[ptn] >= 1)
                        lh_ptn *= SCALING_THRESHOLD;
                    prob_const += lh_ptn;
                }
            } // FOR ptn
        } // FOR packet_id
    }

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

#endif // USE_OPENACC
