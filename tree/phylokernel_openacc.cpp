/***************************************************************************
 *   OpenACC GPU Likelihood Computation for IQ-TREE                       *
 *   Scalar (plain C) likelihood kernels for non-reversible models        *
 *   No SIMD, no Eigen — ready for future GPU offloading via OpenACC      *
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
// Step 3: Scalar (plain C) Partial Likelihood Kernel
// Adapted from phylokernelnonrev.cpp (computeNonrevPartialLikelihood)
// This is the non-SIMD version using plain C loops — ready for future
// OpenACC GPU offloading. No Eigen, no x86 intrinsics.
// ==========================================================================

void PhyloTree::computeNonrevPartialLikelihoodOpenACC(TraversalInfo &info, size_t ptn_lower, size_t ptn_upper, int thread_id) {

    PhyloNeighbor *dad_branch = info.dad_branch;
    PhyloNode *dad = info.dad;

    // don't recompute the likelihood
    ASSERT(dad);
    PhyloNode *node = (PhyloNode*)(dad_branch->node);

    ASSERT(dad_branch->direction != UNDEFINED_DIRECTION);

    size_t nstates = aln->num_states;

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
    PhyloNeighbor *left = NULL, *right = NULL; // left & right are two neighbors leading to 2 subtrees
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

    if (node->degree() > 3) {

        /*--------------------- multifurcating node ------------------*/

        // now for-loop computing partial_lh over all site-patterns
        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            double *partial_lh_all = dad_branch->partial_lh + ptn*block;
            for (i = 0; i < block; i++)
                partial_lh_all[i] = 1.0;
            dad_branch->scale_num[ptn] = 0;

            double *partial_lh_leaf = partial_lh_leaves;
            double *echild = echildren;

            FOR_NEIGHBOR_IT(node, dad, it) {
                PhyloNeighbor *child = (PhyloNeighbor*)*it;
                if (child->node->isLeaf()) {
                    // external node
                    int state_child;
                    if (child->node == root)
                        state_child = 0;
                    else state_child = (ptn < orig_ntn) ? (aln->at(ptn))[child->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn][child->node->id];
                    double *child_lh = partial_lh_leaf + state_child*block;
                    for (c = 0; c < block; c++) {
                        // compute real partial likelihood vector
                        partial_lh_all[c] *= child_lh[c];
                    }
                    partial_lh_leaf += (aln->STATE_UNKNOWN+1)*block;
                } else {
                    // internal node
                    double *partial_lh = partial_lh_all;
                    double *partial_lh_child = child->partial_lh + ptn*block;
                    dad_branch->scale_num[ptn] += child->scale_num[ptn];

                    double *echild_ptr = echild;
                    for (c = 0; c < ncat; c++) {
                        // compute real partial likelihood vector
                        for (x = 0; x < nstates; x++) {
                            double vchild = 0.0;
                            for (i = 0; i < nstates; i++) {
                                vchild += echild_ptr[i] * partial_lh_child[i];
                            }
                            echild_ptr += nstates;
                            partial_lh[x] *= vchild;
                        }
                        partial_lh += nstates;
                        partial_lh_child += nstates;
                    }
                } // if
                echild += block*nstates;
            } // FOR_NEIGHBOR


            double lh_max = partial_lh_all[0];
            for (i = 1; i < block; i++)
                lh_max = max(lh_max, partial_lh_all[i]);

            ASSERT(lh_max > 0.0);
            // check if one should scale partial likelihoods
            if (lh_max == 0.0) {
                // for very shitty data
                for (c = 0; c < ncat; c++)
                    memcpy(&partial_lh_all[c*nstates], &tip_partial_lh[aln->STATE_UNKNOWN*nstates], nstates*sizeof(double));
                dad_branch->scale_num[ptn] += 4;
            } else if (lh_max < SCALING_THRESHOLD) {
                // now do the likelihood scaling
                for (i = 0; i < block; i++) {
                    partial_lh_all[i] = ldexp(partial_lh_all[i], SCALING_THRESHOLD_EXP);
                }
                dad_branch->scale_num[ptn] += 1;
            }

        } // for ptn

        // end multifurcating treatment
    } else if (left->node->isLeaf() && right->node->isLeaf()) {

        /*--------------------- TIP-TIP (cherry) case ------------------*/

        double *partial_lh_left = partial_lh_leaves;
        double *partial_lh_right = partial_lh_leaves + (aln->STATE_UNKNOWN+1)*block;

        if (right->node == root) {
            // swap so that left node is the root
            PhyloNeighbor *tmp = left;
            left = right;
            right = tmp;
            double *etmp = eleft;
            eleft = eright;
            eright = etmp;
            etmp = partial_lh_left;
            partial_lh_left = partial_lh_right;
            partial_lh_right = etmp;
        }

        // scale number must be ZERO
        memset(dad_branch->scale_num + ptn_lower, 0, (ptn_upper-ptn_lower) * sizeof(UBYTE));
        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            double *partial_lh = dad_branch->partial_lh + ptn*block;
            int state_left;
            if (left->node == root)
                state_left = 0;
            else
                state_left = (ptn < orig_ntn) ? (aln->at(ptn))[left->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn][left->node->id];
            int state_right = (ptn < orig_ntn) ? (aln->at(ptn))[right->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn][right->node->id];
            double *vleft = partial_lh_left + (state_left*block);
            double *vright = partial_lh_right + (state_right*block);
            for (i = 0; i < block; i++)
                partial_lh[i] = vleft[i] * vright[i];
        }
    } else if (left->node->isLeaf() && !right->node->isLeaf()) {

        /*--------------------- TIP-INTERNAL NODE case ------------------*/

        // only take scale_num from the right subtree
        memcpy(dad_branch->scale_num + ptn_lower, right->scale_num + ptn_lower, (ptn_upper-ptn_lower) * sizeof(UBYTE));

        double *partial_lh_left = partial_lh_leaves;

        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            double *partial_lh = dad_branch->partial_lh + ptn*block;
            double *partial_lh_right = right->partial_lh + ptn*block;
            int state_left;
            if (left->node == root)
                state_left = 0;
            else
                state_left = (ptn < orig_ntn) ? (aln->at(ptn))[left->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn][left->node->id];
            double *vleft = partial_lh_left + state_left*block;
            double lh_max = 0.0;

            double *eright_ptr = eright;
            for (c = 0; c < ncat; c++) {
                // compute real partial likelihood vector
                for (x = 0; x < nstates; x++) {
                    double vright = 0.0;
                    for (i = 0; i < nstates; i++) {
                        vright += eright_ptr[i] * partial_lh_right[i];
                    }
                    eright_ptr += nstates;
                    lh_max = max(lh_max, (partial_lh[c*nstates+x] = vleft[x]*vright));
                }
                vleft += nstates;
                partial_lh_right += nstates;
            }
            ASSERT(lh_max > 0.0);
            // check if one should scale partial likelihoods
            if (lh_max == 0.0) {
                // for very shitty data
                for (c = 0; c < ncat; c++)
                    memcpy(&partial_lh[c*nstates], &tip_partial_lh[aln->STATE_UNKNOWN*nstates], nstates*sizeof(double));
                dad_branch->scale_num[ptn] += 4;
            } else if (lh_max < SCALING_THRESHOLD) {
                // now do the likelihood scaling
                for (i = 0; i < block; i++) {
                    partial_lh[i] = ldexp(partial_lh[i], SCALING_THRESHOLD_EXP);
                }
                dad_branch->scale_num[ptn] += 1;
            }
        }

    } else {

        /*--------------------- INTERNAL-INTERNAL NODE case ------------------*/

        for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
            double *partial_lh = dad_branch->partial_lh + ptn*block;
            double *partial_lh_left = left->partial_lh + ptn*block;
            double *partial_lh_right = right->partial_lh + ptn*block;
            double lh_max = 0.0;
            dad_branch->scale_num[ptn] = left->scale_num[ptn] + right->scale_num[ptn];

            double *eleft_ptr = eleft;
            double *eright_ptr = eright;

            for (c = 0; c < ncat; c++) {
                // compute real partial likelihood vector
                for (x = 0; x < nstates; x++) {
                    double vleft = 0.0, vright = 0.0;
                    for (i = 0; i < nstates; i++) {
                        vleft += eleft_ptr[i] * partial_lh_left[i];
                        vright += eright_ptr[i] * partial_lh_right[i];
                    }
                    eleft_ptr += nstates;
                    eright_ptr += nstates;
                    lh_max=max(lh_max, (partial_lh[c*nstates+x] = vleft*vright));
                }
                partial_lh_left += nstates;
                partial_lh_right += nstates;
            }

            ASSERT(lh_max > 0.0);
            // check if one should scale partial likelihoods
            if (lh_max == 0.0) {
                // for very shitty data
                for (c = 0; c < ncat; c++)
                    memcpy(&partial_lh[c*nstates], &tip_partial_lh[aln->STATE_UNKNOWN*nstates], nstates*sizeof(double));
                dad_branch->scale_num[ptn] += 4;
            } else if (lh_max < SCALING_THRESHOLD) {
                // now do the likelihood scaling
                for (i = 0; i < block; i++) {
                    partial_lh[i] = ldexp(partial_lh[i], SCALING_THRESHOLD_EXP);
                }
                dad_branch->scale_num[ptn] += 1;
            }

        }

    }
}

// ==========================================================================
// Step 3: Scalar (plain C) Branch Likelihood Kernel
// Adapted from phylokernelnonrev.cpp (computeNonrevLikelihoodBranch)
// Fixed: signature matches ComputeLikelihoodBranchType (added save_log_value)
// Fixed: computeBounds call updated to 4-arg version
// Fixed: loop uses num_packets/packet_id (matching current SIMD convention)
// ==========================================================================

double PhyloTree::computeNonrevLikelihoodBranchOpenACC(PhyloNeighbor *dad_branch, PhyloNode *dad, bool save_log_value) {

    // One-time verification message
    static bool openacc_kernel_printed = false;
    if (!openacc_kernel_printed) {
        cout << "OpenACC: Using scalar (plain C) likelihood kernel "
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
    // (3rd arg = false: don't compute partial likelihoods here, we do it below)
    computeTraversalInfo<Vec1d>(node, dad, false);

    double tree_lh = 0.0;
    size_t nstates = aln->num_states;
    size_t nstatesqr = nstates*nstates;
    size_t ncat = site_rate->getNRate();

    size_t block = ncat * nstates;
    size_t ptn; // for big data size > 4GB memory required
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
            // precompute information from one tip
            // (iterate over all states, matching modern phylokernelnonrev.h)
            for (int state = 0; state <= aln->STATE_UNKNOWN; state++) {
                double *lh_node = partial_lh_node + state*block;
                double *lh_tip = tip_partial_lh + state*nstates;
                double *trans_mat_tmp = trans_mat;
                for (c = 0; c < ncat; c++) {
                    for (i = 0; i < nstates; i++) {
                        lh_node[i] = 0.0;
                        for (x = 0; x < nstates; x++)
                            lh_node[i] += trans_mat_tmp[x] * lh_tip[x];
                        trans_mat_tmp += nstates;
                    }
                    lh_node += nstates;
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

            // reset memory for _pattern_lh_cat
            memset(_pattern_lh_cat+ptn_lower*ncat, 0, (ptn_upper-ptn_lower)*ncat*sizeof(double));

            for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
                double lh_ptn = ptn_invar[ptn];
                double *lh_cat = _pattern_lh_cat + ptn*ncat;
                double *partial_lh_dad = dad_branch->partial_lh + ptn*block;
                int state_dad;
                if (dad == root)
                    state_dad = 0;
                else
                    state_dad = (ptn < orig_nptn) ? (aln->at(ptn))[dad->id] : model_factory->unobserved_ptns[ptn-orig_nptn][dad->id];
                double *lh_node = partial_lh_node + state_dad*block;
                for (c = 0; c < ncat; c++) {
                    for (i = 0; i < nstates; i++) {
                        lh_cat[c] += lh_node[i] * partial_lh_dad[i];
                    }
                    lh_node += nstates;
                    partial_lh_dad += nstates;
                    lh_ptn += lh_cat[c];
                }
                ASSERT(lh_ptn > 0.0);
                if (ptn < orig_nptn) {
                    lh_ptn = log(fabs(lh_ptn)) + dad_branch->scale_num[ptn] * LOG_SCALING_THRESHOLD;
                    _pattern_lh[ptn] = lh_ptn;
                    tree_lh += lh_ptn * ptn_freq[ptn];
                } else {
                    // bugfix 2016-01-21, prob_const can be rescaled
                    if (dad_branch->scale_num[ptn] >= 1)
                        lh_ptn *= SCALING_THRESHOLD;
                    prob_const += lh_ptn;
                }
            } // FOR ptn
        } // FOR packet_id
        delete [] partial_lh_node;
    } else {

        // both dad and node are internal nodes
#ifdef _OPENMP
#pragma omp parallel for reduction(+: tree_lh, prob_const) private(ptn, i, c, x) schedule(static,1) num_threads(num_threads)
#endif
        for (int packet_id = 0; packet_id < num_packets; packet_id++) {
            size_t ptn_lower = limits[packet_id];
            size_t ptn_upper = limits[packet_id+1];
            // first compute partial_lh
            for (vector<TraversalInfo>::iterator it = traversal_info.begin(); it != traversal_info.end(); it++)
                computePartialLikelihood(*it, ptn_lower, ptn_upper, packet_id);

            // reset memory for _pattern_lh_cat
            memset(_pattern_lh_cat+ptn_lower*ncat, 0, (ptn_upper-ptn_lower)*ncat*sizeof(double));

            for (ptn = ptn_lower; ptn < ptn_upper; ptn++) {
                double lh_ptn = ptn_invar[ptn];
                double *lh_cat = _pattern_lh_cat + ptn*ncat;
                double *partial_lh_dad = dad_branch->partial_lh + ptn*block;
                double *partial_lh_node = node_branch->partial_lh + ptn*block;
                double *trans_mat_tmp = trans_mat;
                for (c = 0; c < ncat; c++) {
                    for (i = 0; i < nstates; i++) {
                        double lh_state = 0.0;
                        for (x = 0; x < nstates; x++)
                            lh_state += trans_mat_tmp[x] * partial_lh_node[x];
                        *lh_cat += partial_lh_dad[i] * lh_state;
                        trans_mat_tmp += nstates;
                    }
                    lh_ptn += *lh_cat;
                    partial_lh_node += nstates;
                    partial_lh_dad += nstates;
                    lh_cat++;
                }

                ASSERT(lh_ptn > 0.0);
                if (ptn < orig_nptn) {
                    lh_ptn = log(fabs(lh_ptn)) + (dad_branch->scale_num[ptn] + node_branch->scale_num[ptn])*LOG_SCALING_THRESHOLD;
                    _pattern_lh[ptn] = lh_ptn;
                    tree_lh += lh_ptn * ptn_freq[ptn];
                } else {
                    // bugfix 2016-01-21, prob_const can be rescaled
                    if (dad_branch->scale_num[ptn] + node_branch->scale_num[ptn] >= 1)
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
