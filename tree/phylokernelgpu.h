//
// Created by Hashara Kumarasinghe on 30/6/2025.
//

#ifndef IQTREE_PHYLOKERNELGPU_H
#define IQTREE_PHYLOKERNELGPU_H

#include "phylotree.h"


/**********************************************************************
 *
 *   Likelihood function for GPU
 **********************************************************************/

//void PhyloTree::computePartialLikelihoodGPU(TraversalInfo &info
//        , size_t ptn_lower, size_t ptn_upper, int packet_id)
//{
//
//    // HK: tmp assume SITE_MODEL = false and SAFE_NUMERIC = true
//    int parallel_threads = 1; // to replace the VectorClass::size()
//
//    PhyloNeighbor *dad_branch = info.dad_branch;
//    PhyloNode *dad = info.dad;
//    // don't recompute the likelihood
//    ASSERT(dad);
////    if (dad_branch->partial_lh_computed & 1)
////        return;
////    dad_branch->partial_lh_computed |= 1;
//    PhyloNode *node = (PhyloNode*)(dad_branch->node);
//
//    size_t nstates = aln->num_states;
//    const size_t states_square = nstates*nstates;
//    size_t orig_nptn = aln->size();
//    size_t max_orig_nptn = roundUpToMultiple(orig_nptn, parallel_threads);
//    size_t nptn = max_orig_nptn+model_factory->unobserved_ptns.size();
//
//    if (node->isLeaf()) {
//        return;
//    }
//
//    size_t ncat = site_rate->getNRate();
//    size_t ncat_mix = (model_factory->fused_mix_rate) ? ncat : ncat*model->getNMixtures();
//    size_t mix_addr_nstates_malign[ncat_mix], mix_addr_malign[ncat_mix];
//    size_t denom = (model_factory->fused_mix_rate) ? 1 : ncat;
//    for (size_t c = 0; c < ncat_mix; c++) {
//        size_t m = c/denom;
//        mix_addr_nstates_malign[c] = m * get_safe_upper_limit(nstates);
//        mix_addr_malign[c] = mix_addr_nstates_malign[c]*nstates;
//    }
//    size_t block = nstates * ncat_mix;
//    size_t tip_mem_size = max_orig_nptn * nstates;
//    size_t scale_size = (ptn_upper-ptn_lower) * ncat_mix; // HK: tmp remove SAFE_NUMERIC ?
//
//    double *evec = model->getEigenvectors();
//    double *inv_evec = model->getInverseEigenvectors();
//    ASSERT(inv_evec && evec);
//    double *eval = model->getEigenvalues();
//    size_t num_leaves = 0;
//
//    // internal node
//    PhyloNeighbor *left = NULL, *right = NULL; // left & right are two neighbors leading to 2 subtrees
//    FOR_NEIGHBOR_IT(node, dad, it) {
//            PhyloNeighbor *nei = (PhyloNeighbor*)(*it);
//            // make sure that the partial_lh of children are different!
//            ASSERT(dad_branch->partial_lh != nei->partial_lh);
//            if (!left) left = nei; else right = nei;
//            if (nei->node->isLeaf())
//                num_leaves++;
//        }
//
//    // precomputed buffer to save times
//    size_t thread_buf_size        = (2*block+nstates)*parallel_threads;
//    double *buffer_partial_lh_ptr = buffer_partial_lh + (getBufferPartialLhSize() - thread_buf_size*num_packets);
//    double *echildren = NULL;
//    double *partial_lh_leaves = NULL;
//
//    // pre-compute scaled branch length per category
//    double len_children[ncat*(node->degree()-1)]; // +1 in case num_leaves = 0
//    double *len_left = NULL, *len_right = NULL;
//
//    // HK: tmp remove SITE_MODEL
//
//    if (Params::getInstance().buffer_mem_save) {
//        echildren = aligned_alloc<double>(get_safe_upper_limit(block*nstates*(node->degree()-1)));
//        if (num_leaves > 0)
//            partial_lh_leaves = aligned_alloc<double>(get_safe_upper_limit((aln->STATE_UNKNOWN+1)*block*num_leaves));
//        double *buffer_tmp = aligned_alloc<double>(nstates);
//
//        computePartialInfo<VectorClass>(info, (VectorClass*)buffer_tmp, echildren, partial_lh_leaves);
//
//        aligned_free(buffer_tmp);
//    } else {
//        echildren = info.echildren;
//        partial_lh_leaves = info.partial_lh_leaves;
//    }
//
//
//    double *eleft = echildren, *eright = echildren + block*nstates;
//
//    if (!left->node->isLeaf() && right->node->isLeaf()) {
//        PhyloNeighbor *tmp = left;
//        left = right;
//        right = tmp;
//        double *etmp = eleft;
//        eleft = eright;
//        eright = etmp;
//        etmp = len_left;
//        len_left = len_right;
//        len_right = etmp;
//    }
//
//    // HK: tmp remove multifurcating
//    if (left->node->isLeaf() && right->node->isLeaf()) {
//
//        /*--------------------- TIP-TIP (cherry) case ------------------*/
//
//        double *partial_lh_left = partial_lh_leaves; // HK: tmp remove SITE_MODEL
//        double *partial_lh_right = partial_lh_leaves + (aln->STATE_UNKNOWN+1)*block; // HK: tmp remove SITE_MODEL
//
//        // scale number must be ZERO
//        memset(dad_branch->scale_num + (ptn_lower*ncat_mix), 0, scale_size * sizeof(UBYTE)); // HK: tmp remove SAFE_NUMERIC
//        double *vec_left = buffer_partial_lh_ptr + thread_buf_size * packet_id;
//
//        double *vec_right =  &vec_left[block*parallel_threads]; // HK: tmp remove SITE_MODEL
//        VectorClass *partial_lh_tmp = (VectorClass*)vec_right+block; // HK: tmp remove SITE_MODEL
//
//        auto leftStateRow  = this->getConvertedSequenceByNumber(left->node->id);
//        auto rightStateRow = this->getConvertedSequenceByNumber(right->node->id);
//        auto unknown = aln->STATE_UNKNOWN;
//
//        for (size_t ptn = ptn_lower; ptn < ptn_upper; ptn+=parallel_threads) {
//            VectorClass *partial_lh = (VectorClass*)(dad_branch->partial_lh + ptn*block);
//
//            // HK: tmp remove SITE_MODEL
//            VectorClass *vleft  = (VectorClass*)vec_left;
//            VectorClass *vright = (VectorClass*)vec_right;
//            // load data for tip
//            for (size_t x = 0; x < parallel_threads; x++) {
//                int leftState;
//                int rightState;
//                if (ptn+x < orig_nptn) {
//                    if (leftStateRow!=nullptr) {
//                        leftState = leftStateRow[ptn+x];
//                    } else {
//                        leftState = (aln->at(ptn+x))[left->node->id];
//                    }
//                    if (rightStateRow!=nullptr) {
//                        rightState =  rightStateRow[ptn+x];
//                    } else {
//                        rightState = (aln->at(ptn+x))[right->node->id];
//                    }
//                } else if (ptn+x < max_orig_nptn) {
//                    leftState = unknown;
//                    rightState = unknown;
//                } else if (ptn+x < nptn) {
//                    leftState  = model_factory->unobserved_ptns[ptn+x-max_orig_nptn][left->node->id];
//                    rightState = model_factory->unobserved_ptns[ptn+x-max_orig_nptn][right->node->id];
//                } else {
//                    leftState  = unknown;
//                    rightState = unknown;
//                }
//                double* tip_left  = partial_lh_left  + block*leftState;
//                double* tip_right = partial_lh_right + block*rightState;
//                double* this_vec_left = vec_left+x;
//                double* this_vec_right = vec_right+x;
//                for (size_t i = 0; i < block; i++) {
//                    *this_vec_left = tip_left[i];
//                    *this_vec_right = tip_right[i];
//                    this_vec_left += parallel_threads;
//                    this_vec_right += parallel_threads;
//                }
//            }
//
//
//            for (size_t c = 0; c < ncat_mix; c++) {
//                double *inv_evec_ptr = inv_evec + mix_addr_malign[c];
//                // compute real partial likelihood vector
//                for (size_t x = 0; x < nstates; x++) {
//                    partial_lh_tmp[x] = vleft[x] * vright[x];
//                }
//
//                // compute dot-product with inv_eigenvector
//#ifdef KERNEL_FIX_STATES
//                productVecMat<VectorClass, double, nstates, FMA>(partial_lh_tmp, inv_evec_ptr, partial_lh);
//#else
//                productVecMat<VectorClass, double, FMA> (partial_lh_tmp, inv_evec_ptr, partial_lh, nstates);
//#endif
//
//                // increase pointer
//                vleft += nstates;
//                vright += nstates;
//                partial_lh += nstates;
//            } // FOR category
//
//
//        } // FOR LOOP
//
//
//    } else if (left->node->isLeaf() && !right->node->isLeaf()) {
//
//        /*--------------------- TIP-INTERNAL NODE case ------------------*/
//
//        // only take scale_num from the right subtree
//        memcpy(
//                dad_branch->scale_num + (ptn_lower*ncat_mix ), // HK: tmp remove SAFE_NUMERIC
//                right->scale_num + ( ptn_lower*ncat_mix ), // HK: tmp remove SAFE_NUMERIC
//                scale_size * sizeof(UBYTE));
//
//        double *partial_lh_left = partial_lh_leaves; // HK: tmp remove SITE_MODEL
//
//
//        double *vec_left = buffer_partial_lh_ptr + thread_buf_size * packet_id;
//        VectorClass *partial_lh_tmp = (VectorClass*)vec_left+block; // HK: tmp remove SITE_MODEL
//
//        auto leftStateRow = this->getConvertedSequenceByNumber(left->node->id);
//        auto unknown = aln->STATE_UNKNOWN;
//
//        for (size_t ptn = ptn_lower; ptn < ptn_upper; ptn+=parallel_threads) {
//            VectorClass *partial_lh = (VectorClass*)(dad_branch->partial_lh + ptn*block);
//            VectorClass *partial_lh_right = (VectorClass*)(right->partial_lh + ptn*block);
//            VectorClass lh_max = 0.0;
//
//            // HK: tmp remove SITE_MODEL
//            VectorClass *vleft = (VectorClass*)vec_left;
//            // load data for tip
//            for (size_t x = 0; x < parallel_threads; x++) {
//                int state;
//                if (ptn+x < orig_nptn) {
//                    if (leftStateRow!=nullptr) {
//                        state =  leftStateRow[ptn+x];
//                    } else {
//                        state = (aln->at(ptn+x))[left->node->id];
//                    }
//                } else if (ptn+x < max_orig_nptn) {
//                    state = unknown;
//                } else if (ptn+x < nptn) {
//                    state = model_factory->unobserved_ptns[ptn+x-max_orig_nptn][left->node->id];
//                } else {
//                    state = unknown;
//                }
//                double *tip = partial_lh_left + block*state;
//                double *this_vec_left = vec_left+x;
//                for (size_t i = 0; i < block; i++) {
//                    *this_vec_left = tip[i];
//                    this_vec_left += parallel_threads;
//                }
//            }
//
//            double *eright_ptr = eright;
//            for (size_t c = 0; c < ncat_mix; c++) {
//                lh_max = 0.0; // HK: tmp remove SAFE_NUMERIC
//                double *inv_evec_ptr = inv_evec + mix_addr_malign[c];
//                // compute real partial likelihood vector
//                for (size_t x = 0; x < nstates; x++) {
//                    VectorClass vright;
//
//                    dotProductVec<VectorClass, double, FMA>(eright_ptr, partial_lh_right, vright, nstates);
//
//                    eright_ptr += nstates;
//                    partial_lh_tmp[x] = vleft[x] * (vright);
//                }
//
//                // compute dot-product with inv_eigenvector
//
//                productVecMat<VectorClass, double, FMA> (partial_lh_tmp, inv_evec_ptr, partial_lh, lh_max, nstates);
//
//                // check if one should scale partial likelihoods
//                // HK: tmp remove SAFE_NUMERIC
//                auto underflown = ((lh_max < SCALING_THRESHOLD) & (VectorClass().load_a(&ptn_invar[ptn]) == 0.0));
//                if (horizontal_or(underflown)) { // at least one site has numerical underflown
//                    for (size_t x = 0; x < parallel_threads; x++) {
//                        if (underflown[x]) {
//                            // BQM 2016-05-03: only scale for non-constant sites
//                            // now do the likelihood scaling
//                            double *partial_lh = dad_branch->partial_lh + (ptn*block + c*nstates*parallel_threads + x);
//                            for (size_t i = 0; i < nstates; i++)
//                                partial_lh[i*parallel_threads] = ldexp(partial_lh[i*parallel_threads], SCALING_THRESHOLD_EXP);
//                            dad_branch->scale_num[(ptn+x)*ncat_mix+c] += 1;
//                        }
//                    }
//                }
//
//                vleft += nstates;
//                partial_lh_right += nstates;
//                partial_lh += nstates;
//            } // FOR category
//
//
//            // HK: tmp remove SAFE_NUMERIC
//
//        } // big for loop over ptn
//
//    } else {
//
//        /*--------------------- INTERNAL-INTERNAL NODE case ------------------*/
//
//        VectorClass *partial_lh_tmp
//                = (VectorClass*)(buffer_partial_lh_ptr + thread_buf_size * packet_id);
//        for (size_t ptn = ptn_lower; ptn < ptn_upper; ptn+=parallel_threads) {
//            VectorClass *partial_lh = (VectorClass*)(dad_branch->partial_lh + ptn*block);
//            VectorClass *partial_lh_left = (VectorClass*)(left->partial_lh + ptn*block);
//            VectorClass *partial_lh_right = (VectorClass*)(right->partial_lh + ptn*block);
//            VectorClass lh_max = 0.0;
//            UBYTE *scale_dad, *scale_left, *scale_right;
//
//            // HK: tmp remove SAFE_NUMERIC
//
//            size_t addr = ptn*ncat_mix;
//            scale_dad   = dad_branch->scale_num + addr;
//            scale_left  = left->scale_num + addr;
//            scale_right = right->scale_num + addr;
//
//
//
//            double *eleft_ptr = eleft;
//            double *eright_ptr = eright;
//            VectorClass *expleft, *expright, *eval_ptr, *evec_ptr, *inv_evec_ptr;
//
//            // HK: tmp remove SITE_MODEL
//
//            for (size_t c = 0; c < ncat_mix; c++) {
//                // HK: tmp remove SAFE_NUMERIC
//                lh_max = 0.0;
//                for (size_t x = 0; x < parallel_threads; x++)
//                    scale_dad[x*ncat_mix] = scale_left[x*ncat_mix] + scale_right[x*ncat_mix];
//
//                // HK: tmp remove SITE_MODEL
//                // normal model
//                double *inv_evec_ptr = inv_evec + mix_addr_malign[c];
//                // compute real partial likelihood vector
//                for (size_t x = 0; x < nstates; x++) {
//#ifdef KERNEL_FIX_STATES
//                    dotProductDualVec<VectorClass, double, nstates, FMA>(eleft_ptr, partial_lh_left, eright_ptr, partial_lh_right, partial_lh_tmp[x]);
//#else
//                    dotProductDualVec<VectorClass, double, FMA>(eleft_ptr, partial_lh_left, eright_ptr, partial_lh_right, partial_lh_tmp[x], nstates);
//#endif
//                    eleft_ptr += nstates;
//                    eright_ptr += nstates;
//                }
//
//                // compute dot-product with inv_eigenvector
//#ifdef KERNEL_FIX_STATES
//                productVecMat<VectorClass, double, nstates, FMA>(partial_lh_tmp, inv_evec_ptr, partial_lh, lh_max);
//#else
//                productVecMat<VectorClass, double, FMA> (partial_lh_tmp, inv_evec_ptr, partial_lh, lh_max, nstates);
//#endif
//
//
//                // check if one should scale partial likelihoods
//                // HK: tmp remove SAFE_NUMERIC
//                auto underflown = ((lh_max < SCALING_THRESHOLD) & (VectorClass().load_a(&ptn_invar[ptn]) == 0.0));
//                if (horizontal_or(underflown))
//                    for (size_t x = 0; x < parallel_threads; x++)
//                        if (underflown[x]) {
//                            // BQM 2016-05-03: only scale for non-constant sites
//                            // now do the likelihood scaling
//                            double *partial_lh = dad_branch->partial_lh + (ptn*block + c*nstates*parallel_threads + x);
//                            for (size_t i = 0; i < nstates; i++)
//                                partial_lh[i*parallel_threads] = ldexp(partial_lh[i*parallel_threads], SCALING_THRESHOLD_EXP);
//                            scale_dad[x*ncat_mix] += 1;
//                        }
//                scale_dad++;
//                scale_left++;
//                scale_right++;
//
//                partial_lh_left += nstates;
//                partial_lh_right += nstates;
//                partial_lh += nstates;
//            }
//
//            // HK: tmp remove SAFE_NUMERIC
//        } // big for loop over ptn
//    }
//
//    if (Params::getInstance().buffer_mem_save) {
//        aligned_free(partial_lh_leaves);
//        aligned_free(echildren);
//    }
//
//}


// -23646.018

void PhyloTree::computePartialInfoGPU(TraversalInfo &info, double* buffer, double *echildren, double *partial_lh_leaves) {

    int parallel_threads = 1; // to replace the VectorClass::size()
    size_t nstates = aln->num_states;

    size_t c, i, x;

    // HK: tmp remove Ncat and Ncat_mix because in the Simplest case (-m GTR) they are 1
    size_t block = nstates;
    size_t tip_block = nstates;

    double *evec = model->getEigenvectors();
    double *eval = model->getEigenvalues();

    PhyloNode *dad = info.dad, *node = (PhyloNode*)info.dad_branch->node;
    double *echild = echildren;
    if (echild == NULL)
        echild = info.echildren;
    double *partial_lh_leaf = partial_lh_leaves;
    if (partial_lh_leaf == NULL)
        partial_lh_leaf = info.partial_lh_leaves;

    // HK: tmp remove Nonreversible model
    //----------- Non-reversible model --------------

    //----------- Reversible model --------------
    // HK: removed vectorized version
    // non-vectorized version
    double expchild[nstates];
    FOR_NEIGHBOR_IT(node, dad, it) {
            PhyloNeighbor *child = (PhyloNeighbor*)*it;
            // precompute information buffer
            double *echild_ptr = echild;
//            for (c = 0; c < ncat_mix; c++) {
            double len_child = child->getLength(0); // branch length: since no catergory index is 0
            double *eval_ptr = eval;
            double *evec_ptr = evec;
            for (i = 0; i < nstates; i++) {
                expchild[i] = exp(eval_ptr[i]*len_child);
            }
            for (x = 0; x < nstates; x++) {
                for (i = 0; i < nstates; i++) {
                    echild_ptr[i] = evec_ptr[x*nstates+i] * expchild[i];
                }
                echild_ptr += nstates;
            }
            // pre compute information for tip
            if (child->node->isLeaf()) {
                //vector<int>::iterator it;
                for (int state = 0; state <= aln->STATE_UNKNOWN; state++) {
                    double *this_partial_lh_leaf = partial_lh_leaf + state*block;
                    double *echild_ptr = echild;

                    // HK: tmp remove cat
                    double *this_tip_partial_lh = tip_partial_lh + state*tip_block;
                    for (x = 0; x < nstates; x++) {
                        double vchild = echild_ptr[0] * this_tip_partial_lh[0];
                        for (i = 1; i < nstates; i++) {
                            vchild += echild_ptr[i] * this_tip_partial_lh[i];
                        }
                        this_partial_lh_leaf[x] = vchild;
                        echild_ptr += nstates;
                    }
                    this_partial_lh_leaf += nstates;

                }
                size_t addr = aln->STATE_UNKNOWN * block;
                for (x = 0; x < block; x++) {
                    partial_lh_leaf[addr+x] = 1.0;
                }
                partial_lh_leaf += (aln->STATE_UNKNOWN+1)*block;
            }
            echild += block*nstates;
        }

}



#endif //IQTREE_PHYLOKERNELGPU_H
