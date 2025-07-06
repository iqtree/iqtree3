//
// Created by Hashara Kumarasinghe on 30/6/2025.
//

#include "phylotree.h"

void PhyloTree::computeTipPartialLikelihoodGPU() {
    if ((tip_partial_lh_computed & 1) != 0)
        return;
    tip_partial_lh_computed |= 1;


    //-------------------------------------------------------
    // initialize ptn_freq and ptn_invar
    //-------------------------------------------------------

    computePtnFreq();
    // for +I model
//    computePtnInvar(); // HK: tmp remove +I model

    // HK: tmp remove site-specific model
/*
    if (getModel()->isSiteSpecificModel()) {
        // TODO: THIS NEEDS TO BE CHANGED TO USE ModelSubst::computeTipLikelihood()
//        ModelSet *models = (ModelSet*)model;
        size_t nptn = aln->getNPattern(), max_nptn = ((nptn+vector_size-1)/vector_size)*vector_size, tip_block_size = max_nptn * aln->num_states;
        int nstates = aln->num_states;
        size_t nseq = aln->getNSeq();
        ASSERT(vector_size > 0);


#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int nodeid = 0; nodeid < nseq; nodeid++) {
            auto stateRow = getConvertedSequenceByNumber(nodeid);
            double *partial_lh = tip_partial_lh + tip_block_size*nodeid;
            for (size_t ptn = 0; ptn < nptn; ptn+=vector_size, partial_lh += nstates*vector_size) {
                double *inv_evec = &model->getInverseEigenvectors()[ptn*nstates*nstates];
                for (int v = 0; v < vector_size; v++) {
                    int state = 0;
                    if (ptn+v < nptn) {
                        if (stateRow!=nullptr) {
                            state = stateRow[ptn+v];
                        } else {
                            state = aln->at(ptn+v)[nodeid];
                        }
                    }
                    if (state < nstates) {
                        for (int i = 0; i < nstates; i++)
                            partial_lh[i*vector_size+v] = inv_evec[(i*nstates+state)*vector_size+v];
                    } else if (state == aln->STATE_UNKNOWN) {
                        // special treatment for unknown char
                        for (int i = 0; i < nstates; i++) {
                            double lh_unknown = 0.0;
                            for (int x = 0; x < nstates; x++) {
                                lh_unknown += inv_evec[(i*nstates+x)*vector_size+v];
                            }
                            partial_lh[i*vector_size+v] = lh_unknown;
                        }
                    } else {
                        double lh_ambiguous;
                        // ambiguous characters
                        int ambi_aa[] = {
                                4+8, // B = N or D
                                32+64, // Z = Q or E
                                512+1024 // U = I or L
                        };
                        switch (aln->seq_type) {
                            case SEQ_DNA:
                            {
                                int cstate = state-nstates+1;
                                for (int i = 0; i < nstates; i++) {
                                    lh_ambiguous = 0.0;
                                    for (int x = 0; x < nstates; x++)
                                        if ((cstate) & (1 << x))
                                            lh_ambiguous += inv_evec[(i*nstates+x)*vector_size+v];
                                    partial_lh[i*vector_size+v] = lh_ambiguous;
                                }
                            }
                                break;
                            case SEQ_PROTEIN:
                                //map[(unsigned char)'B'] = 4+8+19; // N or D
                                //map[(unsigned char)'Z'] = 32+64+19; // Q or E
                            {
                                int cstate = state-nstates;
                                for (int i = 0; i < nstates; i++) {
                                    lh_ambiguous = 0.0;
                                    for (int x = 0; x < 11; x++)
                                        if (ambi_aa[cstate] & (1 << x))
                                            lh_ambiguous += inv_evec[(i*nstates+x)*vector_size+v];
                                    partial_lh[i*vector_size+v] = lh_ambiguous;
                                }
                            }
                                break;
                            default:
                                ASSERT(0);
                                break;
                        }
                    }
                    // sanity check
                    //                bool all_zero = true;
                    //                for (i = 0; i < nstates; i++)
                    //                    if (partial_lh[i] != 0) {
                    //                        all_zero = false;
                    //                        break;
                    //                    }
                    //                assert(!all_zero && "some tip_partial_lh are all zeros");

                } // FOR v
            } // FOR ptn
            // NO Need to copy dummy anymore
            // dummy values
//            for (ptn = nptn; ptn < max_nptn; ptn++, partial_lh += nstates)
//                memcpy(partial_lh, partial_lh-nstates, nstates*sizeof(double));
        } // FOR nodeid
        return;
    }
*/

    // 2020-06-23: refactor to use computeTipLikelihood
    int nmixtures = 1;
    // HK: tmp remove Mixture related code
/*    if (getModel()->useRevKernel())
        nmixtures = getModel()->getNMixtures();*/
    int nstates = getModel()->num_states;
    int state;

    // HK: tmp remove SEQ_POMO
/*    if (aln->seq_type == SEQ_POMO) {
        if (aln->pomo_sampling_method != SAMPLING_WEIGHTED_BINOM &&
            aln->pomo_sampling_method != SAMPLING_WEIGHTED_HYPER)
            outError("Sampling method not supported by PoMo.");
        ASSERT(aln->STATE_UNKNOWN == nstates + aln->pomo_sampled_states.size());
    }*/

    // assign tip_partial_lh for all admissible states
    for (state = 0; state <= aln->STATE_UNKNOWN; state++) {
        double *state_partial_lh = &tip_partial_lh[state*nstates*nmixtures];
        getModel()->computeTipLikelihood(state, state_partial_lh);
        if (getModel()->useRevKernel()) {
            // transform to inner product of tip likelihood and inverse-eigenvector
            getModel()->multiplyWithInvEigenvector(state_partial_lh);
        }
    }

}
