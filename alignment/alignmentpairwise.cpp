/***************************************************************************
 *   Copyright (C) 2009 by BUI Quang Minh   *
 *   minh.bui@univie.ac.at   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#include "alignmentpairwise.h"
#include "tree/phylosupertree.h"

AlignmentPairwise::AlignmentPairwise()
        : Alignment(), Optimization()
{
    total_size         = 0;
    pair_freq          = nullptr;
    tree               = nullptr;
    num_states         = 0;
    num_states_squared = 0;
    STATE_UNKNOWN      = 0;
    trans_size         = 0;
    trans_mat          = nullptr;
    sum_trans_mat      = nullptr;
    trans_derv1        = nullptr;
    trans_derv2        = nullptr;
    sum_derv1          = nullptr;
    sum_derv2          = nullptr;
    sum_trans          = nullptr;
    pairCount = 0;
    derivativeCalculationCount = 0;
    costCalculationCount = 0;
}

void AlignmentPairwise::setTree(PhyloTree* atree) {
    //
    //Note: Should only be called from constructors;
    //      If it is called multiple times on the same instance
    //      it will leak memory.
    //
    tree               = atree;
    num_states         = tree->aln->num_states;
    num_states_squared = num_states * num_states;
    STATE_UNKNOWN      = tree->aln->STATE_UNKNOWN;
    trans_size         = 0;
    auto rate          = tree->getRate();
    bool isRateSiteSpecific = (rate==nullptr) ? false : rate->isSiteSpecificRate();
    auto model         = tree->getModel();
    bool isModelSiteSpecific = (model==nullptr) ? false: model->isSiteSpecificModel();
    if (model!=nullptr) {
        trans_size = num_states_squared;
    }
    if (!isModelSiteSpecific && !isRateSiteSpecific
        && rate!=nullptr && rate->getPtnCat(0) >= 0) {
        total_size *= rate->getNDiscreteRate();
    }
    trans_mat     = new double[trans_size];
    sum_trans_mat = new double[trans_size];
    sum_trans     = new double[trans_size];
    sum_derv1     = new double[trans_size];
    sum_derv2     = new double[trans_size];
    trans_derv1   = new double[trans_size];
    trans_derv2   = new double[trans_size];
    total_size    = num_states_squared;
    pair_freq     = new double[total_size];
    
    pairCount = 0;
    derivativeCalculationCount = 0;
    costCalculationCount = 0;
}

AlignmentPairwise::AlignmentPairwise(PhyloTree* tree) {
    setTree(tree);
}

void AlignmentPairwise::setSequenceNumbers(int seq1, int seq2) {
    ++pairCount;
    seq_id1 = seq1;
    seq_id2 = seq2;
    auto rate = tree->getRate();
    bool isRateSiteSpecific = (rate==nullptr) ? false : rate->isSiteSpecificRate();
    auto model = tree->getModel();
    bool isModelSiteSpecific = (model==nullptr) ? false: model->isSiteSpecificModel();
    if (isRateSiteSpecific || isModelSiteSpecific) {
        return;
    }
    memset(pair_freq, 0, sizeof(double)*total_size);
    if (tree->hasMatrixOfConvertedSequences()
         && rate->getPtnCat(0) < 0 ) {
        auto sequence1        = tree->getConvertedSequenceByNumber(seq1);
        auto sequence2        = tree->getConvertedSequenceByNumber(seq2);
        auto frequencies      = tree->getConvertedSequenceFrequencies();
        size_t sequenceLength = tree->getConvertedSequenceLength();
        for (size_t i=0; i<sequenceLength; ++i) {
            int state1 = sequence1[i];
            if (num_states<=state1) {
                continue;
            }
            auto pairRow = pair_freq + state1*num_states;
            int  state2  = sequence2[i];
            if (num_states<=state2) {
                continue;
            }
            if ( state1 != STATE_UNKNOWN && state2 != STATE_UNKNOWN ) {
                pairRow[state2] += frequencies[i];
            }
        }
        //Add back the cumulative frequencies for any sites
        //that have the same state in every sequence.
        for (int state=0; state<num_states; ++state) {
            pair_freq[state*num_states + state]
                += tree->getSumOfFrequenciesForSitesWithConstantState(state);
        }
        //Todo: Handle the multiple category case here
        return;
    } else if (tree->getRate()->getPtnCat(0) >= 0) {
        int i = 0;
        for (auto it = tree->aln->begin(); it != tree->aln->end(); it++, i++) {
            int state1 = tree->aln->convertPomoState((*it)[seq_id1]);
            int state2 = tree->aln->convertPomoState((*it)[seq_id2]);
            addPattern(state1, state2, it->frequency, rate->getPtnCat(i));
        }
        return;
    } else {
        for (auto it = tree->aln->begin(); it != tree->aln->end(); it++) {
            int state1 = tree->aln->convertPomoState((*it)[seq_id1]);
            int state2 = tree->aln->convertPomoState((*it)[seq_id2]);
            addPattern(state1, state2, it->frequency);
        }
        return;
    }
}

AlignmentPairwise::AlignmentPairwise(PhyloTree *atree, int seq1, int seq2)
    : Alignment(), Optimization() {
    setTree(atree);
    setSequenceNumbers(seq1,seq2);
}
    
bool AlignmentPairwise::addPattern(int state1, int state2, int freq, int cat) {
    int i;
    if (state1 == STATE_UNKNOWN || state2 == STATE_UNKNOWN) {
        return true;
    }
    double *pair_pos = pair_freq;
    if (0<cat) {
        pair_pos += cat*num_states_squared;
    }
    if (state1 < num_states && state2 < num_states) {
        // unambiguous case
        pair_pos[state1*num_states + state2] += freq;
        return false;
    }
    
    return true;
    
    if (state1 < num_states) {
        // ambiguous character, for DNA, RNA
        state2 = state2 - (num_states - 1);
        for (i = 0; i < num_states; i++) {
            if (state2 & (1 << i)) {
                pair_pos[state1*num_states + i] += freq;
            }
        }
        return false;
    }

    if (state2 < num_states) {
        // ambiguous character, for DNA, RNA
        state1 = state1 - (num_states - 1);
        for (i = 0; i < num_states; i++) {
            if (state1 & (1 << i)) {
                pair_pos[i*num_states + state2] += freq;
            }
        }
        return false;
    }

    return true;
}

double AlignmentPairwise::computeFunction(double value) {
    ++costCalculationCount;
    ModelSubst *model = tree->getModel();
    RateHeterogeneity *site_rate = tree->getRate();
    ModelFactory *model_factory = tree->getModelFactory();
    size_t nptn = tree->aln->getNPattern();
    size_t ncat = site_rate->getNDiscreteRate(); // # rate categories
    size_t mcat = (model_factory->fused_mix_rate) ? 1 : ncat; // # rate categories per mixture class
    size_t ncat_mix = mcat * model->getNMixtures(); // # rate-mixture categories
    double lh = 0.0;
    if (tree->hasMatrixOfConvertedSequences()) {
        auto sequence1        = tree->getConvertedSequenceByNumber(seq_id1);
        auto sequence2        = tree->getConvertedSequenceByNumber(seq_id2);
        auto frequencies      = tree->getConvertedSequenceFrequencies();
        size_t sequenceLength = tree->getConvertedSequenceLength();
        // site-specific rates
        if (site_rate->isSiteSpecificRate()) {
            for (int i = 0; i < sequenceLength; i++) {
                int state1 = sequence1[i];
                int state2 = sequence2[i];
                if (state1 >= num_states || state2 >= num_states) {
                    continue;
                }
                double rate_val = site_rate->getPtnRate(i);
                double trans = tree->getModelFactory()->computeTrans(value * rate_val, state1, state2);
                lh -= log(trans) * frequencies[i];
            }
            return lh;
        }
        // site-specific model
        if (tree->getModel()->isSiteSpecificModel()) {
            for (int i = 0; i < nptn; i++) {
                int state1 = sequence1[i];
                int state2 = sequence2[i];
                if (state1 >= num_states || state2 >= num_states) {
                    continue;
                }
                int model_id = model->getPtnModelID(i);
                double rate_val = site_rate->getPtnRate(i);
                double trans = tree->getModelFactory()->computeTrans(value * rate_val, state1, state2, model_id);
                lh -= log(trans) * frequencies[i];
            }
            return lh;
        }
    }
    // site-specific rates
    if (site_rate->isSiteSpecificRate()) {
        for (int i = 0; i < nptn; i++) {
            int state1 = tree->aln->at(i)[seq_id1];
            int state2 = tree->aln->at(i)[seq_id2];
            if (state1 >= num_states || state2 >= num_states) {
                continue;
            }
            double rate_val = site_rate->getPtnRate(i);
            double trans = tree->getModelFactory()->computeTrans(value * rate_val, state1, state2);
            lh -= log(trans) * tree->aln->at(i).frequency;
        }
        return lh;
    }
    // site-specific model
    if (tree->getModel()->isSiteSpecificModel()) {
        for (int i = 0; i < nptn; i++) {
            int state1 = tree->aln->at(i)[seq_id1];
            int state2 = tree->aln->at(i)[seq_id2];
            if (state1 >= num_states || state2 >= num_states) {
                continue;
            }
            int model_id = model->getPtnModelID(i);
            double rate_val = site_rate->getPtnRate(i);
            double trans = tree->getModelFactory()->computeTrans(value * rate_val, state1, state2, model_id);
            lh -= log(trans) * tree->aln->at(i).frequency;
        }
        return lh;
    }
    // categorized rates
    if (site_rate->getPtnCat(0) >= 0) {
        for (int cat = 0; cat < ncat; cat++) {
            tree->getModelFactory()->computeTransMatrix(value*site_rate->getRate(cat), trans_mat);
            double *pair_pos = pair_freq + cat*trans_size;
            for (int i = 0; i < trans_size; i++)
                if (pair_pos[i] > Params::getInstance().min_branch_length) {
                    if (trans_mat[i] <= 0) {
                      throw "Negative transition probability";
                    }
                    lh -= pair_pos[i] * log(trans_mat[i]);
                }
        }
        return lh;
    }
    // usual model and rates
    memset(sum_trans_mat, 0, sizeof(double) * trans_size);
    for (size_t cm = 0; cm < ncat_mix; cm++) {
        size_t m = cm/mcat;
        size_t c = cm%ncat;
        double rate = site_rate->getRate(c);
        double prop = site_rate->getProp(c) * model->getMixtureWeight(m);
        tree->getModelFactory()->computeTransMatrix(value * rate, trans_mat, m);
        for (int i = 0; i < trans_size; i++) {
            sum_trans_mat[i] += trans_mat[i] * prop;
        }
    }
    double p_invar = site_rate->getPInvar();
    if (p_invar > 0.0) {
        for (int x = 0; x < num_states; x++) {
            sum_trans_mat[x*num_states+x] += p_invar;
        }
    }
    for (int i = 0; i < trans_size; i++) {
        lh -= pair_freq[i] * log(sum_trans_mat[i]);
    }
    return lh;
}

void AlignmentPairwise::computeFuncDerv(double value, double &df, double &ddf) {
    ++derivativeCalculationCount;
    ModelSubst *model = tree->getModel();
    RateHeterogeneity *site_rate = tree->getRate();
    ModelFactory *model_factory = tree->getModelFactory();
    size_t nptn = tree->aln->getNPattern();
    size_t ncat = site_rate->getNDiscreteRate(); // # rate categories
    size_t mcat = (model_factory->fused_mix_rate) ? 1 : ncat; // # rate categories per mixture class
    size_t ncat_mix = mcat * model->getNMixtures(); // # rate-mixture categories
    df = 0.0;
    ddf = 0.0;
    auto sequence1        = tree->getConvertedSequenceByNumber(seq_id1);
    auto sequence2        = tree->getConvertedSequenceByNumber(seq_id2);
    auto frequencies      = tree->getConvertedSequenceFrequencies();
    size_t sequenceLength = tree->getConvertedSequenceLength();
    if (sequenceLength!=nptn) {
        sequence1 = sequence2 = nullptr;
        frequencies = nullptr;
    }
    // site-specific rates
    if (site_rate->isSiteSpecificRate()) {
        if (sequence1!=nullptr && sequence2!=nullptr && frequencies!=nullptr) {
            #pragma omp parallel for reduction(-:df,ddf) schedule(dynamic,100)
            for (int i = 0; i < nptn; ++i) {
                int state1 = sequence1[i];
                if (num_states<=state1) {
                    continue;
                }
                int state2 = sequence2[i];
                if (num_states<=state2) {
                    continue;
                }
                double freq = frequencies[i];
                double rate_val = site_rate->getPtnRate(i);
                double rate_sqr = rate_val * rate_val;
                double derv1, derv2;
                double trans = tree->getModelFactory()->computeTrans(value * rate_val, state1, state2, derv1, derv2);
                double d1 = derv1 / trans;
                df -= rate_val * d1 * freq;
                ddf -= rate_sqr * (derv2/trans - d1*d1) * freq;
            }
        } else {
            for (int i = 0; i < nptn; i++) {
                int state1 = tree->aln->at(i)[seq_id1];
                if (num_states<=state1) {
                    continue;
                }
                int state2 = tree->aln->at(i)[seq_id2];
                if (num_states<=state2) {
                    continue;
                }
                double rate_val = site_rate->getPtnRate(i);
                double rate_sqr = rate_val * rate_val;
                double derv1, derv2;
                double trans = tree->getModelFactory()->computeTrans(value * rate_val, state1, state2, derv1, derv2);
                double d1 = derv1 / trans;
                double freq = tree->aln->at(i).frequency;
                df -= rate_val * d1 * freq;
                ddf -= rate_sqr * (derv2/trans - d1*d1) * freq;
            }
        }
        return;
    }
    // site-specific model
    if (tree->getModel()->isSiteSpecificModel()) {
        if (sequence1!=nullptr && sequence2!=nullptr && frequencies!=nullptr) {
            #pragma omp parallel for reduction(-:df,ddf) schedule(dynamic,100)
            for (int i = 0; i < nptn; i++) {
                int state1 = sequence1[i];
                if (num_states<=state1) {
                    continue;
                }
                int state2 = sequence2[i];
                if (num_states<=state2) {
                    continue;
                }
                int model_id = model->getPtnModelID(i);
                double freq = frequencies[i];
                double rate_val = site_rate->getPtnRate(i);
                double rate_sqr = rate_val * rate_val;
                double derv1, derv2;
                double trans = tree->getModel()->computeTrans(value * rate_val, state1, state2, derv1, derv2, model_id);
                double d1 = derv1 / trans;
                df -= rate_val * d1 * freq;
                ddf -= rate_sqr * (derv2/trans - d1*d1) * freq;
            }
        } else {
            for (int i = 0; i < nptn; i++) {
                int state1 = tree->aln->at(i)[seq_id1];
                if (num_states<=state1) {
                    continue;
                }
                int state2 = tree->aln->at(i)[seq_id2];
                if (num_states<=state2) {
                    continue;
                }
                int model_id = model->getPtnModelID(i);
                double rate_val = site_rate->getPtnRate(i);
                double rate_sqr = rate_val * rate_val;
                double derv1, derv2;
                double trans = tree->getModel()->computeTrans(value * rate_val, state1, state2, derv1, derv2, model_id);
                double d1 = derv1 / trans;
                double freq = tree->aln->at(i).frequency;
                df -= rate_val * d1 * freq;
                ddf -= rate_sqr * (derv2/trans - d1*d1) * freq;
            }
        }
        return;
    }
    // categorized rates
    if (site_rate->getPtnCat(0) >= 0) {
        for (int cat = 0; cat < ncat; cat++) {
            double rate_val = site_rate->getRate(cat);
            double derv1 = 0.0, derv2 = 0.0;
            tree->getModelFactory()->computeTransDerv(value*rate_val, trans_mat, trans_derv1, trans_derv2);
            double *pair_pos = pair_freq + cat*trans_size;
            for (int i = 0; i < trans_size; i++) if (pair_pos[i] > 0) {
                if (trans_mat[i] <= 0) {
                    throw "Negative transition probability";
                }
                double d1 = trans_derv1[i] / trans_mat[i];
                derv1 += pair_pos[i] * d1;
                derv2 += pair_pos[i] * (trans_derv2[i]/trans_mat[i] - d1 * d1);
            }
            df -= derv1 * rate_val;
            ddf -= derv2 * rate_val * rate_val;
        }
        return;
    }
    // usual model and rates
    memset(sum_trans, 0, sizeof(double) * trans_size);
    memset(sum_derv1, 0, sizeof(double) * trans_size);
    memset(sum_derv2, 0, sizeof(double) * trans_size);
    for (size_t cm = 0; cm < ncat_mix; cm++) {
        size_t m = cm/mcat;
        size_t c = cm%ncat;
        double rate = site_rate->getRate(c);
        double prop = site_rate->getProp(c) * model->getMixtureWeight(m);
        double prop_rate = prop * rate;
        double prop_rate2 = prop_rate * rate;
        tree->getModelFactory()->computeTransDerv(value * rate, trans_mat, trans_derv1, trans_derv2, m);
        for (int i = 0; i < trans_size; i++) {
            sum_trans[i] += trans_mat[i] * prop;
            sum_derv1[i] += trans_derv1[i] * prop_rate;
            sum_derv2[i] += trans_derv2[i] * prop_rate2;
        }
    }
    double p_invar = site_rate->getPInvar();
    if (p_invar > 0.0) {
        for (int x = 0; x < num_states; x++) {
            sum_trans[x*num_states+x] += p_invar;
        }
    }
    for (int i = 0; i < trans_size; i++) {
        if (pair_freq[i] > Params::getInstance().min_branch_length && sum_trans[i] > 0.0) {
            double d1 = sum_derv1[i] / sum_trans[i];
            df  -= pair_freq[i] * d1;
            ddf -= pair_freq[i] * (sum_derv2[i]/sum_trans[i] - d1 * d1);
        }
    }
    return;
}

double AlignmentPairwise::optimizeDist(double initial_dist, double &d2l) {
    // initial guess of the distance using Juke-Cantor correction
    double dist = initial_dist;
    d2l = -1.0;
    
    // if no model or rate is specified, return the JC distance and set variance to const
    if (!tree->getModelFactory() || !tree->getRate()) {
        return dist;
    }
    double negative_lh, ferror;
    double max_genetic_dist = MAX_GENETIC_DIST;
    if (tree->aln->seq_type == SEQ_POMO) {
        int N = tree->aln->virtual_pop_size;
        max_genetic_dist *= N*N;
    }
    ++costCalculationCount;
    double min_branch = Params::getInstance().min_branch_length;
    if (tree->optimize_by_newton) { // Newton-Raphson method
        dist = minimizeNewton(min_branch, dist, max_genetic_dist, min_branch, d2l);
    } else { // Brent method
        dist = minimizeOneDimen(min_branch, dist, max_genetic_dist, min_branch, &negative_lh, &ferror);
    }
    return dist;
}

double AlignmentPairwise::optimizeDist(double initial_dist) {
	double d2l;
	return optimizeDist(initial_dist, d2l);
}

double AlignmentPairwise::recomputeDist
    ( int seq1, int seq2, double initial_dist, double &d2l ) {
    //Only called when -experimental has been passed
    if (initial_dist == 0.0) {
        if (tree->hasMatrixOfConvertedSequences()) {
            int distance    = 0;
            int denominator = 0;
            auto sequence1        = tree->getConvertedSequenceByNumber(seq1);
            auto sequence2        = tree->getConvertedSequenceByNumber(seq2);
            auto nonConstSiteFreq = tree->getConvertedSequenceNonConstFrequencies();
            size_t sequenceLength = tree->getConvertedSequenceLength();
            for (size_t i=0; i<sequenceLength; ++i) {
                auto state1 = sequence1[i];
                auto state2 = sequence2[i];
                if ( state1 != STATE_UNKNOWN && state2 != STATE_UNKNOWN ) {
                    denominator += nonConstSiteFreq[i];
                    if ( state1 != state2 ) {
                        distance += nonConstSiteFreq[i];
                    }
                }
            }
            if (0<distance) {
                initial_dist = (double)distance / (double)denominator;
            }
            if (tree->params->compute_obs_dist) {
                return initial_dist;
            }
            initial_dist = tree->aln->computeJCDistanceFromObservedDistance(initial_dist);
        }
        else if (tree->params->compute_obs_dist)
            return (initial_dist = tree->aln->computeObsDist(seq1, seq2));
        else
            initial_dist = tree->aln->computeDist(seq1, seq2);
    }
    if (!tree->hasModelFactory() || !tree->hasRateHeterogeneity())
    {
        return initial_dist;
    }
    setSequenceNumbers(seq1, seq2);
    return optimizeDist(initial_dist, d2l);
}

AlignmentPairwise::~AlignmentPairwise()
{
    delete [] sum_derv2;
    delete [] sum_derv1;
    delete [] sum_trans;
    delete [] trans_derv2;
    delete [] trans_derv1;
    delete [] sum_trans_mat;
    delete [] trans_mat;
    delete [] pair_freq;
}
