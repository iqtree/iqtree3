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

#include "tree/phylotree.h"
#include <algorithm>

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
    trans_derv1        = nullptr;
    trans_derv2        = nullptr;
    sum_trans          = nullptr;
    sum_derv1          = nullptr;
    sum_derv2          = nullptr;
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
    total_size = num_states_squared;
    if (!isModelSiteSpecific && !isRateSiteSpecific
        && rate!=nullptr && rate->getPtnCat(0) >= 0) {
        total_size *= rate->getNDiscreteRate();
    }
    pair_freq     = new double[total_size];
    trans_mat     = new double[trans_size];
    trans_derv1   = new double[trans_size];
    trans_derv2   = new double[trans_size];
    sum_trans     = new double[trans_size];
    sum_derv1     = new double[trans_size];
    sum_derv2     = new double[trans_size];

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
    ModelSubst *model = tree->getModel();
    RateHeterogeneity *site_rate = tree->getRate();
    size_t nptn = tree->aln->getNPattern();
    bool isModelSiteSpecific = (model) ? model->isSiteSpecificModel() : false;
    bool isRateSiteSpecific = (site_rate) ? site_rate->isSiteSpecificRate() : false;
    bool isRateCategorized = (site_rate) ? (site_rate->getPtnCat(0) >= 0) : false;
    if (isModelSiteSpecific || isRateSiteSpecific) {
        return;
    }
    std::fill_n(pair_freq, total_size, 0.0);
    if (tree->hasMatrixOfConvertedSequences() && !isRateCategorized) {
        const char *sequence1 = tree->getConvertedSequenceByNumber(seq1);
        const char *sequence2 = tree->getConvertedSequenceByNumber(seq2);
        const int *frequencies = tree->getConvertedSequenceFrequencies();
        size_t sequenceLength = tree->getConvertedSequenceLength();
        for (size_t i = 0; i < sequenceLength; ++i) {
            int state1 = sequence1[i];
            int state2 = sequence2[i];
            if (state1 >= num_states || state2 >= num_states) {
                continue;
            }
            double *pairRow = pair_freq + state1*num_states;
            if (state1 != STATE_UNKNOWN && state2 != STATE_UNKNOWN) {
                pairRow[state2] += frequencies[i];
            }
        }
        // Add the cumulative frequencies of the constant patterns if
        // such patterns are not included in the converted sequences
        for (int state = 0; state < num_states; ++state) {
            pair_freq[state*num_states + state]
                += tree->getSumOfFrequenciesForSitesWithConstantState(state);
        }
        return;
    }
    for (size_t ptn = 0; ptn < nptn; ++ptn) {
        const Pattern &pat = tree->aln->at(ptn);
        int state1 = tree->aln->convertPomoState(pat[seq_id1]);
        int state2 = tree->aln->convertPomoState(pat[seq_id2]);
        double freq = double(pat.frequency);
        addPattern(state1, state2, freq, site_rate->getPtnCat(ptn));
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
    double lh = 0.0, df = 0.0, ddf = 0.0;
    likelihoodKernelFunction<false>(value, lh, df, ddf);
    return -lh;
}

void AlignmentPairwise::computeFuncDerv(double value, double &df, double &ddf) {
    ++derivativeCalculationCount;
    double lh = 0.0;
    likelihoodKernelFunction<true>(value, lh, df, ddf);
    df = -df;
    ddf = -ddf;
}

template <bool COMPUTE_DERV>
void AlignmentPairwise::likelihoodKernelFunction(double value, double &lh, double &df, double &ddf) {
    ModelSubst *model = tree->getModel();
    RateHeterogeneity *site_rate = tree->getRate();
    ModelFactory *model_factory = tree->getModelFactory();
    size_t nptn = tree->aln->getNPattern();
    size_t ncat = site_rate->getNRate(); // # rate categories
    size_t mcat = (model_factory->fused_mix_rate) ? 1 : ncat; // # rate categories per mixture class
    size_t nmix = model->getNMixtures(); // # mixture classes
    size_t ncat_mix = mcat * nmix; // # rate-mixture categories
    lh = df = ddf = 0.0;
    const double MIN_FREQ = Params::getInstance().min_branch_length;
    const char *sequence1 = tree->getConvertedSequenceByNumber(seq_id1);
    const char *sequence2 = tree->getConvertedSequenceByNumber(seq_id2);
    const int *frequencies = tree->getConvertedSequenceFrequencies();
    size_t sequenceLength = tree->getConvertedSequenceLength();
    bool use_converted = tree->hasMatrixOfConvertedSequences() && (sequenceLength == nptn);
    auto getPtnStatesAndFreq =
    [use_converted, sequence1, sequence2, frequencies, this](size_t ptn, int &state1, int &state2, double &freq) {
        if (use_converted) {
            state1 = sequence1[ptn];
            state2 = sequence2[ptn];
            freq = double(frequencies[ptn]);
        } else {
            const Pattern &pat = tree->aln->at(ptn);
            state1 = pat[seq_id1];
            state2 = pat[seq_id2];
            freq = double(pat.frequency);
        }
    };
    // site-specific model or rates
    // Covers all relevant combinations:
    // - site-specific model + site-specific/categorized/usual rates
    // - usual model + site-specific rates
    if (model->isSiteSpecificModel() || site_rate->isSiteSpecificRate()) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:lh,df,ddf) schedule(dynamic,100)
#endif
        for (size_t ptn = 0; ptn < nptn; ++ptn) {
            double freq;
            int state1, state2;
            getPtnStatesAndFreq(ptn, state1, state2, freq);
            if (state1 >= num_states || state2 >= num_states) {
                continue;
            }
            double lh_ptn = 0.0, df_ptn = 0.0, ddf_ptn = 0.0;
            int model_id = model->getPtnModelID(ptn);
            double rate = site_rate->getPtnRate(ptn);
            for (size_t cm = 0; cm < ncat_mix; ++cm) {
                size_t m = cm/mcat;
                size_t c = cm%ncat;
                if (nmix > 1) {
                    model_id = m;
                }
                if (ncat > 1) {
                    rate = site_rate->getRate(c);
                }
                double prop = site_rate->getProp(c) * model->getMixtureWeight(m);
                if (!COMPUTE_DERV) {
                    double trans = model_factory->computeTrans(value * rate, state1, state2, model_id);
                    lh_ptn += trans * prop;
                } else {
                    double prop_rate = prop * rate;
                    double prop_rate2 = prop_rate * rate;
                    double derv1, derv2;
                    double trans = model_factory->computeTrans(value * rate, state1, state2, derv1, derv2, model_id);
                    lh_ptn += trans * prop;
                    df_ptn += derv1 * prop_rate;
                    ddf_ptn += derv2 * prop_rate2;
                }
            }
            if (state1 == state2) {
                lh_ptn += site_rate->getPInvar();
            }
            if (!COMPUTE_DERV) {
                lh += log(lh_ptn) * freq;
            } else {
                // df = log(lh)' = lh'/lh
                df_ptn /= lh_ptn;
                df += df_ptn * freq;
                // ddf = log(lh)'' = (lh'/lh)' = lh''/lh - (lh'/lh)^2
                ddf_ptn /= lh_ptn;
                ddf_ptn -= df_ptn * df_ptn;
                ddf += ddf_ptn * freq;
            }
        }
        return;
    }
    // usual model and categorized rates
    if (site_rate->getPtnCat(0) >= 0) {
        ASSERT(site_rate->getPInvar() == 0.0);
        for (size_t cat = 0; cat < site_rate->getNDiscreteRate(); ++cat) {
            std::fill_n(sum_trans, trans_size, 0.0);
            std::fill_n(sum_derv1, trans_size, 0.0);
            std::fill_n(sum_derv2, trans_size, 0.0);
            double rate = site_rate->getRate(cat);
            for (size_t m = 0; m < nmix; ++m) {
                double prop = model->getMixtureWeight(m);
                if (!COMPUTE_DERV) {
                    model_factory->computeTransMatrix(value * rate, trans_mat, m);
                    for (int i = 0; i < trans_size; ++i) {
                        sum_trans[i] += trans_mat[i] * prop;
                    }
                } else {
                    double prop_rate = prop * rate;
                    double prop_rate2 = prop_rate * rate;
                    model_factory->computeTransDerv(value * rate, trans_mat, trans_derv1, trans_derv2, m);
                    for (int i = 0; i < trans_size; ++i) {
                        sum_trans[i] += trans_mat[i] * prop;
                        sum_derv1[i] += trans_derv1[i] * prop_rate;
                        sum_derv2[i] += trans_derv2[i] * prop_rate2;
                    }
                }
            }
            double *pair_pos = pair_freq + cat*trans_size;
            for (int i = 0; i < trans_size; ++i) {
                if (pair_pos[i] > MIN_FREQ) {
                    ASSERT(sum_trans[i] > 0.0);
                    if (!COMPUTE_DERV) {
                        lh += log(sum_trans[i]) * pair_pos[i];
                    } else {
                        // df = log(lh)' = lh'/lh
                        double df_pair = sum_derv1[i] / sum_trans[i];
                        df += df_pair * pair_pos[i];
                        // ddf = log(lh)'' = (lh'/lh)' = lh''/lh - (lh'/lh)^2
                        double ddf_pair = sum_derv2[i] / sum_trans[i] - df_pair * df_pair;
                        ddf += ddf_pair * pair_pos[i];
                    }
                }
            }
        }
        return;
    }
    // usual model and rates
    std::fill_n(sum_trans, trans_size, 0.0);
    std::fill_n(sum_derv1, trans_size, 0.0);
    std::fill_n(sum_derv2, trans_size, 0.0);
    for (size_t cm = 0; cm < ncat_mix; ++cm) {
        size_t m = cm/mcat;
        size_t c = cm%ncat;
        double rate = site_rate->getRate(c);
        double prop = site_rate->getProp(c) * model->getMixtureWeight(m);
        if (!COMPUTE_DERV) {
            model_factory->computeTransMatrix(value * rate, trans_mat, m);
            for (int i = 0; i < trans_size; ++i) {
                sum_trans[i] += trans_mat[i] * prop;
            }
        } else {
            double prop_rate = prop * rate;
            double prop_rate2 = prop_rate * rate;
            model_factory->computeTransDerv(value * rate, trans_mat, trans_derv1, trans_derv2, m);
            for (int i = 0; i < trans_size; ++i) {
                sum_trans[i] += trans_mat[i] * prop;
                sum_derv1[i] += trans_derv1[i] * prop_rate;
                sum_derv2[i] += trans_derv2[i] * prop_rate2;
            }
        }
    }
    double p_invar = site_rate->getPInvar();
    if (p_invar > 0.0) {
        for (int x = 0; x < num_states; ++x) {
            sum_trans[x*num_states+x] += p_invar;
        }
    }
    for (int i = 0; i < trans_size; ++i) {
        if (pair_freq[i] > MIN_FREQ) {
            ASSERT(sum_trans[i] > 0.0);
            if (!COMPUTE_DERV) {
                lh += log(sum_trans[i]) * pair_freq[i];
            } else {
                // df = log(lh)' = lh'/lh
                double df_pair = sum_derv1[i] / sum_trans[i];
                df += df_pair * pair_freq[i];
                // ddf = log(lh)'' = (lh'/lh)' = lh''/lh - (lh'/lh)^2
                double ddf_pair = sum_derv2[i] / sum_trans[i] - df_pair * df_pair;
                ddf += ddf_pair * pair_freq[i];
            }
        }
    }
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
    delete [] trans_mat;
    delete [] pair_freq;
}
