/***************************************************************************
 *   Copyright (C) 2026 by BUI Quang Minh                                  *
 *   m.bui@anu.edu.au                                                      *
 *   This code is started by Claude Code                                   *
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
#include "rategammastarinvar.h"

RateGammaStarInvar::RateGammaStarInvar(int ncat, double shape, double p_invar_sites,
        string optimize_alg, PhyloTree *tree, bool testParamDone) :
        RateInvar(p_invar_sites, tree), RateGammaStar(ncat, shape, tree) {
    name = "+I" + name;
    full_name = "Invar+" + full_name;
    this->optimize_alg = optimize_alg;
    cur_optimize = 0;
    this->testParamDone = testParamDone;
    for (int cat = 0; cat < ncategory; cat++)
        rates[cat] = 1.0 / (1.0 - p_invar);
    computeRates();
}

void RateGammaStarInvar::computeRates() {
    // Preserve the scale currently baked into rates[], exactly as
    // RateGamma::computeRates() does for +I+G. setPInvar() seeds that scale with
    // 1/(1-p_invar), so that the mean rate over all sites, invariable ones
    // included, is (1-p_invar) * sum_i prop[i]*rates[i] == 1.
    // Keeping the scale (instead of deriving it from the current p_invar) matters
    // for the EM algorithm: it updates p_invar without touching the category
    // rates, and would otherwise see the likelihood jump between iterations.
    double cur_scale = 0.0;
    for (int cat = 0; cat < ncategory; cat++)
        cur_scale += prop[cat] * rates[cat];

    // Lloyd-Max bins: weights sum to 1 and sum_i prop[i]*rates[i] == 1
    RateGammaStar::computeRates();

    if (cur_scale > 0.0)
        for (int cat = 0; cat < ncategory; cat++)
            rates[cat] *= cur_scale;
}

void RateGammaStarInvar::setPInvar(double pInvar) {
    p_invar = pInvar;
    // re-seed the rate scale, see computeRates()
    for (int cat = 0; cat < ncategory; cat++)
        rates[cat] = 1.0 / (1.0 - p_invar);
    computeRates();
}

void RateGammaStarInvar::startCheckpoint() {
    // same structure name as RateGammaInvar on purpose, see the header
    checkpoint->startStruct("RateGammaInvar");
}

void RateGammaStarInvar::saveCheckpoint() {
    RateInvar::saveCheckpoint();
    RateGammaStar::saveCheckpoint();
}

void RateGammaStarInvar::restoreCheckpoint() {
    // must restore p_invar first, because RateGamma::restoreCheckpoint() ends
    // with computeRates(), which scales the rates by 1/(1-p_invar)
    RateInvar::restoreCheckpoint();
    for (int cat = 0; cat < ncategory; cat++)
        rates[cat] = 1.0 / (1.0 - p_invar);
    RateGammaStar::restoreCheckpoint();
}

void RateGammaStarInvar::setNCategory(int ncat) {
    RateGammaStar::setNCategory(ncat);
    for (int cat = 0; cat < ncategory; cat++)
        rates[cat] = 1.0 / (1.0 - p_invar);
    name = "+I" + name;
    full_name = "Invar+" + full_name;
    computeRates();
}

string RateGammaStarInvar::getNameParams() {
    return RateInvar::getNameParams() + RateGammaStar::getNameParams();
}

double RateGammaStarInvar::meanRates() {
    double ret = 0.0;
    for (int cat = 0; cat < ncategory; cat++)
        ret += prop[cat] * rates[cat];
    return ret * (1.0 - getPInvar());
}

double RateGammaStarInvar::rescaleRates() {
    double norm = meanRates();
    if (norm <= 0.0)
        return 1.0;
    for (int cat = 0; cat < ncategory; cat++)
        rates[cat] /= norm;
    return norm;
}

double RateGammaStarInvar::computeFunction(double value) {
    if (cur_optimize == 0)
        gamma_shape = value;
    else {
        p_invar = value;
        for (int cat = 0; cat < ncategory; cat++)
            rates[cat] = 1.0 / (1.0 - p_invar);
    }
    // need to compute rates again if p_invar or the Gamma shape changes!
    computeRates();
    phylo_tree->clearAllPartialLH();
    return -phylo_tree->computeLikelihood();
}

void RateGammaStarInvar::writeInfo(ostream &out) {
    RateInvar::writeInfo(out);
    RateGammaStar::writeInfo(out);
}

void RateGammaStarInvar::writeParameters(ostream &out) {
    RateInvar::writeParameters(out);
    RateGammaStar::writeParameters(out);
}

void RateGammaStarInvar::setVariables(double *variables) {
    RateGammaStar::setVariables(variables);
    int gid = RateGammaStar::getNDim();
    RateInvar::setVariables(variables+gid);
}

bool RateGammaStarInvar::getVariables(double *variables) {
    int gid = RateGammaStar::getNDim();
    bool changed = RateGammaStar::getVariables(variables);
    changed |= RateInvar::getVariables(variables+gid);
    if (changed) {
        // NOTE: the virtual computeRates() on purpose -- RateGammaStar's
        // version would leave out the 1/(1-p_invar) scaling
        computeRates();
    }
    return changed;
}

double RateGammaStarInvar::targetFunk(double x[]) {
    ASSERT(phylo_tree);
    getVariables(x);
    phylo_tree->clearAllPartialLH();
    return -phylo_tree->computeLikelihood();
}

void RateGammaStarInvar::setBounds(double *lower_bound, double *upper_bound, bool *bound_check) {
    int gid = RateGammaStar::getNDim();
    RateGammaStar::setBounds(lower_bound, upper_bound, bound_check);
    RateInvar::setBounds(lower_bound+gid, upper_bound+gid, bound_check+gid);
}

double RateGammaStarInvar::optimizeParameters(double gradient_epsilon) {

    int ndim = getNDim();

    // return if nothing to be optimized
    if (ndim == 0) {
        return phylo_tree->computeLikelihood();
    }
    if (verbose_mode >= VB_MED) {
        cout << "Optimizing " << name << " model parameters by " << optimize_alg << " algorithm..." << endl;
    }
    if (optimize_alg.find("EM_RR") != string::npos) {
        return randomRestartOptimization(gradient_epsilon);
    } else if (optimize_alg.find("Brent") != string::npos || phylo_tree->aln->frac_const_sites == 0.0 || isFixPInvar() || isFixGammaShape()) {
        double lh = phylo_tree->computeLikelihood();
        cur_optimize = 0;
        double gamma_lh = RateGammaStar::optimizeParameters(gradient_epsilon);
        ASSERT(gamma_lh >= lh-0.1);
        cur_optimize = 1;
        double invar_lh = RateInvar::optimizeParameters(gradient_epsilon);
        ASSERT(invar_lh >= gamma_lh-0.1);
        cur_optimize = 0;
        return invar_lh;
    } else if (optimize_alg.find("EM") != string::npos) {
        return optimizeWithEM(gradient_epsilon);
    } else if (optimize_alg.find("BFGS") != string::npos) {
        double *variables = new double[ndim+1];
        double *upper_bound = new double[ndim+1];
        double *lower_bound = new double[ndim+1];
        bool *bound_check = new bool[ndim+1];
        double score;

        // by BFGS algorithm
        setVariables(variables);
        setBounds(lower_bound, upper_bound, bound_check);

        score = -minimizeMultiDimen(variables, ndim, lower_bound, upper_bound, bound_check, max(gradient_epsilon, TOL_GAMMA_SHAPE));

        getVariables(variables);

        phylo_tree->clearAllPartialLH();
        score = phylo_tree->computeLikelihood();

        delete [] bound_check;
        delete [] lower_bound;
        delete [] upper_bound;
        delete [] variables;

        return score;
    } else {
        string errMsg = "Unknown optimization algorithm: " + optimize_alg;
        outError(errMsg.c_str());
        return 0.0;
    }
}

int RateGammaStarInvar::computePatternRates(DoubleVector &pattern_rates, IntVector &pattern_cat) {
    phylo_tree->computePatternLhCat(WSL_RATECAT);

    int npattern = phylo_tree->aln->getNPattern();
    pattern_rates.resize(npattern);
    pattern_cat.resize(npattern);

    double *lh_cat = phylo_tree->_pattern_lh_cat;
    for (int i = 0; i < npattern; i++) {
        double sum_rate = 0.0, sum_lh = phylo_tree->ptn_invar[i];
        int best = 0;
        double best_lh = phylo_tree->ptn_invar[i];
        for (int c = 0; c < ncategory; c++) {
            sum_rate += rates[c] * lh_cat[c];
            sum_lh += lh_cat[c];
            if (lh_cat[c] > best_lh || (lh_cat[c] == best_lh && random_double()<0.5)) { // break tie at random
                best = c+1;
                best_lh = lh_cat[c];
            }
        }
        pattern_rates[i] = sum_rate / sum_lh;
        pattern_cat[i] = best;
        lh_cat += ncategory;
    }
    return ncategory+1;
}

double RateGammaStarInvar::optimizeWithEM(double gradient_epsilon) {
    double curlh = phylo_tree->computeLikelihood();

    cur_optimize = 0;
    double gamma_lh = RateGammaStar::optimizeParameters(gradient_epsilon);
    ASSERT(gamma_lh > curlh - 1.0);
    curlh = gamma_lh;

    size_t ncat = getNRate();
    size_t nptn = phylo_tree->aln->getNPattern();
    size_t nSites = phylo_tree->aln->getNSite();

    // Compute the pattern likelihood for each category (invariable and variable category)
    phylo_tree->computePatternLhCat(WSL_RATECAT);
    phylo_tree->computePtnInvar();

    double ppInvar = 0;
    for (size_t ptn = 0; ptn < nptn; ptn++) {
        double *this_lk_cat = phylo_tree->_pattern_lh_cat + ptn * ncat;
        double lk_ptn = phylo_tree->ptn_invar[ptn];
        for (size_t cat = 0; cat < ncat; cat++) {
            lk_ptn += this_lk_cat[cat];
        }
        ASSERT(lk_ptn != 0.0);
        ppInvar += (phylo_tree->ptn_invar[ptn]) * phylo_tree->ptn_freq[ptn] / lk_ptn;
    }

    double newPInvar = ppInvar / nSites;
    ASSERT(newPInvar < 1.0);
    // NOTE: a bare assignment, exactly as RateGammaInvar does. Calling setPInvar()
    // would also rescale the category rates by 1/(1-p_invar) within the EM step,
    // which changes the effective tree length and breaks the monotonicity of EM.
    // The rates are brought back in line by the next computeRates(), i.e. when the
    // shape is optimised in the next round or when setPInvar() is called at the end
    // of ModelFactory::optimizeParametersGammaInvar().
    p_invar = newPInvar;
    phylo_tree->clearAllPartialLH();
    double pinvLH = phylo_tree->computeLikelihood();
    ASSERT(pinvLH > curlh - 1.0);
    return pinvLH;
}

double RateGammaStarInvar::randomRestartOptimization(double gradient_epsilon) {
    if (testParamDone) {
        return optimizeWithEM(gradient_epsilon);
    }
    double frac_const = phylo_tree->aln->frac_const_sites;
    double bestLogl = phylo_tree->getCurScore();
    double bestAlpha = 0.0;
    double bestPInvar = 0.0;
    double testInterval = (frac_const - MIN_PINVAR*2) / 10;
    double initPInv = MIN_PINVAR;
    double initAlpha = getGammaShape();

    while (initPInv <= frac_const) {
        if (verbose_mode >= VB_MED) {
            cout << endl;
            cout << "Testing with init. pinv = " << initPInv << " / init. alpha = " << initAlpha << endl;
        }
        setPInvar(initPInv);
        setGammaShape(initAlpha);
        phylo_tree->clearAllPartialLH();
        double logl = optimizeWithEM(gradient_epsilon);
        double estAlpha = getGammaShape();
        double estPInv = getPInvar();

        if (verbose_mode >= VB_MED) {
            cout << "Est. alpha: " << estAlpha << " / Est. pinv: " << estPInv
            << " / Logl: " << logl << endl;
        }

        initPInv = initPInv + testInterval;

        if (logl > bestLogl) {
            bestLogl = logl;
            bestAlpha = estAlpha;
            bestPInvar = estPInv;
        }
    }

    if (verbose_mode >= VB_MED) {
        cout << "Best gamma shape: " << bestAlpha << " / best p_inv: " << bestPInvar
        << " / logl: " << bestLogl << endl;
    }

    setPInvar(bestPInvar);
    setGammaShape(bestAlpha);
    phylo_tree->clearAllPartialLH();
    testParamDone = true;
    return phylo_tree->computeLikelihood();
}
