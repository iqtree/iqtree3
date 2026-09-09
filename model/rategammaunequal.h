/***************************************************************************
 *   Copyright (C) 2026 by BUI Quang Minh   *
 *   m.bui@anu.edu.au   *
 *   This code is started by Claude Code                                                                      *
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
#ifndef RATEGAMMAUNEQUAL_H
#define RATEGAMMAUNEQUAL_H

#include "rategamma.h"

/** maximum number of Lloyd-Max iterations (coraxlib uses 1000 as well) */
const int MAX_LLOYD_ITER = 1000;

/** relative convergence tolerance on the category rates.
    coraxlib uses 1e-4; we are stricter so that the likelihood is a
    sufficiently smooth function of alpha for the 1-dimensional optimiser */
const double TOL_LLOYD_RATE = 1e-8;

/** bins with less mass than this are left untouched (avoid division by 0) */
const double MIN_LLOYD_MASS = 1e-10;

class PhyloTree;

/**
    Discrete Gamma site-rate model with UNEQUAL category weights.

    RateGamma implements Yang (1994): the Gamma distribution is cut into
    'ncategory' bins of EQUAL probability 1/ncategory, and each bin is
    represented by its mean (or its median). The equal-probability
    constraint is a computational convenience, not a statistical
    requirement -- it is generally not the discretisation that best
    approximates the continuous Gamma with a fixed number of categories.

    This class drops that constraint and computes an optimal quantisation
    of Gamma(alpha, alpha) by the Lloyd-Max algorithm: both the bin
    boundaries and the bin weights are free, and are determined
    (deterministically) by alpha and ncategory. The model therefore still
    has a single free parameter, the shape alpha, exactly like RateGamma;
    only the weights returned by getProp() differ.

    The distortion measure is the generalised Kullback-Leibler (I-)
    divergence d(x,r) = x*log(x/r) - x + r rather than squared error,
    which is the natural loss for a positive rate multiplier. Two
    consequences, both used below:

      - the boundary between two adjacent representatives r1 < r2 is the
        point where the two distortions are equal, i.e. the LOGARITHMIC
        mean (r2 - r1) / (log r2 - log r1);
      - as for any Bregman divergence, the optimal representative of a
        bin is the conditional mean E[X | bin], so a bin rate is still
        the mean of the portion of the Gamma distribution falling in
        that category -- only the bin boundaries and weights differ from
        RateGamma.

    Because sum_i w_i * E[X | bin_i] = E[X] = 1, the mean rate is 1 by
    construction and no rescaling of branch lengths is needed.

    Ported from coraxlib (branch gamma-fix), functions
    corax_compute_gamma_cats_opt_weights() and lloyd_max_init_cats().

    @author BUI Quang Minh <minh.bui@univie.ac.at>
*/
class RateGammaUnequal : public RateGamma
{

public:
    /**
        constructor
        @param ncat number of rate categories
        @param shape Gamma shape parameter
        @param tree associated phylogenetic tree
    */
    RateGammaUnequal(int ncat, double shape, PhyloTree *tree);

    /**
        destructor
    */
    virtual ~RateGammaUnequal();

    /**
        start structure for checkpointing
    */
    virtual void startCheckpoint();

    /**
        @return the type of the Gamma discretisation, used to report how the
        category rates and weights were obtained
    */
    virtual int isGammaRate() { return GAMMA_CUT_LLOYD; }

    /**
        @return model name with parameters, e.g. +GU4{0.5}
    */
    virtual string getNameParams();

    /**
        get the proportion of sites under a specified category.
        @param category category ID from 0 to #category-1
        @return the weight of the specified Lloyd-Max bin
    */
    virtual double getProp(int category) { return prop[category]; }

    /**
        set the proportion of a specified category. NOTE: the weights are
        a deterministic function of alpha, so any value set here is
        overwritten by the next call to computeRates().
        @param category category ID from 0 to #category-1
        @param value the proportion of the specified category
    */
    virtual void setProp(int category, double value) { prop[category] = value; }

    /**
        Lloyd-Max discretisation of Gamma(alpha, alpha).
        It takes 'ncategory' and 'gamma_shape' as input. On output it writes
        to the 'rates' and 'prop' variables.
    */
    virtual void computeRates();

    /**
        set number of rate categories
        @param ncat #categories
    */
    virtual void setNCategory(int ncat);

    /**
        write information
        @param out output stream
    */
    virtual void writeInfo(ostream &out);

    /**
        @return mean rate sum_i w_i * r_i (equal to 1 up to rounding)
    */
    virtual double meanRates();

    /**
        rescale rates s.t. the mean rate is equal to 1
        @return rescaling factor
    */
    virtual double rescaleRates();

protected:

    /**
        Initial (ordered) set of representative rates for the Lloyd-Max
        iteration, following lloyd_max_init_cats() of coraxlib: the
        ncategory inner quantiles i/(ncategory+1) of a Gamma distribution
        with shape (alpha+1)/3 and rate alpha/3. This over-dispersed
        starting point spreads the representatives more widely than the
        equal-probability medians and speeds up convergence for small alpha.
        @param alpha the (bounded) Gamma shape parameter
        @return true on success, false if the quantile function failed and
                the caller should fall back to the Yang (1994) medians
    */
    bool initRates(double alpha);

    /**
        fall-back initialisation: medians of ncategory equal-probability bins
        @param alpha the (bounded) Gamma shape parameter
    */
    void initRatesMedian(double alpha);

    /**
        decision boundary between two adjacent representatives, i.e. the
        logarithmic mean (see class comment). Falls back to the arithmetic
        mean when the two rates are (numerically) equal or non-positive.
    */
    static double binBoundary(double rate_lower, double rate_upper);

    /**
        normalise the weights to sum to 1 and the rates to mean 1
    */
    void normalize();

    /**
        weight (probability mass) of each Lloyd-Max bin, ncategory elements
    */
    double *prop;

};

#endif
