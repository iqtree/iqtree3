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
#ifndef RATEGAMMASTARINVAR_H
#define RATEGAMMASTARINVAR_H

#include "rateinvar.h"
#include "rategammaunequal.h"

/**
    class for the +I+G[n]s rate heterogeneity, i.e. invariable sites combined
    with the DGM-star model (Discrete Gamma Model-star).

    This is the RateGammaInvar analogue for the Lloyd-Max discretisation. Only
    three things differ from RateGammaInvar, all of them because the categories
    carry unequal weights prop[] instead of a uniform 1/ncategory:

      - getProp(c) is (1-p_invar) * prop[c];
      - computeRates() rescales the category rates by 1/(1-p_invar) explicitly.
        RateGammaInvar instead pre-fills rates[] with 1/(1-p_invar) and relies
        on RateGamma::computeRates() preserving that scale, which does not work
        here because RateGammaUnequal::computeRates() normalises to mean rate 1;
      - meanRates() is the prop-weighted mean.

    Everything else (EM, random restart, empirical Bayes site rates, the BFGS
    plumbing) is independent of how the weights are obtained, because the
    likelihood kernels fold getProp() into _pattern_lh_cat.

    @author BUI Quang Minh <m.bui@anu.edu.au>
*/
class RateGammaStarInvar : public RateInvar, public RateGammaUnequal
{
public:
    /**
        constructor
        @param ncat number of rate categories
        @param shape Gamma shape parameter
        @param p_invar_sites proportion of invariable sites
        @param optimize_alg optimization algorithm for p_invar and alpha
        @param tree associated phylogenetic tree
        @param testParamDone TRUE if the parameters were already tested by ModelFinder
    */
    RateGammaStarInvar(int ncat, double shape, double p_invar_sites,
                       string optimize_alg, PhyloTree *tree, bool testParamDone);

    /**
        check whether +I+G is used
    */
    virtual bool isGammai() const { return true; }

    /**
        start structure for checkpointing.
        NOTE: deliberately the same structure name as RateGammaInvar, because
        ModelFinder transfers "RateGammaInvar/gamma_shape" and
        "RateGammaInvar/p_invar" by name to seed the initial values
        (see transferSubCheckpoint() calls in phylotesting.cpp).
        RateContinuousGammaInvar does the same.
    */
    virtual void startCheckpoint();

    /**
        save object into the checkpoint
    */
    virtual void saveCheckpoint();

    /**
        restore object from the checkpoint
    */
    virtual void restoreCheckpoint();

    /**
        get the proportion of sites under a specified category.
        @param category category ID from 0 to #category-1
        @return the weight of the Lloyd-Max bin, scaled by (1-p_invar)
    */
    virtual double getProp(int category) { return (1.0-p_invar) * prop[category]; }

    /**
        @param category category ID from 0 to #category-1
        @return the rate of the specified category
    */
    virtual double getRate(int category) { return RateGammaUnequal::getRate(category); }

    /**
        Lloyd-Max discretisation, with the rates scaled such that the mean rate
        over all sites (invariable ones included) is 1.
    */
    virtual void computeRates();

    /**
        set the proportion of invariable sites
        @param pInvar the proportion of invariable sites
    */
    virtual void setPInvar(double pInvar);

    /**
        @return mean rate (1-p_invar) * sum_i prop[i]*rates[i], equal to 1 up to rounding
    */
    virtual double meanRates();

    /**
        rescale rates s.t. the mean rate is equal to 1
        @return rescaling factor
    */
    virtual double rescaleRates();

    /**
        @return model name with parameters, e.g. +I{0.2}+G4s{0.5}
    */
    virtual string getNameParams();

    /**
        override function from Optimization class, used by minimizeOneDimen() to
        optimize p_invar or the gamma shape parameter.
        @param value value of gamma shape (if cur_optimize == 0) or p_invar (if cur_optimize == 1)
    */
    virtual double computeFunction(double value);

    /**
        setup the bounds for joint optimization with BFGS
    */
    virtual void setBounds(double *lower_bound, double *upper_bound, bool *bound_check);

    /**
        optimize parameters
        @return the best likelihood
    */
    virtual double optimizeParameters(double gradient_epsilon);

    /**
        return the number of dimensions
    */
    virtual int getNDim() { return RateInvar::getNDim() + RateGammaUnequal::getNDim(); }

    /**
        the target function which needs to be optimized
        @param x the input vector x
        @return the function value at x
    */
    virtual double targetFunk(double x[]);

    /**
        write information
        @param out output stream
    */
    virtual void writeInfo(ostream &out);

    /**
        write parameters, used with modeltest
        @param out output stream
    */
    virtual void writeParameters(ostream &out);

    /**
        set number of rate categories
        @param ncat #categories
    */
    virtual void setNCategory(int ncat);

    /**
        Compute site-specific rates by empirical Bayes
        @param pattern_rates (OUT) pattern rates. Resizing if necesary
        @return total number of categories
    */
    virtual int computePatternRates(DoubleVector &pattern_rates, IntVector &pattern_cat);

protected:

    /**
        this function is served for the multi-dimension optimization. It should pack the model parameters
        into a vector that is index from 1 (NOTE: not from 0)
        @param variables (OUT) vector of variables, indexed from 1
    */
    virtual void setVariables(double *variables);

    /**
        this function is served for the multi-dimension optimization. It should assign the model parameters
        from a vector of variables that is index from 1 (NOTE: not from 0)
        @param variables vector of variables, indexed from 1
        @return TRUE if parameters are changed, FALSE otherwise
    */
    virtual bool getVariables(double *variables);

private:
    /**
        which algorithm is used to optimize p_invar and alpha
    */
    string optimize_alg;

    /**
        current parameter to optimize. 0 if gamma shape, 1 if p_invar.
    */
    int cur_optimize;

    /**
        optimize p_invar and gamma shape using the EM algorithm
    */
    double optimizeWithEM(double gradient_epsilon);

    /**
        start with different initial values of p_invar
    */
    double randomRestartOptimization(double gradient_epsilon);
};

#endif
