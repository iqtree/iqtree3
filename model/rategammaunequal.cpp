/***************************************************************************
 *   Copyright (C) 2026 by                                            *
 *   BUI Quang Minh <m.bui@anu.edu.au>                                *
 *   This code is started by Claude Code                                                                      *
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
#include "tree/phylotree.h"
#include "rategammaunequal.h"
#include <cmath>
#include <vector>

/** Gamma quantile: value x with Prob{X < x} = prob for X ~ Gamma(alpha, beta),
    beta being the rate (inverse scale). Same as coraxlib's POINT_GAMMA macro. */
static double pointGamma(double prob, double alpha, double beta) {
    return RateGamma::cmpPointChi2(prob, 2.0 * alpha) / (2.0 * beta);
}

RateGammaUnequal::RateGammaUnequal(int ncat, double shape, PhyloTree *tree)
    : RateGamma(ncat, shape, false, tree)
{
    // NOTE: the base constructor already called RateGamma::setNCategory(), which
    // filled 'rates' with the equal-weight discretisation. Redo it here (now that
    // 'prop' exists and virtual dispatch reaches this class) to obtain the
    // Lloyd-Max rates and weights.
    prop = nullptr;
    setNCategory(ncat);
}

RateGammaUnequal::~RateGammaUnequal()
{
    delete [] prop;
    prop = nullptr;
}

void RateGammaUnequal::startCheckpoint() {
    checkpoint->startStruct("RateGammaUnequal");
}

// saveCheckpoint()/restoreCheckpoint() are inherited from RateGamma: only
// gamma_shape needs to be stored, the rates and weights are recomputed from it
// by the (virtual) computeRates() called in RateGamma::restoreCheckpoint().

void RateGammaUnequal::setNCategory(int ncat) {
    ncategory = ncat;
    delete [] prop;
    delete [] rates;
    rates = new double[ncategory];
    prop  = new double[ncategory];
    for (int cat = 0; cat < ncategory; cat++) {
        rates[cat] = 1.0;
        prop[cat]  = 1.0 / ncategory;
    }
    name = "+G" + convertIntToString(ncategory) + "s";
    full_name = "Gamma with " + convertIntToString(ncategory) + " unequally weighted categories";
    computeRates();
}

string RateGammaUnequal::getNameParams() {
    ostringstream str;
    str << "+G" << ncategory << "s{" << gamma_shape << '}';
    return str.str();
}

double RateGammaUnequal::binBoundary(double rate_lower, double rate_upper) {
    // logarithmic mean: the point where the I-divergences to the two
    // representatives are equal (see class comment)
    if (rate_lower <= 0.0 || rate_upper <= 0.0)
        return 0.5 * (rate_lower + rate_upper);
    double log_diff = log(rate_upper) - log(rate_lower);
    if (fabs(log_diff) < MIN_LLOYD_MASS)
        return 0.5 * (rate_lower + rate_upper);
    return (rate_upper - rate_lower) / log_diff;
}

bool RateGammaUnequal::initRates(double alpha) {
    // over-dispersed starting point of lloyd_max_init_cats()
    double init_alpha = (alpha + 1.0) / 3.0;
    double init_beta  = alpha / 3.0;
    double ratio = 1.0 / (ncategory + 1.0);
    for (int cat = 0; cat < ncategory; cat++) {
        double rate = pointGamma((cat + 1.0) * ratio, init_alpha, init_beta);
        // cmpPointChi2() returns -1 on failure; the representatives must also be
        // strictly increasing for the boundaries below to be ordered
        if (rate <= 0.0 || (cat > 0 && rate <= rates[cat-1]))
            return false;
        rates[cat] = rate;
    }
    return true;
}

void RateGammaUnequal::initRatesMedian(double alpha) {
    for (int cat = 0; cat < ncategory; cat++) {
        double prob = (2.0 * cat + 1.0) / (2.0 * ncategory);
        rates[cat] = fabs(pointGamma(prob, alpha, alpha));
    }
}

void RateGammaUnequal::computeRates() {
    if (ncategory == 1) {
        rates[0] = 1.0;
        prop[0]  = 1.0;
        return;
    }

    // Gamma(alpha, beta) with beta == alpha, so that the mean rate is 1
    double alpha = gamma_shape;
    double min_shape = (phylo_tree && phylo_tree->params) ?
            phylo_tree->params->min_gamma_shape : MIN_GAMMA_SHAPE;
    if (alpha < min_shape)
        alpha = min_shape;
    double beta  = alpha;
    double mean  = alpha / beta;
    double lnga  = cmpLnGamma(alpha);
    double lnga1 = cmpLnGamma(alpha + 1.0);

    // Always start the iteration from scratch: the rates must be a
    // deterministic function of (alpha, ncategory), otherwise the likelihood
    // would depend on the path taken by the shape optimiser.
    if (!initRates(alpha))
        initRatesMedian(alpha);

    // boundaries of the ncategory bins. bound[0] = 0 and the upper boundary of
    // the last bin is infinity, which is handled by using a CDF value of 1.
    vector<double> bound(ncategory + 1, 0.0);

    int iter;
    for (iter = 0; iter < MAX_LLOYD_ITER; iter++) {

        // Step 1: decision boundaries from the current representatives
        bound[0] = 0.0;
        for (int cat = 1; cat < ncategory; cat++)
            bound[cat] = binBoundary(rates[cat-1], rates[cat]);

        // Step 2: bin weight and representative rate.
        //   prop[i]  = P(a <= X < b) = F_alpha(b) - F_alpha(a)
        //   rates[i] = E[X | a <= X < b]
        //            = (alpha/beta) * [F_{alpha+1}(b) - F_{alpha+1}(a)] / prop[i]
        bool converged = true;
        for (int cat = 0; cat < ncategory; cat++) {
            double prev_rate = rates[cat];
            double cdf_low, cdf_low1, cdf_up, cdf_up1;

            if (cat == 0) {
                // the Gamma CDF at 0 is 0 for any positive shape
                cdf_low = cdf_low1 = 0.0;
            } else {
                // cmpIncompleteGamma() expects beta*x for rate parameter beta
                cdf_low  = cmpIncompleteGamma(bound[cat] * beta, alpha, lnga);
                cdf_low1 = cmpIncompleteGamma(bound[cat] * beta, alpha + 1.0, lnga1);
            }

            if (cat == ncategory - 1) {
                cdf_up = cdf_up1 = 1.0;
            } else {
                cdf_up  = cmpIncompleteGamma(bound[cat+1] * beta, alpha, lnga);
                cdf_up1 = cmpIncompleteGamma(bound[cat+1] * beta, alpha + 1.0, lnga1);
            }

            double mass = cdf_up - cdf_low;
            double moment_mass = cdf_up1 - cdf_low1;
            prop[cat] = mass;

            // leave the representative unchanged if the bin is (numerically) empty
            if (mass > MIN_LLOYD_MASS)
                rates[cat] = mean * moment_mass / mass;

            double change = fabs(rates[cat] - prev_rate);
            if (fabs(prev_rate) >= MIN_LLOYD_MASS)
                change /= fabs(prev_rate);
            converged &= (change < TOL_LLOYD_RATE);
        }

        if (converged)
            break;
    }

    if (iter >= MAX_LLOYD_ITER && verbose_mode >= VB_MED)
        outWarning("Lloyd-Max discretisation of the Gamma distribution did not converge for alpha = "
                   + convertDoubleToString(alpha));

    normalize();
}

void RateGammaUnequal::normalize() {
    double sum_prop = 0.0;
    for (int cat = 0; cat < ncategory; cat++) {
        // guard against a zero-weight category
        if (prop[cat] < MIN_LLOYD_MASS)
            prop[cat] = MIN_LLOYD_MASS;
        sum_prop += prop[cat];
    }
    for (int cat = 0; cat < ncategory; cat++)
        prop[cat] /= sum_prop;

    // sum_i prop[i]*rates[i] is 1 by construction, this only removes the
    // rounding error accumulated by the iteration
    rescaleRates();
}

double RateGammaUnequal::meanRates() {
    double ret = 0.0;
    for (int cat = 0; cat < ncategory; cat++)
        ret += prop[cat] * rates[cat];
    return ret;
}

double RateGammaUnequal::rescaleRates() {
    double norm = meanRates();
    if (norm <= 0.0)
        return 1.0;
    for (int cat = 0; cat < ncategory; cat++)
        rates[cat] /= norm;
    return norm;
}

void RateGammaUnequal::writeInfo(ostream &out) {
    out << "Gamma shape alpha: " << gamma_shape << endl;
    if (verbose_mode >= VB_MED) {
        out << "Lloyd-Max categories (rate, weight):";
        for (int cat = 0; cat < ncategory; cat++)
            out << " (" << rates[cat] << ", " << prop[cat] << ")";
        out << endl;
    }
}
