//
// C++ Implementation: substmodel
//
// Description:
//
//
// Author: BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "modelsubst.h"

#include "utils/tools.h"
#include <algorithm>

ModelSubst::ModelSubst(int nstates) {
    num_states = nstates;
    fixed_parameters = false;
    name = "JC";
    full_name = "JC (Jukes and Cantor, 1969)";
    freq_type = FREQ_EQUAL;
    state_freq = new double[num_states];
    std::fill_n(state_freq, num_states, 1.0 / nstates);
}

ModelSubst::~ModelSubst() {
    delete [] state_freq;
    state_freq = nullptr;
}

void ModelSubst::startCheckpoint() {
    checkpoint->startStruct("ModelSubst");
}

void ModelSubst::saveCheckpoint() {
    startCheckpoint();
    // output the frequencies in any circumstances
    CKP_ARRAY_SAVE(num_states, state_freq);
    endCheckpoint();
    CheckpointFactory::saveCheckpoint();
}

void ModelSubst::restoreCheckpoint() {
    CheckpointFactory::restoreCheckpoint();
    startCheckpoint();
    if (freq_type == FREQ_ESTIMATE && !fixed_parameters) {
        CKP_ARRAY_RESTORE(num_states, state_freq);
    }
    endCheckpoint();
    decomposeRateMatrix();
}

// The following functions directly implement the simplest Jukes-Cantor model,
// which is valid for all kinds of data (DNA, AA, MORPH, etc)

void ModelSubst::computeTransMatrix(double time, double *trans_matrix, int, int) {
    double coef = -double(num_states) / (num_states-1);
    double expt = exp(time * coef);
    double lh_non_diag = (1.0 - expt) / num_states;
    double lh_diag = 1.0 - (lh_non_diag * (num_states-1));
    for (int i = 0, k = 0; i < num_states; ++i) {
        for (int j = 0; j < num_states; ++j, ++k) {
            trans_matrix[k] = (i == j) ? lh_diag : lh_non_diag;
        }
    }
}

void ModelSubst::computeTransDerv(double time, double *trans_matrix,
                                  double *trans_derv1, double *trans_derv2, int) {
    double coef = -double(num_states) / (num_states-1);
    double expt = exp(time * coef);
    double lh_non_diag = (1.0 - expt) / num_states;
    double lh_diag = 1.0 - (lh_non_diag * (num_states-1));
    double derv1_non_diag = expt / (num_states-1);
    double derv1_diag = -expt;
    double derv2_non_diag = derv1_non_diag * coef;
    double derv2_diag = derv1_diag * coef;
    for (int i = 0, k = 0; i < num_states; ++i) {
        for (int j = 0; j < num_states; ++j, ++k) {
            trans_matrix[k] = (i == j) ? lh_diag : lh_non_diag;
            trans_derv1[k] = (i == j) ? derv1_diag : derv1_non_diag;
            trans_derv2[k] = (i == j) ? derv2_diag : derv2_non_diag;
        }
    }
}

double ModelSubst::computeTrans(double time, int state1, int state2, int) {
    double coef = -double(num_states) / (num_states-1);
    double expt = exp(time * coef);
    if (state1 != state2) {
        return (1.0 - expt) / num_states;
    }
    return (1.0 + (expt * (num_states-1))) / num_states;
}

double ModelSubst::computeTrans(double time, int state1, int state2,
                                double &derv1, double &derv2, int) {
    double coef = -double(num_states) / (num_states-1);
    double expt = exp(time * coef);
    if (state1 != state2) {
        derv1 = expt / (num_states-1);
        derv2 = derv1 * coef;
        return (1.0 - expt) / num_states;
    }
    derv1 = -expt;
    derv2 = derv1 * coef;
    return (1.0 + (expt * (num_states-1))) / num_states;
}

void ModelSubst::getRateMatrix(double *rate_mat, int) {
    int nrate = getNumRateEntries();
    std::fill_n(rate_mat, nrate, 1.0);
}

void ModelSubst::getQMatrix(double *q_mat, int) {
    double q_non_diag = 1.0 / (num_states-1);
    for (int i = 0, k = 0; i < num_states; ++i) {
        for (int j = 0; j < num_states; ++j, ++k) {
            q_mat[k] = (i == j) ? -1.0 : q_non_diag;
        }
    }
}

void ModelSubst::getStateFrequency(double *freq_vec, int) {
    std::fill_n(freq_vec, num_states, 1.0 / num_states);
}

void ModelSubst::computeTipLikelihood(PML::StateType state, double *state_lk) {
    if (state < num_states) {
        // single state
        std::fill_n(state_lk, num_states, 0.0);
        state_lk[state] = 1.0;
    } else {
        // unknown state
        std::fill_n(state_lk, num_states, 1.0);
    }
}

void ModelSubst::printMrBayesModelText(ofstream &out, string partition, string charset) {
    out << "using MrBayes model GTR+G+I]" << endl;
    out << "  [Model not supported by MrBayes, defaulting to GTR+G+I (DNA)]" << endl;
    outWarning("MrBayes output is not supported by model " + name + ", defaulting to GTR+G+I (DNA)!");
    out << "  lset applyto=(" << partition << ") nucmodel=4by4 nst=" << 6 << " rates=" << "invgamma" << ";" << endl;
}
