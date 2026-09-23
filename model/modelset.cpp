/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2012  BUI Quang Minh <email>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "modelset.h"
#include "modelmixture.h"

ModelSet::ModelSet(const string model_name, ModelsBlock *models_block,
                   StateFreqType freq, string freq_params, PhyloTree *tree)
: ModelMarkov(tree) {
    name = full_name = model_name;
    full_name += "+site-specific state frequency or rate model (unpublished)";
    // init the wrapper model to use its eigen
    ModelMarkov::init(FREQ_EMPIRICAL); // +F is used here as default under SSF
    ModelMarkov::fixParameters(true);  // yet otherwise the wrapper model parameters remain unused
    // init submodels
    ASSERT(freq != FREQ_MIXTURE);
    if (isSSF()) { // default freqs for unspecified sites under SSF
        freq = FREQ_EMPIRICAL;
        freq_params = "";
    }
    double *state_freqs = new double[num_states];
    double *rate_mat = new double[getNumRateEntries()];
    for (size_t ptn = 0; ptn < phylo_tree->aln->getNPattern(); ++ptn) {
        ModelMarkov *submodel;
        if (ptn == 0) { // the front submodel, other submodels will take after it
            submodel = (ModelMarkov*)createModel(model_name, models_block, freq, freq_params, tree);
            if (!submodel->isReversible()) {
                outError("Non-reversible models are incompatible with site-specific models");
            }
            freq = submodel->getFreqType();
            submodel->getStateFrequency(state_freqs);
            submodel->getRateMatrix(rate_mat);
        } else {
            submodel = (ModelMarkov*)createModel(model_name, models_block, FREQ_EQUAL, "", tree);
            submodel->setFreqType(freq);
            submodel->setStateFrequency(state_freqs);
            submodel->setRateMatrix(rate_mat);
        }
        // set site-specific freqs (if any) and re-init
        if (isSSF()) {
            if (phylo_tree->aln->ptn_state_freq[ptn]) {
                submodel->setStateFrequency(phylo_tree->aln->ptn_state_freq[ptn]);
            } // else: unspecified site, continue with the default freqs
            submodel->init(FREQ_USER_DEFINED);
        }
        push_back(submodel);
    }
    delete [] state_freqs;
    delete [] rate_mat;
    // normalize site-specific rates (if any) and rescale the tree
    if (isSSR()) {
        double mean_rate = phylo_tree->aln->normalizePtnRateScaler();
        if (mean_rate != 1.0) {
            phylo_tree->scaleLength(mean_rate);
            phylo_tree->clearAllPartialLH();
        }
    }
    // generate the site-specific eigen of the wrapper model
    joinEigenMemory();
    decomposeRateMatrix();
}

ModelSet::~ModelSet() {
    for (reverse_iterator rit = rbegin(); rit != rend(); ++rit) {
        (*rit)->eigenvalues = nullptr;
        (*rit)->eigenvectors = nullptr;
        (*rit)->inv_eigenvectors = nullptr;
        (*rit)->inv_eigenvectors_transposed = nullptr;
        delete (*rit);
    }
}

void ModelSet::setCheckpoint(Checkpoint *checkpoint) {
    CheckpointFactory::setCheckpoint(checkpoint);
    front()->setCheckpoint(checkpoint);
}

void ModelSet::startCheckpoint() {
    checkpoint->startStruct("ModelSet");
}

void ModelSet::saveCheckpoint() {
    startCheckpoint();
    checkpoint->startStruct("frontModel");
    front()->saveCheckpoint();
    checkpoint->endStruct();
    endCheckpoint();
}

void ModelSet::restoreCheckpoint() {
    startCheckpoint();
    checkpoint->startStruct("frontModel");
    front()->restoreCheckpoint();
    checkpoint->endStruct();
    endCheckpoint();
    if (getNDim() > 0) {
        double *state_freqs = new double[num_states];
        double *rate_mat = new double[getNumRateEntries()];
        getStateFrequency(state_freqs);
        getRateMatrix(rate_mat);
        for (iterator it = begin(); it != end(); ++it) {
            if (getFreqType() == FREQ_ESTIMATE) {
                (*it)->setStateFrequency(state_freqs);
            }
            (*it)->setRateMatrix(rate_mat);
        }
        delete [] state_freqs;
        delete [] rate_mat;
    }
    decomposeRateMatrix();
    if (phylo_tree) {
        phylo_tree->clearAllPartialLH();
    }
}

string ModelSet::getName() {
    if (isSSF()) {
        return name + "+SSF";
    } else {
        return front()->getName();
    }
}

string ModelSet::getNameParams(bool show_fixed_params) {
    ostringstream retname;
    retname << name;
    if (!front()->fixed_parameters && front()->num_params > 0) {
        double *rate_mat = front()->rates;
        retname << '{';
        int nrates = getNumRateEntries();
        for (int i = 0; i < nrates; ++i) {
            if (i > 0) retname << ',';
            retname << rate_mat[i];
        }
        retname << '}';
    }
    if (isSSF()) {
        retname << "+SSF";
    } else {
        front()->getNameParamsFreq(retname);
    }
    return retname.str();
}

void ModelSet::writeInfo(ostream &out) {
    if (empty()) {
        return;
    }
    if (verbose_mode >= VB_DEBUG) {
        out << "Wrapper model:" << endl;
        ModelMarkov::writeInfo(out);
        int i = 1;
        for (iterator it = begin(); it != end(); ++it, ++i) {
            out << "Partition " << i << ":" << endl;
            (*it)->writeInfo(out);
        }
    } else {
        front()->writeInfo(out);
    }
}

void ModelSet::computeTransMatrix(double time, double *trans_matrix, int mixture, int selected_row) {
    // TODO not working with vectorization
    ASSERT(0);
    for (iterator it = begin(); it != end(); ++it) {
        (*it)->computeTransMatrix(time, trans_matrix, mixture, selected_row);
        trans_matrix += (num_states * num_states);
    }
}

void ModelSet::computeTransDerv(double time, double *trans_matrix, double *trans_derv1, double *trans_derv2, int mixture) {
    // TODO not working with vectorization
    ASSERT(0);
    for (iterator it = begin(); it != end(); ++it) {
        (*it)->computeTransDerv(time, trans_matrix, trans_derv1, trans_derv2, mixture);
        trans_matrix += (num_states * num_states);
        trans_derv1 += (num_states * num_states);
        trans_derv2 += (num_states * num_states);
    }
}

int ModelSet::getPtnModelID(int ptn) {
    ASSERT(ptn >= 0 && ptn < size());
    return ptn;
}

double ModelSet::computeTrans(double time, int model_id, int state1, int state2) {
    if (phylo_tree->vector_size == 1) {
        return at(model_id)->computeTrans(time, state1, state2);
    }
    // temporary fix problem with vectorized eigenvectors
    int vsize = phylo_tree->vector_size;
    int nstates = num_states;
    int nstates2 = num_states*num_states;
    int nstates_vsize = num_states*vsize;
    int model_vec_id = model_id % vsize;
    int start_ptn = model_id - model_vec_id;
    double *eval = &eigenvalues[start_ptn*nstates + model_vec_id];
    double *evec = &eigenvectors[start_ptn*nstates2 + model_vec_id + state1*nstates_vsize];
    double *inv_evec = &inv_eigenvectors[start_ptn*nstates2 + model_vec_id + state2*vsize];
    double trans_prob = 0.0;
    for (int i = 0; i < nstates_vsize; i += vsize) {
        double val = eval[i];
        double trans = evec[i] * inv_evec[i*nstates] * exp(time * val);
        trans_prob += trans;
    }
    return trans_prob;
}

double ModelSet::computeTrans(double time, int model_id, int state1, int state2, double &derv1, double &derv2) {
    if (phylo_tree->vector_size == 1) {
        return at(model_id)->computeTrans(time, state1, state2, derv1, derv2);
    }
    // temporary fix problem with vectorized eigenvectors
    int vsize = phylo_tree->vector_size;
    int nstates = num_states;
    int nstates2 = num_states*num_states;
    int nstates_vsize = num_states*vsize;
    int model_vec_id = model_id % vsize;
    int start_ptn = model_id - model_vec_id;
    double *eval = &eigenvalues[start_ptn*nstates + model_vec_id];
    double *evec = &eigenvectors[start_ptn*nstates2 + model_vec_id + state1*nstates_vsize];
    double *inv_evec = &inv_eigenvectors[start_ptn*nstates2 + model_vec_id + state2*vsize];
    double trans_prob = 0.0;
    derv1 = derv2 = 0.0;
    for (int i = 0; i < nstates_vsize; i += vsize) {
        double val = eval[i];
        double trans = evec[i] * inv_evec[i*nstates] * exp(time * val);
        double trans2 = trans * val;
        trans_prob += trans;
        derv1 += trans2;
        derv2 += trans2 * val;
    }
    return trans_prob;
}

void ModelSet::getRateMatrix(double *rate_mat) {
    front()->getRateMatrix(rate_mat);
}

void ModelSet::getStateFrequency(double *state_freq, int mixture) {
    if (isSSF()) {
        ASSERT(mixture >= -1);
        if (mixture >= 0) {
            at(mixture)->getStateFrequency(state_freq);
            return;
        }
        // default: return the +F freqs across all patterns
        ModelMarkov::getStateFrequency(state_freq);
    } else {
        front()->getStateFrequency(state_freq);
    }
}

void ModelSet::getQMatrix(double *q_mat, int mixture) {
    if (isSSF()) { // get the +F derived QMatrix under SSF
        ModelMarkov::getQMatrix(q_mat);
    } else {
        front()->getQMatrix(q_mat);
    }
}

StateFreqType ModelSet::getFreqType() {
    if (isSSF()) { // get the +F type under SSF
        return ModelMarkov::getFreqType();
    } else {
        return front()->getFreqType();
    }
}

int ModelSet::getNDim() {
    return front()->getNDim();
}

int ModelSet::getNDimFreq() {
    return front()->getNDimFreq();
}

bool ModelSet::isUnstableParameters() {
    return front()->isUnstableParameters();
}

void ModelSet::setBounds(double *lower_bound, double *upper_bound, bool *bound_check) {
    front()->setBounds(lower_bound, upper_bound, bound_check);
}

void ModelSet::setVariables(double *variables) {
    front()->setVariables(variables);
}

bool ModelSet::getVariables(double *variables) {
    bool changed = false;
    for (iterator it = begin(); it != end(); ++it) {
        changed |= (*it)->getVariables(variables);
    }
    return changed;
}

void ModelSet::scaleStateFreq(bool sum_one) {
    for (iterator it = begin(); it != end(); ++it) {
        (*it)->scaleStateFreq(sum_one);
    }
}

double ModelSet::optimizeParameters(double gradient_epsilon) {
    if (verbose_mode >= VB_MAX) {
        cout << "Optimizing " << getName() << " model parameters..." << endl;
    }
    int ndim = getNDim();
    if (ndim == 0) {
        return 0.0; // return if nothing to be optimized
    }
    double score;
    // optimize with the BFGS algorithm
    double *variables = new double[ndim+1];
    double *upper_bound = new double[ndim+1];
    double *lower_bound = new double[ndim+1];
    bool *bound_check = new bool[ndim+1];
    setVariables(variables);
    setBounds(lower_bound, upper_bound, bound_check);
    score = -minimizeMultiDimen(variables, ndim, lower_bound, upper_bound, bound_check, max(gradient_epsilon, TOL_RATE));
    bool changed = getVariables(variables);
    delete [] bound_check;
    delete [] lower_bound;
    delete [] upper_bound;
    delete [] variables;
    // BQM 2015-09-07: normalize state_freq
    if (is_reversible && getFreqType() == FREQ_ESTIMATE) {
        scaleStateFreq(true);
        changed = true;
    }
    if (changed || score == -1.0e+30) {
        decomposeRateMatrix();
        phylo_tree->clearAllPartialLH();
        score = phylo_tree->computeLikelihood();
    }
    return score;
}

double ModelSet::targetFunk(double x[]) {
    bool changed = getVariables(x);
    if (changed) {
        decomposeRateMatrix();
        phylo_tree->clearAllPartialLH();
    }
    // avoid numerical issue if state_freq is too small
    double *state_freqs = (isSSF()) ? this->state_freq : front()->state_freq;
    for (int x = 0; x < num_states; ++x) {
        if (state_freqs[x] < Params::getInstance().min_state_freq) {
            return 1.0e+30;
        }
    }
    return -phylo_tree->computeLikelihood();
}

uint64_t ModelSet::getMemoryRequired() {
    uint64_t mem = ModelMarkov::getMemoryRequired();
    for (iterator it = begin(); it != end(); ++it) {
        mem += (*it)->getMemoryRequired();
    }
    return mem;
}

void ModelSet::decomposeRateMatrix() {
    if (empty()) {
        return;
    }
    size_t nstates = num_states;
    // decompose for each submodel, values go to the wrapper model eigen
    for (iterator it = begin(); it != end(); ++it) {
        (*it)->decomposeRateMatrix();
    }
    // set site-specific rates (if any)
    if (isSSR()) {
        // multiply eigenvalues of each submodel with the submodel rate scaler
        for (size_t ptn = 0; ptn < size(); ++ptn) {
            ASSERT(phylo_tree->aln->ptn_rate_scaler[ptn] > 0.0);
            double rate_scaler = phylo_tree->aln->ptn_rate_scaler[ptn];
            double *eval_ptr = &eigenvalues[ptn*nstates];
            for (size_t x = 0; x < nstates; ++x) {
                eval_ptr[x] *= rate_scaler;
            }
        }
    }
    if (phylo_tree->vector_size == 1) {
        return;
    }
    // else rearrange eigen to obey vector_size
    // e.g. if vector_size == 2, then rearrange eigenvalues:
    // from: m1_1 ... m1_20    m2_1 ... m2_20    m3_1 ... m3_20    m4_1 ... m4_20
    // to:   m1_1 m2_1 ... m1_20 m2_20    m3_1 m4_1 ... m3_20 m4_20
    size_t vsize = phylo_tree->vector_size;
    size_t nstates2 = num_states*num_states;
    size_t nmodels = get_safe_upper_limit(size());
    // copy the last submodel eigen to the dummy positions
    for (size_t m = size(); m < nmodels; ++m) {
        memcpy(&eigenvalues[m*nstates], &eigenvalues[(m-1)*nstates], sizeof(double)*nstates);
        memcpy(&eigenvectors[m*nstates2], &eigenvectors[(m-1)*nstates2], sizeof(double)*nstates2);
        memcpy(&inv_eigenvectors[m*nstates2], &inv_eigenvectors[(m - 1)*nstates2], sizeof(double)*nstates2);
        memcpy(&inv_eigenvectors_transposed[m*nstates2], &inv_eigenvectors_transposed[(m-1)*nstates2], sizeof(double)*nstates2);
    }
    // make rearranged eigen for each submodel block
    double new_eval[nstates*vsize];
    double new_evec[nstates2*vsize];
    double new_inv_evec[nstates2*vsize];
    for (size_t ptn = 0; ptn < nmodels; ptn += vsize) {
        // handle a block with vsize submodels
        double *eval_ptr = &eigenvalues[ptn*nstates];
        double *evec_ptr = &eigenvectors[ptn*nstates2];
        double *inv_evec_ptr = &inv_eigenvectors[ptn*nstates2];
        for (size_t i = 0; i < vsize; ++i) {
            // handle a submodel
            for (size_t x = 0; x < nstates; ++x) {
                new_eval[x*vsize+i] = eval_ptr[x];
            }
            for (size_t x = 0; x < nstates2; ++x) {
                new_evec[x*vsize+i] = evec_ptr[x];
                new_inv_evec[x*vsize+i] = inv_evec_ptr[x];
            }
            eval_ptr += nstates;
            evec_ptr += nstates2;
            inv_evec_ptr += nstates2;
        }
        // copy new values to the wrapper model eigen
        memcpy(&eigenvalues[ptn*nstates], new_eval, sizeof(double)*nstates*vsize);
        memcpy(&eigenvectors[ptn*nstates2], new_evec, sizeof(double)*nstates2*vsize);
        memcpy(&inv_eigenvectors[ptn*nstates2], new_inv_evec, sizeof(double)*nstates2*vsize);
        calculateSquareMatrixTranspose(new_inv_evec, nstates, &inv_eigenvectors_transposed[ptn*nstates2]);
    }
}

void ModelSet::joinEigenMemory() {
    size_t nstates = num_states;
    size_t nstates2 = num_states*num_states;
    size_t nmodels = get_safe_upper_limit(size());
    aligned_free(eigenvalues);
    aligned_free(eigenvectors);
    aligned_free(inv_eigenvectors);
    aligned_free(inv_eigenvectors_transposed);
    eigenvalues = aligned_alloc<double>(nmodels*nstates);
    eigenvectors = aligned_alloc<double>(nmodels*nstates2);
    inv_eigenvectors = aligned_alloc<double>(nmodels*nstates2);
    inv_eigenvectors_transposed = aligned_alloc<double>(nmodels*nstates2);
    // assigning memory for individual submodels
    size_t m = 0;
    for (iterator it = begin(); it != end(); ++it, ++m) {
        // first copy the submodel eigen to the wrapper model
        memcpy(&eigenvalues[m*nstates], (*it)->eigenvalues, sizeof(double)*nstates);
        memcpy(&eigenvectors[m*nstates2], (*it)->eigenvectors, sizeof(double)*nstates2);
        memcpy(&inv_eigenvectors[m*nstates2], (*it)->inv_eigenvectors, sizeof(double)*nstates2);
        memcpy(&inv_eigenvectors_transposed[m*nstates2], (*it)->inv_eigenvectors_transposed, sizeof(double)*nstates2);
        // then delete the submodel eigen
        aligned_free((*it)->eigenvalues);
        aligned_free((*it)->eigenvectors);
        aligned_free((*it)->inv_eigenvectors);
        aligned_free((*it)->inv_eigenvectors_transposed);
        // and assign the submodel pointers to the wrapper model eigen
        (*it)->eigenvalues = &eigenvalues[m*nstates];
        (*it)->eigenvectors = &eigenvectors[m*nstates2];
        (*it)->inv_eigenvectors = &inv_eigenvectors[m*nstates2];
        (*it)->inv_eigenvectors_transposed = &inv_eigenvectors_transposed[m*nstates2];
    }
    // copy the last submodel eigen to the dummy positions
    for (size_t m = size(); m < nmodels; ++m) {
        memcpy(&eigenvalues[m*nstates], &eigenvalues[(m-1)*nstates], sizeof(double)*nstates);
        memcpy(&eigenvectors[m*nstates2], &eigenvectors[(m-1)*nstates2], sizeof(double)*nstates2);
        memcpy(&inv_eigenvectors[m*nstates2], &inv_eigenvectors[(m-1)*nstates2], sizeof(double)*nstates2);
        memcpy(&inv_eigenvectors_transposed[m*nstates2], &inv_eigenvectors_transposed[(m-1)*nstates2], sizeof(double)*nstates2);
    }
}
