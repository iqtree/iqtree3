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
#ifndef MODELSET_H_
#define MODELSET_H_

#include "modelmarkov.h"

/**
 *  A set of substitution models to implement site-specific models
 */
class ModelSet : public ModelMarkov, public vector<ModelMarkov*> {
public:
    ModelSet(const string model_name, ModelsBlock *models_block,
             StateFreqType freq, string freq_params, PhyloTree *tree);

    ~ModelSet();

    void setCheckpoint(Checkpoint *checkpoint) override;

    void startCheckpoint() override;

    void saveCheckpoint() override;

    void restoreCheckpoint() override;

    bool isSiteSpecificModel() override { return true; }

    bool isSSF() override { return phylo_tree->aln->isSSF(); }

    bool isSSR() override { return phylo_tree->aln->isSSR(); }

    int getTransMatrixSize() override { return num_states * num_states * size(); }

    string getName() override;

    string getNameParams(bool show_fixed_params = false) override;

    void writeInfo(ostream &out) override;

    void computeTransMatrix(double time, double *trans_matrix, int mixture = 0, int selected_row = -1) override;

    void computeTransDerv(double time, double *trans_matrix,
                          double *trans_derv1, double *trans_derv2, int mixture = 0) override;

    double computeTrans(double time, int state1, int state2) override { return 0; }

    double computeTrans(double time, int state1, int state2, double &derv1, double &derv2) override { return 0; }

    double computeTrans(double time, int model_id, int state1, int state2) override;

    double computeTrans(double time, int model_id, int state1, int state2, double &derv1, double &derv2) override;

    int getPtnModelID(int ptn) override;

    void getRateMatrix(double *rate_mat) override;

    void getStateFrequency(double *state_freq, int mixture = 0) override;

    void getQMatrix(double *q_mat, int mixture = 0) override;

    StateFreqType getFreqType() override;

    int getNDim() override;

    int getNDimFreq() override;

    bool isUnstableParameters() override;

    void setBounds(double *lower_bound, double *upper_bound, bool *bound_check) override;

    void scaleStateFreq(bool sum_one) override;

    double optimizeParameters(double gradient_epsilon) override;

    double targetFunk(double x[]) override;

    uint64_t getMemoryRequired() override;

    void decomposeRateMatrix() override;

protected:
    void setVariables(double *variables) override;

    bool getVariables(double *variables) override;

    /**
     *  Join memory for eigen of submodels into a single chunk
     */
    void joinEigenMemory();
};

#endif /* MODELSET_H_ */
