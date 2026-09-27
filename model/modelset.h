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


#ifndef MODELSET_H
#define MODELSET_H

#include "modelmarkov.h"

/**
 * a set of substitution models, used eg for site-specific state frequency model or 
 * partition model with joint branch lengths
 */
class ModelSet : public ModelMarkov, public vector<ModelMarkov*>
{

public:
    ModelSet(const char *model_name, PhyloTree *tree);

    virtual bool isSiteSpecificModel() { return true; }

    virtual int getPtnModelID(size_t ptn) const;

    virtual void computeTransMatrix(double time, double *trans_matrix, int model_id = -1, int selected_row = -1);

    virtual void computeTransDerv(double time, double *trans_matrix,
                                  double *trans_derv1, double *trans_derv2, int model_id = -1);

    virtual double computeTrans(double time, int state1, int state2, int model_id = -1);

    virtual double computeTrans(double time, int state1, int state2,
                                double &derv1, double &derv2, int model_id = -1);

    virtual void getRateMatrix(double *rate_mat, int model_id = -1);

    virtual void setRateMatrix(double *rate_mat);

    virtual void getQMatrix(double *q_mat, int model_id = -1);

    virtual void setQMatrix(double *q_mat, double *freq_vec);

    virtual void getStateFrequency(double *freq_vec, int model_id = -1);

    virtual void setStateFrequency(double *freq_vec);

    virtual void adaptStateFrequency(double *freq_vec);

	/**
		return the number of dimensions
	*/
	virtual int getNDim();
	

	/**
		write information
		@param out output stream
	*/
	virtual void writeInfo(ostream &out);

	/**
		decompose the rate matrix into eigenvalues and eigenvectors
	*/
	virtual void decomposeRateMatrix();

    virtual ~ModelSet();

    /**
     * compute the memory size for the model, can be large for site-specific models
     * @return memory size required in bytes
     */
    virtual uint64_t getMemoryRequired() {
    	uint64_t mem = ModelMarkov::getMemoryRequired();
    	for (iterator it = begin(); it != end(); it++)
    		mem += (*it)->getMemoryRequired();
    	return mem;
    }

    /**
        join memory for eigen into one chunk
    */
    void joinEigenMemory();

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
		@return TRUE if parameters are changed, FALSE otherwise (2015-10-20)
	*/
	virtual bool getVariables(double *variables);

	
};

#endif // MODELSET_H
