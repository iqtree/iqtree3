//
// C++ Interface: substmodel
//
// Description: 
//
//
// Author: BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SUBSTMODEL_H
#define SUBSTMODEL_H

#include <string>
#include "utils/tools.h"
#include "utils/optimization.h"
#include "utils/checkpoint.h"
#include "phylo-yaml/statespace.h"

// using namespace std;

const char OPEN_BRACKET = '{';
const char CLOSE_BRACKET = '}';

class PhyloTree;

/**
Substitution model abstract class

	@author BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>
*/
class ModelSubst: public Optimization, public CheckpointFactory
{
	friend class ModelFactory;
    friend class PartitionModel;
    friend class IQTreeMix;

public:
	/**
		constructor
		@param nstates number of states, e.g. 4 for DNA, 20 for proteins.
	*/
    ModelSubst(int nstates);


	/**
		@return the number of dimensions
	*/
	virtual int getNDim() { return 0; }

	/**
		@return the number of dimensions corresponding to state frequencies
	*/
	virtual int getNDimFreq() { return 0; }
	
	/**
	 * @return model name
	 */
	virtual string getName() { return name; }

	/**
	 * @return model name with parameters in form of e.g. GTR{a,b,c,d,e,f}
	 */
	virtual string getNameParams(bool show_fixed_params = false) { return name; }

	/**
		@return TRUE if model is time-reversible, FALSE otherwise
	*/
	virtual bool isReversible() { return true; };

    /** return true if using reversible likelihood kernel, false for using non-reversible kernel */
    bool useRevKernel() {
        return isReversible() && !Params::getInstance().kernel_nonrev;
    };

    /**
        fix parameters of the model
        @param fix true to fix, false to not fix
        @return the current state of fixing parameters
     */
    virtual bool fixParameters(bool fix) {
        bool current = fixed_parameters;
        fixed_parameters = fix;
        return current;
    }
    
	/**
	 * @return TRUE if this is a site-specific model, FALSE otherwise
	 */
	virtual bool isSiteSpecificModel() { return false; }

	/**
	 * @return TRUE if this is a mixture model, FALSE otherwise
	 */
	virtual bool isMixture() { return false; }
    
    /**
     * @return TRUE if this is a liemarkov model, FALSE otherwise
     */
    virtual bool isLieMarkov() { return false; }
    
    /**
     * @return TRUE if this is a mixture model and all model components share the same rate matrix, FALSE otherwise
     */
    virtual bool isMixtureSameQ() { return false; }
    
    /**
     * @return TRUE if this is a DNA error model, FALSE otherwise
     */
    virtual bool containDNAerror() { return false; }
    
    /**
     * get the dna error probability, by default error probability = 0
     */
    virtual double getDNAErrProb(int mixture_index = 0) { return 0; }

    /** 
     * Confer to modelpomo.h.
     * 
     * @return TRUE if PoMo is being used, FALSE otherise.
     */
    virtual bool isPolymorphismAware() { return false; }

	/**
	 * @return the number of mixture model components
	 */
	virtual int getNMixtures() { return 1; }

	// initial the parameters from the (K-1)-class mixture model
	virtual void initFromClassMinusOne(double init_weight) {}
    
	/**
	 * @param cat mixture class
	 * @return weight of a mixture model component
	 */
	virtual double getMixtureWeight(int cat) { return 1.0; }

	/**
	 * @param cat mixture class
	 * @return weight of a mixture model component
	 */
	virtual void setMixtureWeight(int cat, double weight) {}

	/**
	 * @param cat mixture class
	 * @return weight of a mixture model component
	 */
	virtual void setFixMixtureWeight(bool fix_prop) {}

	/**
	 * @param cat mixture class ID
	 * @return corresponding mixture model component
	 */
    virtual ModelSubst* getMixtureClass(int cat) { return nullptr; }

	/**
	 * @param cat mixture class ID
	 * @param m mixture model class to set
	 */
    virtual void setMixtureClass(int cat, ModelSubst* m) { }

	/**
		@return the number of rate entries, equal to the number of elements
			in the upper-diagonal of the rate matrix (since model is reversible)
	*/
	virtual int getNumRateEntries() { return num_states*(num_states-1)/2; }
    
    /**
     set num_params variable
     */
    virtual void setNParams(int num_params) {}
    
    /**
     get num_params variable
     */
    virtual int getNParams() {
        return 0;
    }

    /**
     *  Map alignment patterns to submodels (for site-specific models)
     *  @param ptn ID of an alignment pattern
     *  @return ID of the corresponding model
     */
    virtual int getPtnModelID(size_t ptn) const { return -1; }

    /**
     *  Compute the transition probability matrix on a branch.
     *  The default is the JC model, valid for all kinds of data
     *  @param time The branch length
     *  @param model_id ID of a submodel (for mixture and site-specific models)
     *  @param selected_row Only compute the entries for the selected row,
     *                      the default is to compute entries for all rows
     *  @param[out] trans_matrix The transition matrix between all pairs of states,
     *                           assumed to have the size of num_states*num_states
     */
    virtual void computeTransMatrix(double time, double *trans_matrix, int model_id = -1, int selected_row = -1);

    /**
     *  The same as computeTransMatrix() above, but also computes the
     *  1st and 2nd derivative matrices with respect to the branch length
     *  @param[out] trans_matrix The transition matrix between all pairs of states,
     *                           assumed to have the size of num_states*num_states
     *  @param[out] trans_derv1 The 1st derivative matrix between all pairs of states
     *  @param[out] trans_derv2 The 2nd derivative matrix between all pairs of states
     */
    virtual void computeTransDerv(double time, double *trans_matrix,
                                  double *trans_derv1, double *trans_derv2, int model_id = -1);

    /**
     *  Compute the transition probability between the two states on a branch.
     *  The default is the JC model, valid for all kinds of data
     *  @param time The branch length between the two states
     *  @param model_id ID of a submodel (for mixture and site-specific models)
     *  @param state1 The start state
     *  @param state2 The end state
     *  @param[out] derv1 The 1st derivative
     *  @param[out] derv2 The 2nd derivative
     *  @return The transition probability
     */
    virtual double computeTrans(double time, int state1, int state2, int model_id = -1);

    /**
     *  The same as computeTrans() above, but also computes the
     *  1st and 2nd derivatives with respect to the branch length
     *  @param[out] derv1 The 1st derivative
     *  @param[out] derv2 The 2nd derivative
     *  @return The transition probability
     */
    virtual double computeTrans(double time, int state1, int state2,
                                double &derv1, double &derv2, int model_id = -1);

    /**
     *  Get the rate parameters, such as a,b,c,d,e,f for a DNA model.
     *  Get the above-diagonal entries of the rate matrix, assuming that
     *  the last element is 1.
     *  The default is equal rates of 1 (JC Model), valid for all kinds of data
     *  @param[out] rate_mat An upper-triangle rate matrix, assumed to have the
     *                       size of num_states*(num_states-1)/2
     *  @param model_id ID of a submodel (for mixture and site-specific models)
     */
    virtual void getRateMatrix(double *rate_mat, int model_id = -1);

    /**
     *  Get the instantaneous rate matrix Q.
     *  The default is derived from equal rates and equal state frequencies
     *  @param[out] q_mat A full matrix: qij >= 0, qii = -sum_j qij (j != i),
     *                    assumed to have the size of num_states*num_states
     *  @param model_id ID of a submodel (for mixture and site-specific models)
     */
    virtual void getQMatrix(double *q_mat, int model_id = -1);

    /**
     *  Get the state frequency vector.
     *  The default is equal state frequencies, valid for all kinds of data
     *  @param[out] freq_vec A state frequency vector, assumed to have the
     *                       size of num_states
     *  @param model_id ID of a submodel (for mixture and site-specific models)
     */
    virtual void getStateFrequency(double *freq_vec, int model_id = -1);

    /**
     *  Set the state frequency vector
     *  @param freq_vec A state frequency vector, assumed to have the
     *                  size of num_states
     */
    virtual void setStateFrequency(double *freq_vec) {}

	/**
		get frequency type
		@return frequency type
	*/
	virtual StateFreqType getFreqType() { return FREQ_EQUAL; }

    /**
        set the associated tree
        @param tree the associated tree
    */
    virtual void setTree(PhyloTree *tree) {}

    /**
     *  For reversible models, multiply the partial likelihood vector with
     *  the matrix of inverse eigenvectors for the fast pruning algorithm
     *  @param[in/out] state_lh The partial likelihood vector
     */
    virtual void multiplyWithInvEigenvector(double *state_lh) {}

    /** compute the tip likelihood vector of a state for Felsenstein's pruning algorithm
     @param state character state
     @param[out] state_lk state likehood vector of size num_states
     */
    virtual void computeTipLikelihood(PML::StateType state, double *state_lk);

	/**
		decompose the rate matrix into eigenvalues and eigenvectors
	*/
	virtual void decomposeRateMatrix() {}


    /** 
        set number of optimization steps
        @param opt_steps number of optimization steps
    */
    virtual void setOptimizeSteps(int optimize_steps) { }

	/**
		optimize model parameters. One should override this function when defining new model.
		The default does nothing since it is a Juke-Cantor type model, hence no parameters involved.
		@param epsilon accuracy of the parameters during optimization
		@return the best likelihood 
	*/
	virtual double optimizeParameters(double gradient_epsilon) { return 0.0; }

	/**
	 * @return TRUE if parameters are at the boundary that may cause numerical unstability
	 */
	virtual bool isUnstableParameters() { return false; }

	/**
		write information
		@param out output stream
	*/
	virtual void writeInfo(ostream &out) {}

	/**
		report model
		@param out output stream
	*/
    virtual void report(ostream &out) {}

	virtual double *getEigenvalues() const {
		return nullptr;
	}

	virtual double *getEigenvectors() const {
		return nullptr;
	}

	virtual double *getInverseEigenvectors() const {
		return nullptr;
	}

    virtual double *getInverseEigenvectorsTransposed() const {
        return nullptr;
    }

    
    /**
     * compute the memory size for the model, can be large for site-specific models
     * @return memory size required in bytes
     */
    virtual uint64_t getMemoryRequired() {
    	return num_states*sizeof(double);
    }
    
    /** @return true if model is a mixture model and it's fused with site_rate */
    virtual bool isFused(){
        return false;
    };

    /**
    * get the underlying mutation model, used with PoMo model
    */
    virtual ModelSubst *getMutationModel() { return this; }

    /**
     * Print the model information in a format that can be accepted by MrBayes, using lset and prset.<br>
     * By default, it simply prints a warning to the log and to the stream, stating that this model is not supported by MrBayes.
     * @param out the ofstream to print the result to
     * @param partition the partition to apply lset and prset to
     * @param charset the current partition's charset.
     */
    virtual void printMrBayesModelText(ofstream& out, string partition, string charset);

	/*****************************************************
		Checkpointing facility
	*****************************************************/

    /**
        start structure for checkpointing
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
		number of states
	*/
	int num_states;

	/**
		name of the model
	*/
	string name;


	/**
		full name of the model
	*/
	string full_name;
    
    /** true to fix parameters, otherwise false */
    bool fixed_parameters;

	/**
	 state frequencies
	 */
	double *state_freq;
	

	/**
		state frequency type
	*/
	StateFreqType freq_type;

    /** state set for each sequence in the alignment */
    //vector<vector<int> > seq_states;

	/**
		destructor
	*/
    virtual ~ModelSubst();

protected:

	/**
		this function is served for the multi-dimension optimization. It should pack the model parameters
		into a vector that is index from 1 (NOTE: not from 0)
		@param variables (OUT) vector of variables, indexed from 1
	*/
	virtual void setVariables(double *variables) {}

	/**
		this function is served for the multi-dimension optimization. It should assign the model parameters
		from a vector of variables that is index from 1 (NOTE: not from 0)
		@param variables vector of variables, indexed from 1
		@return TRUE if parameters are changed, FALSE otherwise (2015-10-20)
	*/
	virtual bool getVariables(double *variables) { return false; }

    
};

#endif
