//
//  modelcodonmixture.h
//  iqtree
//
//  Created by Minh Bui on 4/3/2025.
//

#ifndef modelcodonmixture_h
#define modelcodonmixture_h

#include <stdio.h>
#include "modelcodon.h"
#include "modelmixture.h"

/** Codon Mixture models like M2a, M3, M7, M8, notations following PAML package */
class ModelCodonMixture : public ModelMixture {
public:
    
    /**
        constructor
        @param model_name model name, e.g., CM2a, CM3
        @param freq state frequency type
        @param tree associated phylogenetic tree
    */
    ModelCodonMixture(string orig_model_name, string model_name, ModelsBlock *models_block,
            StateFreqType freq, string freq_params, PhyloTree *tree, bool optimize_weights);

    /**
     * destructor
     */
    virtual ~ModelCodonMixture();

private:

    bool link_kappa = true;

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

#endif /* modelcodonmixture_h */
