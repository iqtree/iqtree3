#include <vectorclass/vectorclass.h>
#include <vectorclass/vectormath_exp.h>
#include "phylokernel.h"
//#include "phylokernelsafe.h"
//#include "phylokernelmixture.h"
//#include "phylokernelmixrate.h"
//#include "phylokernelsitemodel.h"

#include "phylokernelnew.h"
#include "phylokernelnonrev.h"
#define KERNEL_FIX_STATES
#include "phylokernelnew.h"
#include "phylokernelnonrev.h"


#if !defined(NOSSE)
#error "You must compile this file with NOSSE!"
#endif

void PhyloTree::setParsimonyKernelSSE() {
    if (cost_matrix) {
        // Sankoff kernel
        computeParsimonyBranchPointer = &PhyloTree::computeParsimonyBranchSankoffSIMD<Vec1ui>;
        computePartialParsimonyPointer = &PhyloTree::computePartialParsimonySankoffSIMD<Vec1ui>;
        return;
    }
    // Fitch kernel
    computeParsimonyBranchPointer = &PhyloTree::computeParsimonyBranchFastSIMD<Vec1ui>;
    computePartialParsimonyPointer = &PhyloTree::computePartialParsimonyFastSIMD<Vec1ui>;
}

void PhyloTree::setDotProductSSE() {
#ifdef BOOT_VAL_FLOAT
    dotProduct = &PhyloTree::dotProductSIMD<float, Vec1f>;
#else
    dotProduct = &PhyloTree::dotProductSIMD<double, Vec1d>;
#endif
    dotProductDouble = &PhyloTree::dotProductSIMD<double, Vec1d>;
}

void PhyloTree::setLikelihoodKernelSSE() {
    vector_size = 2;
    bool site_model = model_factory && model_factory->model->isSiteSpecificModel();

    if (site_model && ((model_factory && !model_factory->model->isReversible()) || params->kernel_nonrev))
        outError("Site-specific model is not yet supported for nonreversible models");

    setParsimonyKernelSSE();
    computeLikelihoodDervMixlenPointer = NULL;

    if (site_model && safe_numeric) {
        switch (aln->num_states) {
            case 4:
                computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchSIMD    <Vec1d, SAFE_LH, 4, false, true>;
                computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervSIMD      <Vec1d, SAFE_LH, 4, false, true>;
                computePartialLikelihoodPointer    =  &PhyloTree::computePartialLikelihoodSIMD  <Vec1d, SAFE_LH, 4, false, true>;
                computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferSIMD<Vec1d, 4, false, true>;
                break;
            case 20:
                computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchSIMD    <Vec1d, SAFE_LH, 20, false, true>;
                computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervSIMD      <Vec1d, SAFE_LH, 20, false, true>;
                computePartialLikelihoodPointer    = &PhyloTree::computePartialLikelihoodSIMD   <Vec1d, SAFE_LH, 20, false, true>;
                computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferSIMD<Vec1d, 20, false, true>;
                break;
            default:
                computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchGenericSIMD    <Vec1d, SAFE_LH, false, true>;
                computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervGenericSIMD      <Vec1d, SAFE_LH, false, true>;
                computePartialLikelihoodPointer    = &PhyloTree::computePartialLikelihoodGenericSIMD   <Vec1d, SAFE_LH, false, true>;
                computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferGenericSIMD<Vec1d, false, true>;
                break;
        }
        return;
    }

    if (site_model) {
        switch (aln->num_states) {
            case 4:
                computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchSIMD    <Vec1d, NORM_LH, 4, false, true>;
                computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervSIMD      <Vec1d, NORM_LH, 4, false, true>;
                computePartialLikelihoodPointer    =  &PhyloTree::computePartialLikelihoodSIMD  <Vec1d, NORM_LH, 4, false, true>;
                computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferSIMD<Vec1d, 4, false, true>;
                break;
            case 20:
                computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchSIMD    <Vec1d, NORM_LH, 20, false, true>;
                computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervSIMD      <Vec1d, NORM_LH, 20, false, true>;
                computePartialLikelihoodPointer    = &PhyloTree::computePartialLikelihoodSIMD   <Vec1d, NORM_LH, 20, false, true>;
                computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferSIMD<Vec1d, 20, false, true>;
                break;
            default:
                ASSERT(0);
                break;
        }
        return;
    }

    if ((model_factory && !model_factory->model->isReversible()) || params->kernel_nonrev) {
        // if nonreversible model
        if (safe_numeric) {
            switch (aln->num_states) {
                case 4:
                    computeLikelihoodBranchPointer  = &PhyloTree::computeNonrevLikelihoodBranchSIMD <Vec1d, SAFE_LH, 4>;
                    computeLikelihoodDervPointer    = &PhyloTree::computeNonrevLikelihoodDervSIMD   <Vec1d, SAFE_LH, 4>;
                    computePartialLikelihoodPointer = &PhyloTree::computeNonrevPartialLikelihoodSIMD<Vec1d, SAFE_LH, 4>;
                    break;
                default:
                    computeLikelihoodBranchPointer  = &PhyloTree::computeNonrevLikelihoodBranchGenericSIMD <Vec1d, SAFE_LH>;
                    computeLikelihoodDervPointer    = &PhyloTree::computeNonrevLikelihoodDervGenericSIMD   <Vec1d, SAFE_LH>;
                    computePartialLikelihoodPointer = &PhyloTree::computeNonrevPartialLikelihoodGenericSIMD<Vec1d, SAFE_LH>;
                    break;
            }
        } else {
            switch (aln->num_states) {
                case 4:
                    computeLikelihoodBranchPointer  = &PhyloTree::computeNonrevLikelihoodBranchSIMD <Vec1d, NORM_LH, 4>;
                    computeLikelihoodDervPointer    = &PhyloTree::computeNonrevLikelihoodDervSIMD   <Vec1d, NORM_LH, 4>;
                    computePartialLikelihoodPointer = &PhyloTree::computeNonrevPartialLikelihoodSIMD<Vec1d, NORM_LH, 4>;
                    break;
                default:
                    computeLikelihoodBranchPointer  = &PhyloTree::computeNonrevLikelihoodBranchGenericSIMD <Vec1d, NORM_LH>;
                    computeLikelihoodDervPointer    = &PhyloTree::computeNonrevLikelihoodDervGenericSIMD   <Vec1d, NORM_LH>;
                    computePartialLikelihoodPointer = &PhyloTree::computeNonrevPartialLikelihoodGenericSIMD<Vec1d, NORM_LH>;
                    break;
            }
        }
        computeLikelihoodFromBufferPointer = NULL;
        return;
    }

    if (safe_numeric) {
        switch(aln->num_states) {
            case 4:
                computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchSIMD    <Vec1d, SAFE_LH, 4>;
                computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervSIMD      <Vec1d, SAFE_LH, 4>;
                computeLikelihoodDervMixlenPointer = &PhyloTree::computeLikelihoodDervMixlenSIMD<Vec1d, SAFE_LH, 4>;
                computePartialLikelihoodPointer    = &PhyloTree::computePartialLikelihoodSIMD   <Vec1d, SAFE_LH, 4>;
                computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferSIMD<Vec1d, 4>;
                break;
            case 20:
                computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchSIMD    <Vec1d, SAFE_LH, 20>;
                computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervSIMD      <Vec1d, SAFE_LH, 20>;
                computeLikelihoodDervMixlenPointer = &PhyloTree::computeLikelihoodDervMixlenSIMD<Vec1d, SAFE_LH, 20>;
                computePartialLikelihoodPointer    = &PhyloTree::computePartialLikelihoodSIMD   <Vec1d, SAFE_LH, 20>;
                computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferSIMD<Vec1d, 20>;
                break;
            default:
                computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchGenericSIMD    <Vec1d, SAFE_LH>;
                computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervGenericSIMD      <Vec1d, SAFE_LH>;
                computeLikelihoodDervMixlenPointer = &PhyloTree::computeLikelihoodDervMixlenGenericSIMD<Vec1d, SAFE_LH>;
                computePartialLikelihoodPointer    = &PhyloTree::computePartialLikelihoodGenericSIMD   <Vec1d, SAFE_LH>;
                computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferGenericSIMD<Vec1d>;
                break;
        }
        return;
    }

    switch(aln->num_states) {
        case 4:
            computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchSIMD    <Vec1d, NORM_LH, 4>;
            computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervSIMD      <Vec1d, NORM_LH, 4>;
            computeLikelihoodDervMixlenPointer = &PhyloTree::computeLikelihoodDervMixlenSIMD<Vec1d, NORM_LH, 4>;
            computePartialLikelihoodPointer    = &PhyloTree::computePartialLikelihoodSIMD   <Vec1d, NORM_LH, 4>;
            computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferSIMD<Vec1d, 4>;
            break;
        case 20:
            computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchSIMD    <Vec1d, NORM_LH, 20>;
            computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervSIMD      <Vec1d, NORM_LH, 20>;
            computeLikelihoodDervMixlenPointer = &PhyloTree::computeLikelihoodDervMixlenSIMD<Vec1d, NORM_LH, 20>;
            computePartialLikelihoodPointer    = &PhyloTree::computePartialLikelihoodSIMD   <Vec1d, NORM_LH, 20>;
            computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferSIMD<Vec1d, 20>;
            break;
        default:
            ASSERT(0);
            break;
    }
}

