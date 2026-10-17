//
//  superalignmentpairwiseplen.cpp
//  iqtree
//
//  Created by Olga on 04/05/17.
//
//

#include <stdio.h>
#include "superalignmentpairwiseplen.h"

/**********************************************************
 * class SuperAlignmentPairwisePlen
 **********************************************************/

SuperAlignmentPairwisePlen::SuperAlignmentPairwisePlen(PhyloSuperTreePlen *atree, int seq1, int seq2)
: SuperAlignmentPairwise((PhyloSuperTree*) atree, seq1, seq2)
{
    part_info = &(atree->part_info);
}

SuperAlignmentPairwisePlen::SuperAlignmentPairwisePlen(PhyloSuperTreePlen *atree)
: SuperAlignmentPairwise((PhyloSuperTree*) atree)
{
    part_info = &(atree->part_info);
}

double SuperAlignmentPairwisePlen::computeFunction(double value) {
    double lh = 0.0;
    for (int part : pair_parts) {
        lh += partitions[part]->computeFunction(part_info->at(part).part_rate*value);
    }
    return lh;
}

void SuperAlignmentPairwisePlen::computeFuncDerv(double value, double &df, double &ddf) {
    df = 0.0;
    ddf = 0.0;
    for (int part : pair_parts) {
        double d1, d2;
        double rate = part_info->at(part).part_rate;
        partitions[part]->computeFuncDerv(rate*value, d1, d2);
        df += rate*d1;
        ddf += rate*rate*d2;
    }
}

SuperAlignmentPairwisePlen::~SuperAlignmentPairwisePlen()
{}
