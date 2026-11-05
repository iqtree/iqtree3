/***************************************************************************
 *   Copyright (C) 2009 by BUI Quang Minh   *
 *   minh.bui@univie.ac.at   *
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
#include "superalignmentpairwise.h"

SuperAlignmentPairwise::SuperAlignmentPairwise()
 : AlignmentPairwise()
{
}

SuperAlignmentPairwise::SuperAlignmentPairwise(PhyloSuperTree *atree)
 : AlignmentPairwise()
{
	tree = atree;
	partitions.reserve(atree->size());
	for (auto it = atree->begin(); it != atree->end(); it++) {
		partitions.push_back(new AlignmentPairwise(*it));
	}
}

SuperAlignmentPairwise::SuperAlignmentPairwise(PhyloSuperTree *atree, int seq1, int seq2)
 : SuperAlignmentPairwise(atree)
{
	setSequenceNumbers(seq1, seq2);
}

void SuperAlignmentPairwise::setSequenceNumbers(int seq1, int seq2) {
	++pairCount;
	seq_id1 = seq1;
	seq_id2 = seq2;
	SuperAlignment *aln = (SuperAlignment*) tree->aln;
	pair_parts.clear();
	for (int part = 0; part < partitions.size(); part++) {
		int id1 = aln->taxa_index[seq1][part];
		int id2 = aln->taxa_index[seq2][part];
		if (id1 >= 0 && id2 >= 0) {
			partitions[part]->setSequenceNumbers(id1, id2);
			pair_parts.push_back(part);
		}
	}
}

double SuperAlignmentPairwise::recomputeDist(int seq1, int seq2, double initial_dist, double &d2l) {
	// initial guess of the distance using Jukes-Cantor correction over all partitions
	if (initial_dist == 0.0) {
		if (tree->params->compute_obs_dist)
			return tree->aln->computeObsDist(seq1, seq2);
		initial_dist = tree->aln->computeDist(seq1, seq2);
	}
	if (initial_dist == MAX_GENETIC_DIST) return initial_dist;
	// if no model or site rate is specified, return JC distance
	if (!tree->hasModelFactory() || !tree->hasRateHeterogeneity()) return initial_dist;
	// now optimize the distance based on the model and site rate
	setSequenceNumbers(seq1, seq2);
	return optimizeDist(initial_dist, d2l);
}

double SuperAlignmentPairwise::computeFunction(double value) {
	double lh = 0.0;
	for (int part : pair_parts) {
		lh += partitions[part]->computeFunction(value);
	}
	return lh;
}

void SuperAlignmentPairwise::computeFuncDerv(double value, double &df, double &ddf) {
	df = 0.0;
	ddf = 0.0;
	for (int part : pair_parts) {
		double d1, d2;
		partitions[part]->computeFuncDerv(value, d1, d2);
		df += d1;
		ddf += d2;
	}
}

SuperAlignmentPairwise::~SuperAlignmentPairwise()
{
	for (auto it : partitions)
		delete it;
}
