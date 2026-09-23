/*
 * phylosupertreeplen.h
 *
 *  Created on: Aug 5, 2013
 *      Author: olga
 */

#ifndef PHYLOSUPERTREEPLEN_H_
#define PHYLOSUPERTREEPLEN_H_

#include "phylosupertree.h"
#include "model/partitionmodel.h"
#include "alignment/superalignmentpairwise.h"

/**
 * this is to classify the cases which happen on the subtree
 *
 *  NNI_NONE_EPSILON: all 5 branches have images on subtree, this corresponds to change in subtree topology
 * 					  2 partial_lh vectors for -nni1 or 6 partial_lh vectors for -nni5 options
 *  NNI_ONE_EPSILON:  only one of the 5 branches has no image on subtree, this does not change subtree topology, but changes branch length of subtrees
 * 					  we need to allocate partial likelihood memory (1 partial_lh vectors for -nni1 option or 3 partial_lh for -nni5 option)
 * 	NNI_TWO_EPSILON:  two branches (on different sides of central branch) have no images, here after the NNI swap,
 * 					  the image of central branch either does not change or is equal to epsilon (then we decrease the branch length)
 * 					  and no allocation of partial_lh is needed
 * 	NNI_THREE_EPSILON: central and two adjacent edges have no images: after the NNI swap, central branch will have image and we need to relink it
 * 					no allocation of partial_lh is needed
 *  NNI_MANY_EPSILON: more than 3 branches have no images on subtree: nothing changes in subtree and no recomputation of partial likelihood are required
 */
enum NNIType {NNI_NO_EPSILON, NNI_ONE_EPSILON, NNI_TWO_EPSILON, NNI_THREE_EPSILON, NNI_MANY_EPSILON};


/**
Edge lengths in subtrees are proportional to edge lengths in a supertree.

	@author Olga Chernomor <olga.chernomor@univie.ac.at>
*/

class PhyloSuperTreePlen : public PhyloSuperTree {

public:
	/**
		constructors
	*/
	PhyloSuperTreePlen();
//    PhyloSuperTreePlen(Params &params);
    PhyloSuperTreePlen(SuperAlignment *alignment, int partition_type);
	PhyloSuperTreePlen(SuperAlignment *alignment, PhyloSuperTree *super_tree);

	~PhyloSuperTreePlen() override;

    /** normalize part_rate of part_info */
    void normalizePartRate();

    /**
        save object into the checkpoint
    */
    virtual void saveCheckpoint() override;

    /**
        restore object from the checkpoint
    */
    virtual void restoreCheckpoint() override;

    /**
     print tree to .treefile
     @param suffix suffix of the output file
     */
    virtual void printResultTree(string suffix = "") override;

    /**
            Read the tree saved with Taxon Names and branch lengths.
            @param tree_string tree string to read from
     */
    virtual void readTreeString(const string &tree_string) override;

    /**
     * Return the tree string containing taxon names and branch lengths
     * @return tree string
     */
    virtual string getTreeString() override;


    /**
            compute the distance between 2 sequences.
            @param seq1 index of sequence 1
            @param seq2 index of sequence 2
            @param initial_dist initial distance
            @return distance between seq1 and seq2
     */

    virtual double computeDist(int seq1, int seq2, double initial_dist, double &var) override;

	/**
		create sub-trees T|Y_1,...,T|Y_k of the current super-tree T
		and map F={f_1,...,f_k} the edges of supertree T to edges of subtrees T|Y_i
	*/
	virtual void mapTrees() override;

	/**
	 * Given current supertree T and subtrees T|Y_1,...,T|Y_k, build all maps f_1,...,f_k
	 */
	virtual void linkTrees() override;

    /**
            initialize partial_lh vector of all PhyloNeighbors, allocating central_partial_lh
     */
    virtual void initializeAllPartialLh() override;

    /**
            initialize partial_lh vector of all PhyloNeighbors, allocating central_partial_lh
            @param node the current node
            @param dad dad of the node, used to direct the search
            @param index the index
     */
    virtual void initializeAllPartialLh(int &index, int &indexlh, PhyloNode *node = nullptr, PhyloNode *dad = nullptr) override;

    void initializeAllPartialLh(double* &lh_addr, UBYTE* &scale_addr, UINT* &pars_addr, PhyloNode *node = nullptr, PhyloNode *dad = nullptr);

    /**
            de-allocate central_partial_lh
     */
    virtual void deleteAllPartialLh() override;

	/**
	 * get NNI type
	 */
	void getNNIType(PhyloNode *node1, PhyloNode *node2, vector<NNIType> &nni_type);

    /**
            Inherited from Optimization class.
            This function calculate f(value), first derivative f'(value) and 2nd derivative f''(value).
            used by Newton raphson method to minimize the function.
            @param value current branch length
            @param df (OUT) first derivative
            @param ddf (OUT) second derivative
     */
    virtual void computeFuncDerv(double value, double &df, double &ddf) override;

    /**
            inherited from Optimization class, to return to likelihood of the tree
            when the current branch length is set to value
            @param value current branch length
            @return negative of likelihood (for minimization)
     */
    virtual double computeFunction(double value) override;

    /**
            compute tree likelihood on a branch. used to optimize branch length
            @param dad_branch the branch leading to the subtree
            @param dad its dad, used to direct the tranversal
            @return tree likelihood
     */
    virtual double computeLikelihoodBranch(PhyloNeighbor *dad_branch, PhyloNode *dad, bool save_log_value = true) override;

    /**
            compute tree likelihood on a branch given buffer (theta_all), used after optimizing branch length
            @return tree likelihood
     */

    virtual double computeLikelihoodFromBuffer() override;

    /**
            optimize all branch lengths of all subtrees, then compute branch lengths
            of supertree as weighted average over all subtrees
            @param my_iterations number of iterations to loop through all branches
            @return the likelihood of the tree
     */
    virtual double optimizeAllBranches(int my_iterations = 100, double tolerance = TOL_LIKELIHOOD, int maxNRStep = 100) override;

    /**
            optimize one branch length by ML by optimizing all mapped branches of subtrees
            @param node1 1st end node of the branch
            @param node2 2nd end node of the branch
            @param clearLH true to clear the partial likelihood, otherwise false     */
    virtual void optimizeOneBranch(PhyloNode *node1, PhyloNode *node2, bool clearLH = true, int maxNRStep = 100) override;

    /**
            search the best swap for a branch
            @return NNIMove The best Move/Swap
            @param node1 1 of the 2 nodes on the branch
            @param node2 1 of the 2 nodes on the branch
     */
    virtual NNIMove getBestNNIForBran(PhyloNode *node1, PhyloNode *node2, NNIMove *nniMoves = nullptr) override;


    /**
            Do an NNI on the supertree and synchronize all subtrees respectively
            @param move the single NNI
     */
    virtual void doNNI(NNIMove &move, bool clearLH = true) override;

	/**
            apply  NNIs from the non-conflicting NNI list
            @param compatibleNNIs vector of all compatible NNIs
            @param changeBran whether or not the computed branch lengths should be applied
     */
    virtual void doNNIs(vector<NNIMove> &compatibleNNIs, bool changeBran = true) override;

    /**
     *   Apply 5 new branch lengths stored in the NNI move
     *   @param nnimove the NNI move currently in consideration
     */
    virtual void changeNNIBrans(NNIMove &nnimove) override;

    /**
            This is for ML. try to swap the tree with nearest neigbor interchange at the branch connecting node1-node2.
            If a swap shows better score, return the swapped tree and the score.
            @param cur_score current likelihood score
            @param node1 1st end node of the branch
            @param node2 2nd end node of the branch
            @param nni_param (OUT) if not nullptr: swapping information returned
            @return the likelihood of the tree
     */
    virtual double swapNNIBranch(double cur_score, PhyloNode *node1, PhyloNode *node2, SwapNNIParam *nni_param = nullptr, NNIMove *nniMoves = nullptr);

    /**
     *	used in swapNNIBranch to update link_neighbors of other SuperNeighbors that point to the same branch on SubTree as (node,dad)
     *	@param saved_link_dad_nei   pointer to link_neighbor dad_nei
     */
    void linkCheck(int part, Node* node, Node* dad, PhyloNeighbor* saved_link_dad_nei);
    void linkCheckRe(int part, Node* node, Node* dad, PhyloNeighbor* saved_link_dad_nei,PhyloNeighbor* saved_link_node_nei);

	/**
		compute the weighted average of branch lengths over partitions
	*/
	virtual void computeBranchLengths() override;

	// bool checkBranchLen();
	void mapBranchLen();
	void mapBranchLen(int part);
	virtual void printMapInfo() override;

//	virtual void restoreAllBrans(PhyloNode *node, PhyloNode *dad);

	/**
	 * initialize partition information for super tree
	*/
	virtual void initPartitionInfo() override;

	void printNNIcasesNUM();

    /**
     * 		indicates whether partition rates are fixed or not
     */

    bool fixed_rates;

    /*
     * 1 - # of is_nni on subtree
     * 2 - # of relink branch to an empty one
     * 3 - # of empty to empty
     * 4 - # of relink branch to a  new one (50% saving on these cases compared to the previous implementation)
     * 5 - # of relink branch to an old one + relink empty to some branch (100% saving on these cases)
     */
    int allNNIcases_computed[5];

    /**
            Neighbor-joining/parsimony tree might contain negative branch length. This
            function will fix this.
            @param node the current node
            @param dad dad of the node, used to direct the search
            @return The number of branches that have no/negative length
     */
    virtual int fixNegativeBranch(bool force = false, Node *node = nullptr, Node *dad = nullptr) override;

    virtual void reorientPartialLh(PhyloNeighbor* dad_branch, Node *dad) override;

protected:
	vector<uint64_t> partial_lh_entries, scale_num_entries, partial_pars_entries, block_size, scale_block_size,
    branch_block_size, g_matrix_block_size;

};



#endif /* PHYLOSUPERTREEPLEN_H_ */
