//
//  phylotreemixlen.h
//  iqtree
//
//  Created by Minh Bui on 24/08/15.
//
//
#ifndef PHYLOTREEMIXLEN_H_
#define PHYLOTREEMIXLEN_H_

#include "iqtree.h"

/**
 *  Phylogenetic tree with a mixture of branch lengths
 *  Started within a joint project with Stephen Crotty
 */
class PhyloTreeMixlen : public IQTree {
public:
    PhyloTreeMixlen();

    PhyloTreeMixlen(Alignment *aln);

    virtual bool isMixlen() const override { return !initializing_mixlen; }

    virtual int getNMixlen() const override { return (initializing_mixlen) ? 1 : mixlen; }

    virtual int getCurMixture() const override { return cur_mixture; }

    virtual void setCurMixture(int c) override;

    void clearRelativeTreelen() { relative_treelen.clear(); }

    /**
        start structure for checkpointing
    */
    virtual void startCheckpoint() override;

    /** 
        save object into the checkpoint
    */
    virtual void saveCheckpoint() override;

    /** 
        restore object from the checkpoint
    */
    virtual void restoreCheckpoint() override;

    /**
            allocate a new node. Override this if you have an inherited Node class.
            @param node_id node ID
            @param node_name node name
            @return a new node
     */
    virtual Node* newNode(int node_id = -1, const char* node_name = nullptr) override;

    /**
            allocate a new node. Override this if you have an inherited Node class.
            @param node_id node ID
            @param node_name node name issued by an interger
            @return a new node
     */
    virtual Node* newNode(int node_id, int node_name) override;
    
    /**
            refactored 2015-12-22: Taxon IDs instead of Taxon names to save space!
            Read the tree saved with Taxon IDs and branch lengths.
            @param tree_string tree string to read from
     */
    virtual void readTreeString(const string &tree_string) override;

    /**
            @param[out] lenvec tree lengths for each class in mixlen model
            @param node the starting node, nullptr to start from the root
            @param dad dad of the node, used to direct the search
     */
    virtual void treeLengths(DoubleVector &lenvec, Node *node = nullptr, Node *dad = nullptr) override;

    /**
     *  internal function called by printTree to print branch length
     *  @param out output stream
     *  @param length_nei target Neighbor to print
     */
    virtual void printBranchLength(ostream &out, int brtype, bool print_slash, Neighbor *length_nei) override;

    /**
            print tree to .treefile
            @param suffix suffix of the output file
     */
    virtual void printResultTree(string suffix = "") override;

    /**
     *  Set the model factory
     *  @param model_fac Model factory
     */
    virtual void setModelFactory(ModelFactory *model_fac) override;

    /** initialize parameters if necessary */
    void initializeMixlen(double tolerance, bool write_info);

    /**
        called by fixNegativeBranch to fix one branch
        @param branch_length new branch length
        @param dad_branch dad branch
        @param dad dad node
    */
    virtual void fixOneNegativeBranch(double branch_length, Neighbor *dad_branch, Node *dad) override;

    /**
     * IMPORTANT: semantic change: this function does not return score anymore, for efficiency purpose
            optimize one branch length by ML
            @param node1 1st end node of the branch
            @param node2 2nd end node of the branch
            @param clearLH true to clear the partial likelihood, otherwise false
            @param maxNRStep maximum number of Newton-Raphson steps
     */
    virtual void optimizeOneBranch(PhyloNode *node1, PhyloNode *node2, bool clearLH = true, int maxNRStep = 100) override;

    /**
            optimize all branch lengths of the tree
            @param my_iterations number of iterations to loop through all branches
            @return the likelihood of the tree
     */
    virtual double optimizeAllBranches(int my_iterations = 100, double tolerance = TOL_LIKELIHOOD, int maxNRStep = 100) override;

	/**
		This function calculate f(value), first derivative f'(value) and 2nd derivative f''(value).
		used by Newton raphson method to minimize the function.
		Please always override this function to adapt to likelihood or parsimony score.
		The default is for function f(x) = x^2.
		@param value x-value of the function
		@param df (OUT) first derivative
		@param ddf (OUT) second derivative
	*/
	virtual void computeFuncDervMulti(double *value, double *df, double *ddf) override;

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
		return the number of dimensions
	*/
	virtual int getNDim() override;


	/**
		the target function which needs to be optimized
		@param x the input vector x
		@return the function value at x
	*/
	virtual double targetFunk(double x[]) override;

	/**
		the approximated derivative function
		@param x the input vector x
		@param dfx the derivative at x
		@return the function value at x
	*/
	virtual double derivativeFunk(double x[], double dfx[]) override;

    /**
     *  Optimize current tree using NNI
     *
     *  @return
     *      <number of NNI steps, number of NNIs> done
     */
    virtual pair<int, int> optimizeNNI(bool speedNNI = true) override;

protected:
    /**
     *  Initialize mixlen branch lengths from branch lengths and relative_treelen
     */
    void initializeMixBranches(PhyloNode *node = nullptr, PhyloNode *dad = nullptr);

    /**
     *  Set branch lengths as respective mean mixlen branch lengths
     */
    void assignMeanMixBranches(Node *node = nullptr, Node *dad = nullptr);

protected:
    /** number of mixture categories */
    int mixlen;

    /** current category, for optimizing a branch length */
    int cur_mixture;

    /** relative rate, used to initialize branch lengths */
    DoubleVector relative_treelen;

    /** true if within initializeMixlen() */
    bool initializing_mixlen;
};

#endif /* PHYLOTREEMIXLEN_H_ */
