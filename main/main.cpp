/***************************************************************************
 *   Copyright (C) 2006 by BUI Quang Minh, Steffen Klaere, Arndt von Haeseler   *
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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iqtree_config.h>

#if defined WIN32 || defined _WIN32 || defined __WIN32__ || defined WIN64
#include <winsock2.h>
//#include <windows.h>
//extern __declspec(dllexport) int gethostname(char *name, int namelen);
#else
#include <sys/resource.h>
#endif

#include <stdio.h>
#include "tree/phylotree.h"
#include <signal.h>
#include <cstdio>
#include <streambuf>
#include <iostream>
#include <cstdlib>
#include <errno.h>
#include "pda/greedy.h"
#include "pda/pruning.h"
#include "pda/splitgraph.h"
#include "pda/circularnetwork.h"
#include "tree/mtreeset.h"
#include "tree/mexttree.h"
#include "ncl/ncl.h"
#include "nclextra/msetsblock.h"
#include "nclextra/myreader.h"
#include "phyloanalysis.h"
#include "alisim.h"
#include "tree/matree.h"
#include "obsolete/parsmultistate.h"
#include "alignment/maalignment.h" //added by MA
#include "tree/ncbitree.h"
#include "pda/ecopd.h"
#include "tree/upperbounds.h"
#include "terraceanalysis.h"
#include "pda/ecopdmtreeset.h"
#include "pda/gurobiwrapper.h"
#include "utils/timeutil.h"
#include "utils/operatingsystem.h" //for getOSName()
#include <stdlib.h>
#include "vectorclass/instrset.h"

#include "utils/MPIHelper.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace std;

inline void separator(ostream &out, int type = 0) {
    switch (type) {
    case 0:
        out << endl << "==============================================================================" << endl;
        break;
    case 1:
        out << endl << "-----------------------------------------------------------" << endl;
        break;
    default:
        break;
    }
}

void printCopyright(ostream &out) {
    string osname, pre, post;
    osname = getOSName();

    size_t osx_pos = osname.find("Mac OS X");
    size_t linux_pos = osname.find("Linux");
    if (osx_pos != string::npos) {
        // change the "Mac OS X" to "MacOS ARM" or "MacOS Intel"
        pre = osname.substr(0,osx_pos);
        post = osname.substr(osx_pos+8);
        #if defined(__ARM_NEON)
            osname = pre + "MacOS ARM" + post;
        #else
            osname = pre + "MacOS Intel" + post;
        #endif
    } else if (linux_pos != string::npos) {
        // change the "Linux" to "Linux ARM" or "Linux x86"
        pre = osname.substr(0,linux_pos);
        post = osname.substr(linux_pos+5);
        #if defined(__ARM_NEON)
            osname = pre + "Linux ARM" + post;
        #else
            osname = pre + "Linux x86" + post;
        #endif
    }
    
#ifdef IQ_TREE
     out << "IQ-TREE";
    #ifdef _IQTREE_MPI
    out << " MPI";
    #endif
    #ifndef _OPENMP
    out << " single-core";
    #endif
    #ifdef __AVX512KNL
    out << " Xeon Phi KNL";
    #endif
     out << " version ";
#else
     out << "PDA - Phylogenetic Diversity Analyzer version ";
#endif
    out << iqtree_VERSION_MAJOR << "." << iqtree_VERSION_MINOR << iqtree_VERSION_PATCH; // << " COVID-edition";
    out << " for " << osname;
    out << " built " << __DATE__;
#if defined DEBUG
    out << " - debug mode";
#endif

#ifdef IQ_TREE
    out << endl
        << "Developed by Bui Quang Minh, Thomas Wong, Nhan Ly-Trong, Huaiyan Ren" << endl
        << "Contributed by Lam-Tung Nguyen, Dominik Schrempf, Chris Bielow," << endl
        << "Olga Chernomor, Michael Woodhams, Diep Thi Hoang, Heiko Schmidt" << endl << endl;
#else
    out << endl << "Copyright (c) 2006-2014 Olga Chernomor, Arndt von Haeseler and Bui Quang Minh." << endl << endl;
#endif
}

void printRunMode(ostream &out, RunMode run_mode) {
    switch (run_mode) {
        case RunMode::DETECTED: out << "Detected"; break;
        case RunMode::GREEDY: out << "Greedy"; break;
        case RunMode::PRUNING: out << "Pruning"; break;
        case RunMode::BOTH_ALG: out << "Greedy and Pruning"; break;
        case RunMode::EXHAUSTIVE: out << "Exhaustive"; break;
        case RunMode::DYNAMIC_PROGRAMMING: out << "Dynamic Programming"; break;
        case RunMode::LINEAR_PROGRAMMING: out << "Integer Linear Programming"; break;
        default: outError(ERR_INTERNAL);
    }
}

/**
    summarize the running with header
*/
void summarizeHeader(ostream &out, Params &params, bool budget_constraint, InputType analysis_type) {
    printCopyright(out);
    out << "Input tree/split network file name: " << params.user_file << endl;
    if(params.eco_dag_file)
        out << "Input food web file name: "<<params.eco_dag_file<<endl;
     out << "Input file format: " << ((params.intype == IN_NEWICK) ? "Newick" : ( (params.intype == IN_NEXUS) ? "Nexus" : "Unknown" )) << endl;
    if (params.initial_file != NULL)
        out << "Initial taxa file: " << params.initial_file << endl;
    if (params.param_file != NULL)
        out << "Parameter file: " << params.param_file << endl;
    out << endl;
    out << "Type of measure: " << ((params.root != NULL || params.is_rooted) ? "Rooted": "Unrooted") <<
            (analysis_type== IN_NEWICK ? " phylogenetic diversity (PD)" : " split diversity (SD)");
    if (params.root != NULL) out << " at " << params.root;
    out << endl;
    if (params.run_mode != RunMode::CALC_DIST && params.run_mode != RunMode::PD_USER_SET) {
        out << "Search objective: " << ((params.find_pd_min) ? "Minimum" : "Maximum") << endl;
        out << "Search algorithm: ";
        printRunMode(out, params.run_mode);
        if (params.run_mode == RunMode::DETECTED) {
            out << " -> ";
            printRunMode(out, params.detected_mode);
        }
        out << endl;
        out << "Search option: " << ((params.find_all) ? "Multiple optimal sets" : "Single optimal set") << endl;
    }
    out << endl;
    out << "Type of analysis: ";
    switch (params.run_mode) {
        case RunMode::PD_USER_SET: out << "PD/SD of user sets";
            if (params.pdtaxa_file) out << " (" << params.pdtaxa_file << ")"; break;
        case RunMode::CALC_DIST: out << "Distance matrix computation"; break;
        default:
            out << ((budget_constraint) ? "Budget constraint " : "Subset size k ");
            if (params.intype == IN_NEWICK)
                out << ((analysis_type == IN_NEWICK) ? "on tree" : "on tree -> split network");
            else
                out << "on split network";
    }
    out << endl;
    //out << "Random number seed: " << params.ran_seed << endl;
}

void summarizeFooter(ostream &out, Params &params) {
    separator(out);
    time_t beginTime;
    time (&beginTime);
    char *date;
    date = ctime(&beginTime);

    out << "Time used: " << params.run_time  << " seconds." << endl;
    out << "Finished time: " << date << endl;
}


int getMaxNameLen(vector<string> &setName) {
    int len = 0;
    for (vector<string>::iterator it = setName.begin(); it != setName.end(); it++)
        if (len < (*it).length())
            len = (*it).length();
    return len;
}

void printPDUser(ostream &out, Params &params, PDRelatedMeasures &pd_more) {
    out << "List of user-defined sets of taxa with PD score computed" << endl << endl;
    int maxlen = getMaxNameLen(pd_more.setName)+2;
    out.width(maxlen);
    out << "Name" << "     PD";
    if (params.exclusive_pd) out << "   excl.-PD";
    if (params.endemic_pd) out << "   PD-Endem.";
    if (params.complement_area) out << "   PD-Compl. given area " << params.complement_area;
    out << endl;
    int cnt;
    for (cnt = 0; cnt < pd_more.setName.size(); cnt++) {
        out.width(maxlen);
        out << pd_more.setName[cnt] << " ";
        out.width(7);
        out << pd_more.PDScore[cnt] << "  ";
        if (params.exclusive_pd) {
            out.width(7);
            out << pd_more.exclusivePD[cnt] << "  ";
        }
        if (params.endemic_pd) {
            out.width(7);
            out << pd_more.PDEndemism[cnt] << "  ";
        }
        if (params.complement_area) {
            out.width(8);
            out << pd_more.PDComplementarity[cnt];
        }
        out << endl;
    }
    separator(out, 1);
}

void summarizeTree(Params &params, PDTree &tree, vector<PDTaxaSet> &taxa_set,
    PDRelatedMeasures &pd_more) {
    string filename;
    if (params.out_file == NULL) {
        filename = params.out_prefix;
        filename += ".pda";
    } else
        filename = params.out_file;

    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(filename.c_str());

        summarizeHeader(out, params, false, IN_NEWICK);
        out << "Tree size: " << tree.leafNum-params.is_rooted << " taxa, " <<
            tree.nodeNum-1-params.is_rooted << " branches" << endl;
        separator(out);

        vector<PDTaxaSet>::iterator tid;

        if (params.run_mode == RunMode::PD_USER_SET) {
            printPDUser(out, params, pd_more);
        }
        else if (taxa_set.size() > 1)
            out << "Optimal PD-sets with k = " << params.min_size-params.is_rooted <<
            " to " << params.sub_size-params.is_rooted << endl << endl;


        int subsize = params.min_size-params.is_rooted;
        if (params.run_mode == RunMode::PD_USER_SET) subsize = 1;
        for (tid = taxa_set.begin(); tid != taxa_set.end(); tid++, subsize++) {
            if (tid != taxa_set.begin())
                separator(out, 1);
            if (params.run_mode == RunMode::PD_USER_SET) {
                out << "Set " << subsize << " has PD score of " << tid->score << endl;
            }
            else {
                out << "For k = " << subsize << " the optimal PD score is " << (*tid).score << endl;
                out << "The optimal PD set has " << subsize << " taxa:" << endl;
            }
            for (NodeVector::iterator it = (*tid).begin(); it != (*tid).end(); it++)
                if ((*it)->name != ROOT_NAME){
                    out << (*it)->name << endl;
                }
            if (!tid->tree_str.empty()) {
                out << endl << "Corresponding sub-tree: " << endl;
                out << tid->tree_str << endl;
            }
            tid->clear();
        }
        taxa_set.clear();

        summarizeFooter(out, params);
        out.close();
        cout << endl << "Results are summarized in " << filename << endl << endl;

    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, filename);
    }
}


void printTaxaSet(Params &params, vector<PDTaxaSet> &taxa_set, RunMode cur_mode) {
    int subsize = params.min_size-params.is_rooted;
    ofstream out;
    ofstream scoreout;
    string filename;
    filename = params.out_prefix;
    filename += ".score";
    scoreout.open(filename.c_str());
    if (!scoreout.is_open())
        outError(ERR_WRITE_OUTPUT, filename);
    cout << "PD scores printed to " << filename << endl;

    if (params.nr_output == 1) {
        filename = params.out_prefix;
        filename += ".pdtaxa";
        out.open(filename.c_str());
        if (!out.is_open())
            outError(ERR_WRITE_OUTPUT, filename);
    }
    for (vector<PDTaxaSet>::iterator tid = taxa_set.begin(); tid != taxa_set.end(); tid++, subsize++) {
        if (params.nr_output > 10) {
            filename = params.out_prefix;
            filename += ".";
            filename += subsize;
            if (params.run_mode == RunMode::BOTH_ALG) {
                if (cur_mode == RunMode::GREEDY)
                    filename += ".greedy";
                else
                    filename += ".pruning";
            } else {
                filename += ".pdtree";
            }
            (*tid).printTree((char*)filename.c_str());

            filename = params.out_prefix;
            filename += ".";
            filename += subsize;
            filename += ".pdtaxa";
            (*tid).printTaxa((char*)filename.c_str());
        } else {
            out << subsize << " " << (*tid).score << endl;
            scoreout << subsize << " " << (*tid).score << endl;
            (*tid).printTaxa(out);
        }
    }

    if (params.nr_output == 1) {
        out.close();
        cout << "All taxa list(s) printed to " << filename << endl;
    }

    scoreout.close();
}


/**
    run PD algorithm on trees
*/
void runPDTree(Params &params)
{

    if (params.run_mode == RunMode::CALC_DIST) {
        bool is_rooted = false;
        MExtTree tree(params.user_file, is_rooted);
        cout << "Tree contains " << tree.leafNum << " taxa." << endl;
        cout << "Calculating distance matrix..." << endl;
        tree.calcDist(params.dist_file);
        cout << "Distances printed to " << params.dist_file << endl;
        return;
    }

    double t_begin, t_end;
    //char filename[300];
    //int idx;

    vector<PDTaxaSet> taxa_set;

    if (params.run_mode == RunMode::PD_USER_SET) {
        // compute score of user-defined sets
        t_begin = getCPUTime();
        cout << "Computing PD score for user-defined set of taxa..." << endl;
        PDTree tree(params);
        PDRelatedMeasures pd_more;
        tree.computePD(params, taxa_set, pd_more);

        if (params.endemic_pd)
            tree.calcPDEndemism(taxa_set, pd_more.PDEndemism);
        if (params.complement_area != NULL)
            tree.calcPDComplementarity(taxa_set, params.complement_area, pd_more.PDComplementarity);

        t_end = getCPUTime();
        params.run_time = (t_end-t_begin);
        summarizeTree(params, tree, taxa_set, pd_more);
        return;
    }


    /*********************************************
        run greedy algorithm
    *********************************************/

    if (params.sub_size < 2) {
        outError(ERR_NO_K);
    }

    bool detected_greedy = (params.run_mode != RunMode::PRUNING);

    Greedy test_greedy;

    test_greedy.init(params);

    if (params.root == NULL && !params.is_rooted)
        cout << endl << "Running PD algorithm on UNROOTED tree..." << endl;
    else
        cout << endl << "Running PD algorithm on ROOTED tree..." << endl;

    if (verbose_mode >= VB_DEBUG)
        test_greedy.drawTree(cout, WT_INT_NODE + WT_BR_SCALE + WT_BR_LEN);

    if (params.run_mode == RunMode::GREEDY || params.run_mode == RunMode::BOTH_ALG ||
        (params.run_mode == RunMode::DETECTED)) {

        if (params.run_mode == RunMode::DETECTED && params.sub_size >= test_greedy.leafNum * 7 / 10
            && params.min_size < 2)
            detected_greedy = false;

        if (detected_greedy) {
            params.detected_mode = RunMode::GREEDY;
            t_begin=getCPUTime();
            cout << endl << "Greedy Algorithm..." << endl;

            taxa_set.clear();
            test_greedy.run(params, taxa_set);

            t_end=getCPUTime();
            params.run_time = (t_end-t_begin);
            cout << "Time used: " << params.run_time << " seconds." << endl;
            if (params.min_size == params.sub_size)
                cout << "Resulting tree length = " << taxa_set[0].score << endl;

            if (params.nr_output > 0)
                printTaxaSet(params, taxa_set, RunMode::GREEDY);

            PDRelatedMeasures pd_more;

            summarizeTree(params, test_greedy, taxa_set, pd_more);
        }
    }

    /*********************************************
        run pruning algorithm
    *********************************************/
    if (params.run_mode == RunMode::PRUNING || params.run_mode == RunMode::BOTH_ALG ||
        (params.run_mode == RunMode::DETECTED)) {

        Pruning test_pruning;

        if (params.run_mode == RunMode::PRUNING || params.run_mode == RunMode::BOTH_ALG) {
            //Pruning test_pruning(params);
            test_pruning.init(params);
        } else if (!detected_greedy) {
            test_pruning.init(test_greedy);
        } else {
            return;
        }
        params.detected_mode = RunMode::PRUNING;
        t_begin=getCPUTime();
        cout << endl << "Pruning Algorithm..." << endl;
        taxa_set.clear();
        test_pruning.run(params, taxa_set);

        t_end=getCPUTime();
        params.run_time = (t_end-t_begin) ;
        cout << "Time used: " << params.run_time << " seconds.\n";
        if (params.min_size == params.sub_size)
            cout << "Resulting tree length = " << taxa_set[0].score << endl;

        if (params.nr_output > 0)
            printTaxaSet(params, taxa_set, RunMode::PRUNING);

        PDRelatedMeasures pd_more;

        summarizeTree(params, test_pruning, taxa_set, pd_more);

    }

}

void checkSplitDistance(ostream &out, PDNetwork &sg) {
    mmatrix(double) dist;
    sg.calcDistance(dist);
    int ntaxa = sg.getNTaxa();
    int i, j;
    bool found = false;
    for (i = 0; i < ntaxa-1; i++) {
        bool first = true;
        for (j = i+1; j < ntaxa; j++)
            if (abs(dist[i][j]) <= 1e-5) {
                if (!found) {
                    out << "The following sets of taxa (each set in a line) have very small split-distance" << endl;
                    out << "( <= 1e-5) as computed from the split system. To avoid a lot of multiple" << endl;
                    out << "optimal PD sets to be reported, one should only keep one taxon from each set" << endl;
                    out << "and exclude the rest from the analysis." << endl << endl;
                }
                if (first)
                    out << sg.getTaxa()->GetTaxonLabel(i);
                found = true;
                first = false;
                out << ", " << sg.getTaxa()->GetTaxonLabel(j);
            }
        if (!first) out << endl;
    }
    if (found)
        separator(out);
}



/**
    check if the set are nested and there are no multiple optimal sets.
    If yes, return the ranking as could be produced by a greedy algorithm
*/
bool makeRanking(vector<SplitSet> &pd_set, IntVector &indices, IntVector &ranking) {
    vector<SplitSet>::iterator it;
    IntVector::iterator inti;
    ranking.clear();
    bool nested = true;
    Split *cur_sp = NULL;
    int id = 1;
    for (it = pd_set.begin(); it != pd_set.end(); it++) {
        if ((*it).empty()) continue;
        if ((*it).size() > 1) {
            nested = false;
            ranking.push_back(-10);
            indices.push_back(0);
        }
        Split *sp = (*it)[0];

        if (!cur_sp) {
            IntVector sp_tax;
            sp->getTaxaList(sp_tax);
            ranking.insert(ranking.end(), sp_tax.begin(), sp_tax.end());
            for (inti = sp_tax.begin(); inti != sp_tax.end(); inti++)
                indices.push_back(id++);
        } else {
            if ( !cur_sp->subsetOf(*sp)) {
                ranking.push_back(-1);
                indices.push_back(0);
                nested = false;
            }
            Split sp_diff(*sp);
            sp_diff -= *cur_sp;
            Split sp_diff2(*cur_sp);
            sp_diff2 -= *sp;
            IntVector sp_tax;
            sp_diff2.getTaxaList(sp_tax);
            ranking.insert(ranking.end(), sp_tax.begin(), sp_tax.end());
            for (inti = sp_tax.begin(); inti != sp_tax.end(); inti++)
                indices.push_back(-id);
            sp_diff.getTaxaList(sp_tax);
            ranking.insert(ranking.end(), sp_tax.begin(), sp_tax.end());
            for (inti = sp_tax.begin(); inti != sp_tax.end(); inti++)
                indices.push_back(id);
            if ( !cur_sp->subsetOf(*sp)) {
                ranking.push_back(-2);
                indices.push_back(0);
            }
            id++;
        }
        cur_sp = sp;
    }
    return nested;
}


void printNexusSets(const char *filename, PDNetwork &sg, vector<SplitSet> &pd_set) {
    try {
        ofstream out;
        out.open(filename);
        out << "#NEXUS" << endl << "BEGIN Sets;" << endl;
        vector<SplitSet>::iterator it;
        for (it = pd_set.begin(); it != pd_set.end(); it++) {
            int id = 1;
            for (SplitSet::iterator sit = (*it).begin(); sit != (*it).end(); sit++, id++) {
                IntVector taxa;
                (*sit)->getTaxaList(taxa);
                out << "   TAXSET Opt_" << taxa.size() << "_" << id << " =";
                for (IntVector::iterator iit = taxa.begin(); iit != taxa.end(); iit++) {
                    if (sg.isPDArea())
                        out << " '" << sg.getSetsBlock()->getSet(*iit)->name << "'";
                    else
                        out << " '" << sg.getTaxa()->GetTaxonLabel(*iit) << "'";
                }
                out << ";" << endl;
            }
        }
        out << "END; [Sets]" << endl;
        out.close();
        cout << endl << "Optimal sets are written to nexus file " << filename << endl;
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, filename);
    }

}



void computeTaxaFrequency(SplitSet &taxa_set, DoubleVector &freq) {
    ASSERT(taxa_set.size());
    int ntaxa = taxa_set[0]->getNTaxa();
    int i;

    freq.resize(ntaxa, 0);
    for (SplitSet::iterator it2 = taxa_set.begin(); it2 != taxa_set.end(); it2++) {
        for ( i = 0; i < ntaxa; i++)
            if ((*it2)->containTaxon(i)) freq[i] += 1.0;
    }

    for ( i = 0; i < ntaxa; i++)
        freq[i] /= taxa_set.size();

}

/**
    summarize the running results
*/
void summarizeSplit(Params &params, PDNetwork &sg, vector<SplitSet> &pd_set, PDRelatedMeasures &pd_more, bool full_report) {
    int i;


    if (params.nexus_output) {
        string nex_file = params.out_prefix;
        nex_file += ".pdsets.nex";
        printNexusSets(nex_file.c_str(), sg, pd_set);
    }
    string filename;
    if (params.out_file == NULL) {
        filename = params.out_prefix;
        filename += ".pda";
    } else
        filename = params.out_file;

    try {
        ofstream out;
        out.open(filename.c_str());
        /****************************/
        /********** HEADER **********/
        /****************************/
        summarizeHeader(out, params, sg.isBudgetConstraint(), IN_NEXUS);

        out << "Network size: " << sg.getNTaxa()-params.is_rooted << " taxa, " <<
            sg.getNSplits()-params.is_rooted << " splits (of which " <<
            sg.getNTrivialSplits() << " are trivial splits)" << endl;
        out << "Network type: " << ((sg.isCircular()) ? "Circular" : "General") << endl;

        separator(out);

        checkSplitDistance(out, sg);

        int c_num = 0;
        //int subsize = (sg.isBudgetConstraint()) ? params.budget : (params.sub_size-params.is_rooted);
        //subsize -= pd_set.size()-1;
        int subsize = (sg.isBudgetConstraint()) ? params.min_budget : params.min_size-params.is_rooted;
        int stepsize = (sg.isBudgetConstraint()) ? params.step_budget : params.step_size;
        if (params.detected_mode != RunMode::LINEAR_PROGRAMMING) stepsize = 1;
        vector<SplitSet>::iterator it;
        SplitSet::iterator it2;


        if (params.run_mode == RunMode::PD_USER_SET) {
            printPDUser(out, params, pd_more);
        }

        /****************************/
        /********** SUMMARY *********/
        /****************************/

        if (params.run_mode != RunMode::PD_USER_SET && !params.num_bootstrap_samples) {
            out << "Summary of the PD-score and the number of optimal PD-sets with the same " << endl << "optimal PD-score found." << endl;

            if (sg.isBudgetConstraint())
                out << endl << "Budget   PD-score   %PD-score   #PD-sets" << endl;
            else
                out << endl << "Size-k   PD-score   %PD-score   #PD-sets" << endl;

            int sizex = subsize;
            double total = sg.calcWeight();

            for (it = pd_set.begin(); it != pd_set.end(); it++, sizex+=stepsize) {
                out.width(6);
                out << right << sizex << " ";
                out.width(10);
                out << right << (*it).getWeight() << " ";
                out.width(10);
                out << right << ((*it).getWeight()/total)*100.0 << " ";
                out.width(6);
                out << right << (*it).size();
                out << endl;
            }

            out << endl;
            if (!params.find_all)
                out << "Note: You did not choose the option to find multiple optimal PD sets." << endl <<
                    "That's why we only reported one PD-set per size-k or budget. If you want" << endl <<
                    "to determine all multiple PD-sets, use the '-a' option.";
            else {
                out << "Note: The number of multiple optimal PD sets to be reported is limited to " << params.pd_limit << "." << endl <<
                    "There might be cases where the actual #PD-sets exceeds that upper-limit but" << endl <<
                    "won't be listed here. Please refer to the above list to identify such cases." << endl <<
                    "To increase the upper-limit, use the '-lim <limit_number>' option.";
            }
            out << endl;
            separator(out);
        }

        if (!full_report) {
            out.close();
            return;
        }


        /****************************/
        /********* BOOTSTRAP ********/
        /****************************/
        if (params.run_mode != RunMode::PD_USER_SET && params.num_bootstrap_samples) {
            out << "Summary of the bootstrap analysis " << endl;
            for (it = pd_set.begin(); it != pd_set.end(); it++) {
                DoubleVector freq;
                computeTaxaFrequency((*it), freq);
                out << "For k/budget = " << subsize << " the " << ((sg.isPDArea()) ? "areas" : "taxa")
                    << " supports are: " << endl;
                for (i = 0; i < freq.size(); i++)
                    out << ((sg.isPDArea()) ? sg.getSetsBlock()->getSet(i)->name : sg.getTaxa()->GetTaxonLabel(i))
                        << "\t" << freq[i] << endl;
                if ((it+1) != pd_set.end()) separator(out, 1);
            }
            out << endl;
            separator(out);
        }

        /****************************/
        /********** RANKING *********/
        /****************************/

        if (params.run_mode != RunMode::PD_USER_SET && !params.num_bootstrap_samples) {


            IntVector ranking;
            IntVector index;

            out << "Ranking based on the optimal sets" << endl;


            if (!makeRanking(pd_set, index, ranking)) {
                out << "WARNING: Optimal sets are not nested, so ranking should not be considered stable" << endl;
            }
            if (subsize > 1) {
                out << "WARNING: The first " << subsize << " ranks should be treated equal" << endl;
            }
            out << endl << "Rank*   ";
            if (!sg.isPDArea())
                out << "Taxon names" << endl;
            else
                out << "Area names" << endl;


            for (IntVector::iterator intv = ranking.begin(), intid = index.begin(); intv != ranking.end(); intv ++, intid++) {
                if (*intv == -10)
                    out << "<--- multiple optimal set here --->" << endl;
                else if (*intv == -1)
                    out << "<--- BEGIN: greedy does not work --->" << endl;
                else if (*intv == -2)
                    out << "<--- END --->" << endl;
                else {
                    out.width(5);
                    out <<  right << *intid << "   ";
                    if (sg.isPDArea())
                        out << sg.getSetsBlock()->getSet(*intv)->name << endl;
                    else
                        out << sg.getTaxa()->GetTaxonLabel(*intv) << endl;
                }
            }
            out << endl;
            out <<  "(*) Negative ranks indicate the point at which the greedy algorithm" << endl <<
                    "    does not work. In that case, the corresponding taxon/area names" << endl <<
                    "    should be deleted from the optimal set of the same size" << endl;
            separator(out);
        }

        int max_len = sg.getTaxa()->GetMaxTaxonLabelLength();

        /****************************/
        /***** DETAILED SETS ********/
        /****************************/

        if (params.run_mode != RunMode::PD_USER_SET)
            out << "Detailed information of all taxa found in the optimal PD-sets" << endl;

        if (pd_set.size() > 1) {
            if (sg.isBudgetConstraint())
                out << "with budget = " << params.min_budget <<
                    " to " << params.budget << endl << endl;
            else
                out << "with k = " << params.min_size-params.is_rooted <<
                    " to " << params.sub_size-params.is_rooted << endl << endl;
        }

        if (params.run_mode != RunMode::PD_USER_SET)
            separator(out,1);

        for (it = pd_set.begin(); it != pd_set.end(); it++, subsize+=stepsize) {

            // check if the pd-sets are the same as previous one
            if (sg.isBudgetConstraint() && it != pd_set.begin()) {
                vector<SplitSet>::iterator prev, next;
                for (next=it, prev=it-1; next != pd_set.end() && next->getWeight() == (*prev).getWeight() &&
                    next->size() == (*prev).size(); next++ ) ;
                if (next != it) {
                    // found something in between!
                    out << endl;
                    //out << endl << "**************************************************************" << endl;
                    out << "For budget = " << subsize << " -> " << subsize+(next-it-1)*stepsize <<
                        " the optimal PD score and PD sets" << endl;
                    out << "are identical to the case when budget = " << subsize-stepsize << endl;
                    //out << "**************************************************************" << endl;
                    subsize += (next-it)*stepsize;
                    it = next;
                    if (it == pd_set.end()) break;
                }
            }

            if (it != pd_set.begin()) separator(out, 1);

            int num_sets = (*it).size();
            double weight = (*it).getWeight();

            if (params.run_mode != RunMode::PD_USER_SET) {
                out << "For " << ((sg.isBudgetConstraint()) ? "budget" : "k") << " = " << subsize;
                out << " the optimal PD score is " << weight << endl;

                if (num_sets == 1) {
                    if (!sg.isBudgetConstraint())
                        out << "The optimal PD set has " << (*it)[0]->countTaxa()-params.is_rooted <<
                            ((sg.isPDArea()) ? " areas" : " taxa");
                    else
                        out << "The optimal PD set has " << (*it)[0]->countTaxa()-params.is_rooted <<
                        ((sg.isPDArea()) ? " areas" : " taxa") << " and requires " << sg.calcCost(*(*it)[0]) << " budget";
                    if (!sg.isPDArea()) out << " and covers " << sg.countSplits(*(*it)[0]) <<
                        " splits (of which " << sg.countInternalSplits(*(*it)[0]) << " are internal splits)";
                    out << endl;
                }
                else
                    out << "Found " << num_sets << " PD sets with the same optimal score." << endl;
            }
            for (it2 = (*it).begin(), c_num=1; it2 != (*it).end(); it2++, c_num++){
                Split *this_set = *it2;

                if (params.run_mode == RunMode::PD_USER_SET && it2 != (*it).begin())
                    separator(out, 1);

                if (params.run_mode == RunMode::PD_USER_SET) {
                    if (!sg.isBudgetConstraint())
                        out << "Set " << c_num << " has PD score of " << this_set->getWeight();
                    else
                        out << "Set " << c_num << " has PD score of " << this_set->getWeight() <<
                        " and requires " << sg.calcCost(*this_set) << " budget";
                } else if (num_sets > 1) {
                    if (!sg.isBudgetConstraint())
                        out << endl << "PD set " << c_num;
                    else
                        out << endl << "PD set " << c_num << " has " << this_set->countTaxa()-params.is_rooted <<
                        " taxa and requires " << sg.calcCost(*this_set) << " budget";
                }

                if (!sg.isPDArea() && (num_sets > 1 || params.run_mode == RunMode::PD_USER_SET ))
                    out << " and covers " << sg.countSplits(*(*it)[0]) << " splits (of which "
                    << sg.countInternalSplits(*(*it)[0]) << " are internal splits)";
                out << endl;

                if (params.run_mode != RunMode::PD_USER_SET && sg.isPDArea()) {
                    for (i = 0; i < sg.getSetsBlock()->getNSets(); i++)
                        if (this_set->containTaxon(i)) {
                            if (sg.isBudgetConstraint()) {
                                out.width(max_len);
                                out << left << sg.getSetsBlock()->getSet(i)->name << "\t";
                                out.width(10);
                                out << right << sg.getPdaBlock()->getCost(i);
                                out << endl;

                            } else {
                                out << sg.getSetsBlock()->getSet(i)->name << endl;
                            }
                        }

                    Split sp(sg.getNTaxa());
                    for (i = 0; i < sg.getSetsBlock()->getNSets(); i++)
                        if (this_set->containTaxon(i))
                            sp += *(sg.area_taxa[i]);
                    out << endl << "which contains " << sp.countTaxa() - params.is_rooted << " taxa: " << endl;
                    for (i = 0; i < sg.getNTaxa(); i++)
                        if (sg.getTaxa()->GetTaxonLabel(i) != ROOT_NAME && sp.containTaxon(i))
                            out << sg.getTaxa()->GetTaxonLabel(i) << endl;

                } else
                for ( i = 0; i < sg.getNTaxa(); i++)
                    if (sg.getTaxa()->GetTaxonLabel(i) != ROOT_NAME && this_set->containTaxon(i)) {
                        if (sg.isBudgetConstraint()) {
                            out.width(max_len);
                            out << left << sg.getTaxa()->GetTaxonLabel(i) << "\t";
                            out.width(10);
                            out << right << sg.getPdaBlock()->getCost(i);
                            out << endl;

                        } else {
                            out << sg.getTaxa()->GetTaxonLabel(i) << endl;
                        }
                    }
            }
        }

        /****************************/
        /********** FOOTER **********/
        /****************************/

        summarizeFooter(out, params);

        out.close();
        cout << endl << "Results are summarized in " << filename << endl << endl;
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, filename);
    }
}

void printGainMatrix(char *filename, mmatrix(double) &delta_gain, int start_k) {
    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(filename);
        int k = start_k;
        for (mmatrix(double)::iterator it = delta_gain.begin(); it != delta_gain.end(); it++, k++) {
            out << k;
            for (int i = 0; i < (*it).size(); i++)
                out << "  " << (*it)[i];
            out << endl;
        }
        out.close();
        cout << "PD gain matrix printed to " << filename << endl;
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, filename);
    }
}

/**
    run PD algorithm on split networks
*/
void runPDSplit(Params &params) {

    cout << "Using NCL - Nexus Class Library" << endl << endl;

    // init a split graph class from the parameters
    CircularNetwork sg(params);
    int i;

    // this vector of SplitSet store all the optimal PD sets
    vector<SplitSet> pd_set;
    // this define an order of taxa (circular order in case of circular networks)
    vector<int> taxa_order;
    // this store a particular taxa set
    Split taxa_set;


    if (sg.isCircular()) {
        // is a circular network, get circular order
        for (i = 0; i < sg.getNTaxa(); i++)
            taxa_order.push_back(sg.getCircleId(i));
    } else
        // otherwise, get the incremental order
        for (i = 0; i < sg.getNTaxa(); i++)
            taxa_order.push_back(i);

    PDRelatedMeasures pd_more;

    // begining time of the algorithm run
    double time_begin = getCPUTime();
    //time(&time_begin);
    // check parameters
    if (sg.isPDArea()) {
        if (sg.isBudgetConstraint()) {
            int budget = (params.budget >= 0) ? params.budget : sg.getPdaBlock()->getBudget();
            if (budget < 0 && params.pd_proportion == 0.0) params.run_mode = RunMode::PD_USER_SET;
        } else {
            int sub_size = (params.sub_size >= 1) ? params.sub_size : sg.getPdaBlock()->getSubSize();
            if (sub_size < 1 && params.pd_proportion == 0.0) params.run_mode = RunMode::PD_USER_SET;

        }
    }

    if (params.run_mode == RunMode::PD_USER_SET) {
        // compute score of user-defined sets
        cout << "Computing PD score for user-defined set of taxa..." << endl;
        pd_set.resize(1);
        sg.computePD(params, pd_set[0], pd_more);
        if (params.endemic_pd)
            sg.calcPDEndemism(pd_set[0], pd_more.PDEndemism);

        if (params.complement_area != NULL)
            sg.calcPDComplementarity(pd_set[0], params.complement_area, pd_more.setName, pd_more.PDComplementarity);

    } else {
        // otherwise, call the main function
        if (params.num_bootstrap_samples) {
            cout << endl << "======= START BOOTSTRAP ANALYSIS =======" << endl;
            MTreeSet *mtrees = sg.getMTrees();
            if (mtrees->size() < 100)
                cout << "WARNING: bootstrap may be unstable with less than 100 trees" << endl;
            vector<string> taxname;
            sg.getTaxaName(taxname);
            i = 1;
            for (MTreeSet::iterator it = mtrees->begin(); it != mtrees->end(); it++, i++) {
                cout << "---------- TREE " << i << " ----------" << endl;
                // convert tree into split sytem
                SplitGraph sg2;
                (*it)->convertSplits(taxname, sg2);
                // change the current split system
                for (SplitGraph::reverse_iterator it = sg.rbegin(); it != sg.rend(); it++) {
                    delete *it;
                }
                sg.clear();
                sg.insert(sg.begin(), sg2.begin(), sg2.end());
                sg2.clear();

                // now findPD on the converted tree-split system
                sg.findPD(params, pd_set, taxa_order);
            }
            cout << "======= DONE BOOTSTRAP ANALYSIS =======" << endl << endl;
        } else {
            sg.findPD(params, pd_set, taxa_order);
        }
    }

    // ending time
    double time_end = getCPUTime();
    //time(&time_end);
    params.run_time = time_end - time_begin;

    cout << "Time used: " << (double) (params.run_time) << " seconds." << endl;

    if (verbose_mode >= VB_DEBUG && !sg.isPDArea()) {
        cout << "PD set(s) with score(s): " << endl;
        for (vector<SplitSet>::iterator it = pd_set.begin(); it != pd_set.end(); it++)
        for (SplitSet::iterator it2 = (*it).begin(); it2 != (*it).end(); it2++ ){
            //(*it)->report(cout);
            cout << "  " << (*it2)->getWeight() << "    ";
            for (i = 0; i < sg.getNTaxa(); i++)
                if ((*it2)->containTaxon(i))
                cout << sg.getTaxa()->GetTaxonLabel(i) << "  ";
            if (sg.isBudgetConstraint())
                cout << " (budget = " << sg.calcCost(*(*it2)) << ")";
            cout << endl;
        }
    }

    sg.printOutputSetScore(params, pd_set);


    summarizeSplit(params, sg, pd_set, pd_more, true);

    if (params.calc_pdgain) {
        mmatrix(double) delta_gain;
        sg.calcPDGain(pd_set, delta_gain);
        string filename = params.out_prefix;
        filename += ".pdgain";
        printGainMatrix((char*)filename.c_str(), delta_gain, pd_set.front().front()->countTaxa());
        //cout << delta_gain;
    }


    //for (i = pd_set.size()-1; i >= 0; i--)
    //    delete pd_set[i];

}

void printSplitSet(SplitGraph &sg, SplitIntMap &hash_ss) {
/*
    for (SplitIntMap::iterator it = hash_ss.begin(); it != hash_ss.end(); it++) {
        if ((*it)->getWeight() > 50 && (*it)->countTaxa() > 1)
        (*it)->report(cout);
    }*/
    sg.getTaxa()->Report(cout);
    for (SplitGraph::iterator it = sg.begin(); it != sg.end(); it++) {
        if ((*it)->getWeight() > 50 && (*it)->countTaxa() > 1)
        (*it)->report(cout);
    }
}

void readTaxaOrder(char *taxa_order_file, StrVector &taxa_order) {

}

void calcTreeCluster(Params &params) {
    ASSERT(params.taxa_order_file);
    MExtTree tree(params.user_file, params.is_rooted);
//    StrVector taxa_order;
    //readTaxaOrder(params.taxa_order_file, taxa_order);
    NodeVector taxa;
    mmatrix(int) clusters;
    clusters.reserve(tree.leafNum - 3);
    tree.getTaxa(taxa);
    sort(taxa.begin(), taxa.end(), nodenamecmp);
    tree.createCluster(taxa, clusters);
    int cnt = 1;

    string treename = params.out_prefix;
    treename += ".clu-id";
    tree.printTree(treename.c_str());

    for (mmatrix(int)::iterator it = clusters.begin(); it != clusters.end(); it++, cnt++) {
        ostringstream filename;
        filename << params.out_prefix << "." << cnt << ".clu";
        ofstream out(filename.str().c_str());

        ostringstream filename2;
        filename2 << params.out_prefix << "." << cnt << ".name-clu";
        ofstream out2(filename2.str().c_str());

        out << "w" << endl << "c" << endl << "4" << endl << "b" << endl << "g" << endl << 4-params.is_rooted << endl;
        IntVector::iterator it2;
        NodeVector::iterator it3;
        for (it2 = (*it).begin(), it3 = taxa.begin(); it2 != (*it).end(); it2++, it3++)
            if ((*it3)->name != ROOT_NAME) {
                out << char((*it2)+'a') << endl;
                out2 << (*it3)->name << "  " << char((*it2)+'a') << endl;
            }
        out << "y" << endl;
        out.close();
        out2.close();
        cout << "Cluster " << cnt << " printed to " << filename.rdbuf() << " and " << filename2.rdbuf() << endl;
    }
}


void printTaxa(Params &params) {
    MTree mytree(params.user_file, params.is_rooted);
    vector<string> taxname;
    taxname.resize(mytree.leafNum);
    mytree.getTaxaName(taxname);
    sort(taxname.begin(), taxname.end());

    string filename = params.out_prefix;
    filename += ".taxa";

    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(filename.c_str());
        for (vector<string>::iterator it = taxname.begin(); it != taxname.end(); it++) {
            if ((*it) != ROOT_NAME) out << (*it);
            out << endl;
        }
        out.close();
        cout << "All taxa names printed to " << filename << endl;
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, filename);
    }
}

void printAreaList(Params &params) {
    MSetsBlock *sets;
    sets = new MSetsBlock();
     cout << "Reading input file " << params.user_file << "..." << endl;

    MyReader nexus(params.user_file);

    nexus.Add(sets);

    MyToken token(nexus.inf);
    nexus.Execute(token);

    //sets->Report(cout);

    TaxaSetNameVector *allsets = sets->getSets();

    string filename = params.out_prefix;
    filename += ".names";

    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(filename.c_str());
        for (TaxaSetNameVector::iterator it = allsets->begin(); it != allsets->end(); it++) {
            out << (*it)->name;
            out << endl;
        }
        out.close();
        cout << "All area names printed to " << filename << endl;
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, filename);
    }

    delete sets;
}

void scaleBranchLength(Params &params) {
    params.is_rooted = true;
    PDTree tree(params);
    if (params.run_mode == RunMode::SCALE_BRANCH_LEN) {
        cout << "Scaling branch length with a factor of " << params.scaling_factor << " ..." << endl;
        tree.scaleLength(params.scaling_factor, false);
    } else {
        cout << "Scaling clade support with a factor of " << params.scaling_factor << " ..." << endl;
        tree.scaleCladeSupport(params.scaling_factor, false);
    }
    if (params.out_file != NULL)
        tree.printTree(params.out_file);
    else {
        tree.printTree(cout);
        cout << endl;
    }
}

void calcDistribution(Params &params) {

    PDTree mytree(params);

    string filename = params.out_prefix;
    filename += ".randompd";

    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(filename.c_str());
        for (int size = params.min_size; size <= params.sub_size; size += params.step_size) {
            out << size;
            for (int sample = 0; sample < params.sample_size; sample++) {
                Split taxset(mytree.leafNum);
                taxset.randomize(size);
                mytree.calcPD(taxset);
                out << "  " << taxset.getWeight();
            }
            out << endl;
        }
        out.close();
        cout << "PD distribution is printed to " << filename << endl;
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, filename);
    }
}

void printRFDist(string filename, double *rfdist, int n, int m, int rf_dist_mode, bool print_msg = true) {
    int i, j;

    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(filename);
        if (Params::getInstance().output_format == FORMAT_CSV) {
            out << "# Robinson-Foulds distances" << endl
            << "# This file can be read in MS Excel or in R with command:" << endl
            << "#    dat=read.csv('" <<  filename << "',comment.char='#')" << endl
            << "# Columns are comma-separated with following meanings:" << endl
            << "#    ID1:     Tree 1 ID" << endl
            << "#    ID2:     Tree 2 ID" << endl
            << "#    Dist:    Robinson-Foulds distance" << endl
            << "ID1,ID2,Dist" << endl;
            if (rf_dist_mode == RF_ADJACENT_PAIR) {
                for (i = 0; i < n; i++)
                    out << i+1 << ',' << i+2 << ',' << rfdist[i] << endl;
            } else if (Params::getInstance().rf_same_pair) {
                for (i = 0; i < n; i++)
                    out << i+1 << ',' << i+1 << ',' << rfdist[i] << endl;
            } else {
                for (i = 0; i < n; i++)  {
                    for (j = 0; j < m; j++)
                        out << i+1 << ',' << j+1 << ',' << rfdist[i*m+j] << endl;
                }
            }
        } else if (rf_dist_mode == RF_ADJACENT_PAIR || Params::getInstance().rf_same_pair) {
            out << "XXX        ";
            out << 1 << " " << n << endl;
            for (i = 0; i < n; i++)
                out << " " << rfdist[i];
            out << endl;
        } else {
            // all pairs
            out << n << " " << m << endl;
            for (i = 0; i < n; i++)  {
                out << "Tree" << i << "      ";
                for (j = 0; j < m; j++)
                    out << " " << rfdist[i*m+j];
                out << endl;
            }
        }
        out.close();
        if (print_msg)
            cout << "Robinson-Foulds distances printed to " << filename << endl;
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, filename);
    }
}

void computeRFDistExtended(const char *trees1, const char *trees2, const char *filename) {
    cout << "Reading input trees 1 file " << trees1 << endl;
    int ntrees = 0, ntrees2 = 0;
    double *rfdist_raw = NULL;
    try {
        ifstream in;
        in.exceptions(ios::failbit | ios::badbit);
        in.open(trees1);
        IntVector rfdist;
        for (ntrees = 1; !in.eof(); ntrees++) {
            MTree tree;
            bool is_rooted = false;

            // read in the tree and convert into split system for indexing
            tree.readTree(in, is_rooted);
            if (verbose_mode >= VB_DEBUG)
                cout << ntrees << " " << endl;
            DoubleVector dist;
            tree.computeRFDist(trees2, dist);
            ntrees2 = dist.size();
            rfdist.insert(rfdist.end(), dist.begin(), dist.end());
            char ch;
            in.exceptions(ios::goodbit);
            (in) >> ch;
            if (in.eof()) break;
            in.unget();
            in.exceptions(ios::failbit | ios::badbit);

        }

        in.close();
        ASSERT(ntrees * ntrees2 == rfdist.size());
        rfdist_raw = new double[rfdist.size()];
        copy(rfdist.begin(), rfdist.end(), rfdist_raw);

    } catch (ios::failure) {
        outError(ERR_READ_INPUT, trees1);
    }

    printRFDist(filename, rfdist_raw, ntrees, ntrees2, RF_TWO_TREE_SETS_EXTENDED);
    delete [] rfdist_raw;
}

void computeRFDistSamePair(const char *trees1, const char *trees2, const char *filename) {
    cout << "Reading input trees 1 file " << trees1 << endl;
    int ntrees = 0, ntrees2 = 0;
    double *rfdist_raw = NULL;
    try {
        ifstream in;
        in.exceptions(ios::failbit | ios::badbit);
        in.open(trees1);

        ifstream in2;
        in2.exceptions(ios::failbit | ios::badbit);
        in2.open(trees2);

        DoubleVector rfdist;
        for (ntrees = 1; !in.eof() && !in2.eof(); ntrees++) {
            MTree tree;
            bool is_rooted = false;
            // read in the tree and convert into split system for indexing
            tree.readTree(in, is_rooted);

            if (verbose_mode >= VB_DEBUG)
                cout << ntrees << " " << endl;
            DoubleVector dist;
            tree.computeRFDist(in2, dist, 0, true);
            ntrees2 = dist.size();
            rfdist.insert(rfdist.end(), dist.begin(), dist.end());
            char ch;
            in.exceptions(ios::goodbit);
            (in) >> ch;
            if (in.eof()) break;
            in.unget();
            in.exceptions(ios::failbit | ios::badbit);
            
        }
        
        in.close();
        in2.close();
        ASSERT(ntrees * ntrees2 == rfdist.size());
        rfdist_raw = new double[rfdist.size()];
        copy(rfdist.begin(), rfdist.end(), rfdist_raw);
        
    } catch (ios::failure) {
        outError(ERR_READ_INPUT, trees1);
    }
    
    printRFDist(filename, rfdist_raw, ntrees, ntrees2, RF_TWO_TREE_SETS_EXTENDED);

    delete [] rfdist_raw;
}

void computeRFDist(Params &params) {

    if (!params.user_file) outError("User tree file not provided");

    string filename = params.out_prefix;
    filename += ".rfdist";

    if (params.rf_dist_mode == RF_TWO_TREE_SETS_EXTENDED) {
        computeRFDistExtended(params.user_file, params.second_tree, filename.c_str());
        return;
    }

    if (params.rf_same_pair) {
        computeRFDistSamePair(params.user_file, params.second_tree, filename.c_str());
        return;
    }

    MTreeSet trees(params.user_file, params.is_rooted, params.tree_burnin, params.tree_max_count);
    int n = trees.size(), m = trees.size();
    double *rfdist;
    double *incomp_splits = NULL;
    string infoname = params.out_prefix;
    infoname += ".rfinfo";
    string treename = params.out_prefix;
    treename += ".rftree";
    if (params.rf_dist_mode == RF_TWO_TREE_SETS) {
        MTreeSet treeset2(params.second_tree, params.is_rooted, params.tree_burnin, params.tree_max_count);
        cout << "Computing Robinson-Foulds distances between two sets of trees" << endl;
        m = treeset2.size();
        size_t size = n*m;
        if (params.rf_same_pair) {
            if (m != n)
                outError("Tree sets has different number of trees");
            size = n;
        }
        rfdist = new double [size];
        memset(rfdist, 0, size*sizeof(double));
        if (verbose_mode >= VB_MAX) {
            incomp_splits = new double [size];
            memset(incomp_splits, 0, size*sizeof(double));
        }
        if (verbose_mode >= VB_MED)
            trees.computeRFDist(rfdist, &treeset2, params.rf_same_pair,
                                infoname.c_str(),treename.c_str(), incomp_splits);
        else
            trees.computeRFDist(rfdist, &treeset2, params.rf_same_pair);
    } else {
        rfdist = new double [n*n];
        memset(rfdist, 0, n*n* sizeof(double));
        trees.computeRFDist(rfdist, params.rf_dist_mode, params.split_weight_threshold);
    }

    //if (verbose_mode >= VB_MED) printRFDist(cout, rfdist, n, m, params.rf_dist_mode);

    printRFDist(filename, rfdist, n, m, params.rf_dist_mode);

    if (incomp_splits) {
        filename = params.out_prefix;
        filename += ".incomp";
        printRFDist(filename, incomp_splits, n, m, params.rf_dist_mode, false);
        cout << "Number of incompatible splits in printed to " << filename << endl;
    }

    if (incomp_splits) delete [] incomp_splits;
    delete [] rfdist;
}


void testInputFile(Params &params) {
    SplitGraph sg(params);
    if (sg.isWeaklyCompatible())
        cout << "The split system is weakly compatible." << endl;
    else
        cout << "The split system is NOT weakly compatible." << endl;

}

/**MINH ANH: for some statistics about the branches on the input tree*/
void branchStats(Params &params){
    MaTree mytree(params.user_file, params.is_rooted);
    mytree.drawTree(cout,WT_TAXON_ID + WT_INT_NODE);
    //report to output file
    string output;
    if (params.out_file)
        output = params.out_file;
    else {
        if (params.out_prefix)
            output = params.out_prefix;
        else
            output = params.user_file;
        output += ".stats";
    }

    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(output.c_str());
        mytree.printBrInfo(out);
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, output);
    }
    cout << "Information about branch lengths of the tree is printed to: " << output << endl;
    
    /***** Following added by BQM to print internal branch lengths */
    NodeVector nodes1, nodes2;
    mytree.generateNNIBraches(nodes1, nodes2);
    output = params.out_prefix;
    output += ".inlen";
    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(output.c_str());
        for (int i = 0; i < nodes1.size(); i++)
            out << nodes1[i]->findNeighbor(nodes2[i])->length << " ";
        out << endl;
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, output);
    }
    cout << "Internal branch lengths printed to: " << output << endl;
}

/**MINH ANH: for comparison between the input tree and each tree in a given set of trees*/
void compare(Params &params){
    MaTree mytree(params.second_tree, params.is_rooted);
    //sort taxon names and update nodeID, to be consistent with MTreeSet
    NodeVector taxa;
    mytree.getTaxa(taxa);
    sort(taxa.begin(), taxa.end(), nodenamecmp);
    int i;
    NodeVector::iterator it;
    for (it = taxa.begin(), i = 0; it != taxa.end(); it++, i++)
            (*it)->id = i;

    string drawFile = params.second_tree;
    drawFile += ".draw";
    try {
        ofstream out1;
        out1.exceptions(ios::failbit | ios::badbit);
        out1.open(drawFile.c_str());
        mytree.drawTree(out1,WT_TAXON_ID + WT_INT_NODE);
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, drawFile);
    }
    cout << "Tree with branchID (nodeID) was printed to: " << drawFile << endl;


    MTreeSet trees(params.user_file,params.is_rooted, params.tree_burnin, params.tree_max_count);
    DoubleMatrix brMatrix;
    DoubleVector BSDs;
    IntVector RFs;
    mytree.comparedTo(trees, brMatrix, RFs, BSDs);
    int numTree = trees.size();
    int numNode = mytree.nodeNum;

    string output;
    if (params.out_file)
        output = params.out_file;
    else {
        if (params.out_prefix)
            output = params.out_prefix;
        else
            output = params.user_file;
        output += ".compare";
    }

    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(output.c_str());
        //print the header
        out << "tree  " ;
        for (int nodeID = 0; nodeID < numNode; nodeID++ )
            if ( brMatrix[0][nodeID] != -2 )
                out << "br_" << nodeID << "  ";
        out << "RF  BSD" << endl;
        for ( int treeID = 0; treeID < numTree; treeID++ )
        {
            out << treeID << "  ";
            for (int nodeID = 0; nodeID < numNode; nodeID++ )
                if ( brMatrix[treeID][nodeID] != -2 )
                    out << brMatrix[treeID][nodeID] << "  ";
            out << RFs[treeID] << "  " << BSDs[treeID] << endl;
        }
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, output);
    }
    cout << "Comparison with the given set of trees is printed to: " << output << endl;
}

/**MINH ANH: to compute 'guided bootstrap' alignment*/
void guidedBootstrap(Params &params)
{
    MaAlignment inputAlign(params.aln_file,params.sequence_type, params.intype, params.model_name);
    inputAlign.readLogLL(params.siteLL_file);

    string outFre_name = params.out_prefix;
    outFre_name += ".patInfo";

    inputAlign.printPatObsExpFre(outFre_name.c_str());

    string gboAln_name = params.out_prefix;
    gboAln_name += ".gbo";

    MaAlignment gboAlign;
    double prob;
    gboAlign.generateExpectedAlignment(&inputAlign, prob);
    gboAlign.printAlignment(IN_PHYLIP, gboAln_name.c_str());


    string outProb_name = params.out_prefix;
    outProb_name += ".gbo.logP";
    try {
        ofstream outProb;
        outProb.exceptions(ios::failbit | ios::badbit);
        outProb.open(outProb_name.c_str());
        outProb.precision(10);
        outProb << prob << endl;
        outProb.close();
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, outProb_name);
    }

    cout << "Information about patterns in the input alignment is printed to: " << outFre_name << endl;
    cout << "A 'guided bootstrap' alignment is printed to: " << gboAln_name << endl;
    cout << "Log of the probability of the new alignment is printed to: " << outProb_name << endl;
}

/**MINH ANH: to compute the probability of an alignment given the multinomial distribution of patterns frequencies derived from a reference alignment*/
void computeMulProb(Params &params)
{
    Alignment refAlign(params.second_align, params.sequence_type, params.intype, params.model_name);
    Alignment inputAlign(params.aln_file, params.sequence_type, params.intype, params.model_name);
    double prob;
    inputAlign.multinomialProb(refAlign,prob);
    //Printing
    string outProb_name = params.out_prefix;
    outProb_name += ".mprob";
    try {
        ofstream outProb;
        outProb.exceptions(ios::failbit | ios::badbit);
        outProb.open(outProb_name.c_str());
        outProb.precision(10);
        outProb << prob << endl;
        outProb.close();
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, outProb_name);
    }
    cout << "Probability of alignment " << params.aln_file << " given alignment " << params.second_align << " is: " << prob << endl;
    cout << "The probability is printed to: " << outProb_name << endl;
}

void processNCBITree(Params &params) {
    NCBITree tree;
    Node *dad = tree.readNCBITree(params.user_file, params.ncbi_taxid, params.ncbi_taxon_level, params.ncbi_ignore_level);
    if (params.ncbi_names_file) tree.readNCBINames(params.ncbi_names_file);

    cout << "Dad ID: " << dad->name << " Root ID: " << tree.root->name << endl;
    string str = params.user_file;
    str += ".tree";
    if (params.out_file) str = params.out_file;
    //tree.printTree(str.c_str(), WT_SORT_TAXA | WT_BR_LEN);
    cout << "NCBI tree printed to " << str << endl;
    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(str.c_str());
        tree.printTree(out, WT_SORT_TAXA | WT_BR_LEN | WT_TAXON_ID, tree.root, dad);
        out << ";" << endl;
        out.close();
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, str);
    }
}

/* write simultaneously to cout/cerr and a file */
class outstreambuf : public streambuf {
public:
    outstreambuf* open( const char* name, ios::openmode mode = ios::out);
    bool is_open();
    outstreambuf* close();
    ~outstreambuf() { close(); }
    streambuf *get_fout_buf() {
        return fout_buf;
    }
    streambuf *get_cout_buf() {
        return cout_buf;
    }
    ofstream *get_fout() {
        return &fout;
    }
    
protected:
    ofstream fout;
    streambuf *cout_buf;
    streambuf *fout_buf;
    virtual int     overflow( int c = EOF);
    virtual int     sync();
};

outstreambuf* outstreambuf::open( const char* name, ios::openmode mode) {
    if (!(Params::getInstance().suppress_output_flags & OUT_LOG)) {
        if (MPIHelper::getInstance().isMaster()) {
            fout.open(name, mode);
            if (!fout.is_open()) {
                cerr << "ERROR: Could not open " << name << " for logging" << endl;
                exit(EXIT_FAILURE);
                return NULL;
            }
            fout_buf = fout.rdbuf();
        }
    }
    cout_buf = cout.rdbuf();
    cout.rdbuf(this);
    return this;
}

bool outstreambuf::is_open() {
    return fout.is_open();
}

outstreambuf* outstreambuf::close() {
    cout.rdbuf(cout_buf);
    if ( fout.is_open()) {
        sync();
        fout.close();
        return this;
    }
    return NULL;
}

int outstreambuf::overflow( int c) { // used for output buffer only
    if ((verbose_mode >= VB_MIN && MPIHelper::getInstance().isMaster()) || verbose_mode >= VB_MED)
        if (cout_buf->sputc(c) == EOF) return EOF;
    if (Params::getInstance().suppress_output_flags & OUT_LOG)
        return c;
    if (!MPIHelper::getInstance().isMaster())
        return c;
    if (fout_buf->sputc(c) == EOF) return EOF;
    return c;
}



int outstreambuf::sync() { // used for output buffer only
    if ((verbose_mode >= VB_MIN && MPIHelper::getInstance().isMaster()) || verbose_mode >= VB_MED)
        cout_buf->pubsync();
    if ((Params::getInstance().suppress_output_flags & OUT_LOG) || !MPIHelper::getInstance().isMaster())
        return 0;        
    return fout_buf->pubsync();
}

class errstreambuf : public streambuf {
public:
    void init(streambuf *fout_buf) {
        this->fout_buf = fout_buf;
        cerr_buf = cerr.rdbuf();
        cerr.rdbuf(this);
        new_line = true;
    }
    
    void reset() {
        cerr.rdbuf(cerr_buf);
    }
    
    ~errstreambuf() {
        cerr.rdbuf(cerr_buf);
    }
    
protected:
    streambuf *cerr_buf;
    streambuf *fout_buf;
    bool new_line;
    
    virtual int overflow( int c = EOF) {
        if (new_line)
            cerr_buf->sputn("ERROR: ", 7);
        if (cerr_buf->sputc(c) == EOF) {
            new_line = false;
            if (c == '\n') new_line = true;
            return EOF;
        }
        if ((Params::getInstance().suppress_output_flags & OUT_LOG)) {
            new_line = false;
            if (c == '\n') new_line = true;
            return c;
        }
        if (new_line)
            fout_buf->sputn("ERROR: ", 7);
        new_line = false;
        if (c == '\n') new_line = true;
        if (fout_buf->sputc(c) == EOF) return EOF;
        return c;
    }
    
    virtual int sync() {
        cerr_buf->pubsync();
        if (Params::getInstance().suppress_output_flags & OUT_LOG)
            return 0;        
        return fout_buf->pubsync();
    }
};

class muststreambuf : public streambuf {
public:
    void init(streambuf *cout_buf, streambuf *fout_buf) {
        this->fout_buf = fout_buf;
        this->cout_buf = cout_buf;
    }
    
protected:
    streambuf *cout_buf;
    streambuf *fout_buf;
    
    virtual int overflow( int c = EOF) {
        if (cout_buf->sputc(c) == EOF) {
            return EOF;
        }
        if (fout_buf->sputc(c) == EOF) return EOF;
        return c;
    }
    
    virtual int sync() {
        cout_buf->pubsync();
        return fout_buf->pubsync();
    }
};


/*********************************************************************************
 * GLOBAL VARIABLES
 *********************************************************************************/
outstreambuf _out_buf;
errstreambuf _err_buf;
muststreambuf _must_buf;
ostream cmust(&_must_buf);

string _log_file;
int _exit_wait_optn = FALSE;

extern "C" void startLogFile(bool append_log) {
    if (append_log)
        _out_buf.open(_log_file.c_str(), ios::app);
    else
        _out_buf.open(_log_file.c_str());
    _err_buf.init(_out_buf.get_fout_buf());
    _must_buf.init(_out_buf.get_cout_buf(), _out_buf.get_fout_buf());
}

extern "C" void endLogFile() {
    if (_out_buf.is_open())
        _out_buf.close();
    _err_buf.reset();
}

void funcExit(void) {
    if(_exit_wait_optn) {
        printf("\npress [return] to finish: ");
        fflush(stdout);
        while (getchar() != '\n');
    }
    
    endLogFile();
    MPIHelper::getInstance().finalize();
}

extern "C" void funcAbort(int signal_number)
{
    /*Your code goes here. You can output debugging info.
      If you return from this function, and it was called 
      because abort() was called, your program will exit or crash anyway
      (with a dialog box on Windows).
     */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(WIN32) && !defined(WIN64) && !defined(__CYGWIN__)
    print_stacktrace(cerr);
#endif

    cerr << endl << "*** IQ-TREE CRASHES WITH SIGNAL ";
    switch (signal_number) {
        case SIGABRT: cerr << "ABORTED"; break;
        case SIGFPE:  cerr << "ERRONEOUS NUMERIC"; break;
        case SIGILL:  cerr << "ILLEGAL INSTRUCTION"; break;
        case SIGSEGV: cerr << "SEGMENTATION FAULT"; break;
#if !defined WIN32 && !defined _WIN32 && !defined __WIN32__ && !defined WIN64
        case SIGBUS: cerr << "BUS ERROR"; break;
#endif
    }
    cerr << endl;
    cerr << "*** For bug report please send to developers:" << endl << "***    Log file: " << _log_file;
    cerr << endl << "***    Alignment files (if possible)" << endl;
    funcExit();
    signal(signal_number, SIG_DFL);
}

extern "C" void getintargv(int *argc, char **argv[]) 
{
    int    done;
    int    count;
    int    n;
    int    l;
    char   ch;
    char  *argtmp;
    char **argstr;

    argtmp = (char  *)calloc(10100, sizeof(char));
    argstr = (char **)calloc(100, sizeof(char*));
    for(n=0; n<100; n++) {
        argstr[n] = &(argtmp[n * 100]);
    }
    n=1;

    fprintf(stdout, "\nYou seem to have click-started this program,");
    fprintf(stdout, "\ndo you want to enter commandline parameters: [y]es, [n]o: ");
    fflush(stdout);

    /* read one char */
    ch = getc(stdin);
    if (ch != '\n') {
        do ;
        while (getc(stdin) != '\n');
    }
    ch = (char) tolower((int) ch);

    if (ch == 'y') {
        done=FALSE;

        fprintf(stdout, "\nEnter single parameter [! for none]: ");
        fflush(stdout);
        count = fscanf(stdin, "%s", argstr[n]);
        do ;
        while (getc(stdin) != '\n');

        if(argstr[0][0] == '!') {
            count = 0;
        } else {
            if (strlen(argstr[n]) > 100) {
                fprintf(stdout, "\nParameter too long!!!\n");
            } else {
                n++;
            }
        }

        while(!done) {
            fprintf(stdout, "\nCurrent commandline: ");
            for(l=1; l<n; l++) {
                fprintf(stdout, "%s ", argstr[l]);
            }
            fprintf(stdout, "\nQuit [q]; confirm [y]%s%s%s: ",
                (n<99 ? ", extend [e]" : ""),
                (n>1 ? ", delete last [l]" : ""),
                (n>1 ? ", delete all [a]" : ""));
            fflush(stdout);

            /* read one char */
            ch = getc(stdin);
            /* ch = getchar(); */
            if (ch != '\n') {
                do ;
                while (getc(stdin) != '\n');
                /* while (getchar() != '\n'); */
            }
            ch = (char) tolower((int) ch);
        
            switch (ch) {
                case 'y': 
                    done=TRUE;
                    break;
                case 'e': 
                    fprintf(stdout, "\nEnter single parameter [! for none]: ");
                    fflush(stdout);
                    count = fscanf(stdin, "%s", argstr[n]);
                    do ;
                    while (getc(stdin) != '\n');
        
                    if(argstr[0][0] == '!') {
                        count = 0;
                    } else {
                        if (strlen(argstr[n]) > 100) {
                            fprintf(stdout, "\nParameter too long!!!\n");
                        } else {
                            n++;
                        }
                    }
                    break;
                case 'l': 
                    if (n>1) n--;
                    break;
                case 'a': 
                    n=1;
                    break;
                case 'q': 
                       // tp_exit(0, NULL, FALSE, __FILE__, __LINE__, _exit_wait_optn);
                    if(_exit_wait_optn) {
                        printf("\npress [return] to finish: ");
                        fflush(stdout);
                        while (getchar() != '\n');
                    }
                    exit(0);
                    break;
            }
        }
    }

    *argc = n;
    *argv = argstr;
} /* getintargv */

/*********************************************************************************************************************************
    Olga: ECOpd - phylogenetic diversity with ecological constraint: choosing a viable subset of species which maximizes PD/SD
*********************************************************************************************************************************/

void processECOpd(Params &params) {
    double startTime = getCPUTime();
    params.detected_mode = RunMode::LINEAR_PROGRAMMING;
    cout<<"----------------------------------------------------------------------------------------"<<endl;
    int i;
    double score;
    double *variables;
    int threads = params.gurobi_threads;
    params.gurobi_format=true;

    string model_file,subFoodWeb,outFile;

    model_file = params.out_prefix;
    model_file += ".lp";

    subFoodWeb = params.out_prefix;
    subFoodWeb += ".subFoodWeb";

    outFile = params.out_prefix;
    outFile += ".pda";

    //Defining the input phylo type: t - rooted/unrooted tree, n - split network
    params.intype=detectInputFile(params.user_file);
    if(params.intype == IN_NEWICK){
        params.eco_type = "t";
    } else if(params.intype == IN_NEXUS){
        params.eco_type = "n";
    }

    // Checking whether to treat the food web as weighted or non weighted
    if(params.diet_max == 0){
        params.eco_weighted = false;
    }else if(params.diet_max > 100 || params.diet_max < 0){
        cout<<"The minimum percentage of the diet to be conserved for each predator"<<endl;
        cout<<"d = "<<params.diet_max<<endl;
        cout<<"ERROR: Wrong value of parameter d. It must be within the range 0 <= d <= 100"<<endl;
        exit(0);
    }else{
        params.eco_weighted = true;
    }

    if(strcmp(params.eco_type,"t")==0){
    /*--------------------------------- EcoPD Trees ---------------------------------*/
        ECOpd tree(params.user_file,params.is_rooted);

        // Setting all the information-----------------
        tree.phyloType = "t";
        tree.TaxaNUM = tree.leafNum;
        if(verbose_mode == VB_MAX){
            cout<<"TaxaNUM = "<<tree.TaxaNUM<<endl;
            cout<<"LeafNUM = "<<tree.leafNum<<endl;
            cout<<"root_id = "<<tree.root->id<<" root_name = "<<tree.root->name<<endl;

            for(i=0; i<tree.leafNum; i++){
                cout<<i<<" "<<tree.findNodeID(i)->name <<endl;
            }
        }

        //Getting Species Names from tree
        for(i = 0; i < tree.TaxaNUM; i++)
            (tree.phyloNames).push_back(tree.findNodeID(i)->name);
        //for(i=0;i<tree.phyloNames.size();i++)
        //    cout<<"["<<i<<"] "<<tree.phyloNames[i]<<endl;

        // Full species list including info from tree and food web. Here adding names from phyloInput.
        for(i=0; i<tree.TaxaNUM; i++)
            tree.names.push_back(&(tree.phyloNames[i]));

        // Read the taxa to be included in the final optimal subset
        if(params.initial_file)
            tree.readInitialTaxa(params.initial_file);

        // Read the DAG file, Synchronize species on the Tree and in the Food Web
        tree.weighted = params.eco_weighted;
        tree.T = params.diet_max*0.01;
        tree.readDAG(params.eco_dag_file);
        tree.defineK(params);

        // IP formulation
        cout<<"Formulating an IP problem..."<<endl;
        if(tree.rooted){
            tree.printECOlpRooted(model_file.c_str(),tree);
        } else {
            tree.printECOlpUnrooted(model_file.c_str(),tree);
        }

        // Solve IP problem
        cout<<"Solving the problem..."<<endl;
        variables = new double[tree.nvar];
        int g_return = gurobi_solve((char*)model_file.c_str(), tree.nvar, &score, variables, verbose_mode, threads);
        if(verbose_mode == VB_MAX){
            cout<<"GUROBI finished with "<<g_return<<" return."<<endl;
            for(i=0; i<tree.nvar; i++)
                cout<<"x"<<i<<" = "<<variables[i]<<endl;
            cout<<"score = "<<score<<endl;
        }
        tree.dietConserved(variables);
        params.run_time = getCPUTime() - startTime;
        tree.printResults((char*)outFile.c_str(),variables,score,params);
        tree.printSubFoodWeb((char*)subFoodWeb.c_str(),variables);
        delete[] variables;

    } else if(strcmp(params.eco_type,"n")==0){
    /*----------------------------- EcoPD SplitNetwork ------------------------------*/
        params.intype=detectInputFile(params.user_file);
        PDNetwork splitSYS(params);
        ECOpd ecoInfDAG;

        // Get the species names from SplitNetwork
        splitSYS.speciesList(&(ecoInfDAG.phyloNames));
        //for(i=0;i<ecoInfDAG.phyloNames.size();i++)
        //    cout<<"["<<i<<"] "<<ecoInfDAG.phyloNames[i]<<endl;

        ecoInfDAG.phyloType = "n";
        ecoInfDAG.TaxaNUM = splitSYS.getNTaxa();

        // Full species list including info from tree and food web
        for(i=0; i<ecoInfDAG.TaxaNUM; i++)
            ecoInfDAG.names.push_back(&(ecoInfDAG.phyloNames[i]));

        ecoInfDAG.weighted = params.eco_weighted;
        // Read the taxa to be included in the final optimal subset
        if(params.initial_file)
            ecoInfDAG.readInitialTaxa(params.initial_file);
        ecoInfDAG.T = params.diet_max*0.01;
        ecoInfDAG.readDAG(params.eco_dag_file);
        ecoInfDAG.defineK(params);

        cout<<"Formulating an IP problem..."<<endl;
        splitSYS.transformEcoLP(params, model_file.c_str(), 0);
        /**
         * (subset_size-4) - influences constraints for conserved splits.
         * should be less than taxaNUM in the split system.
         * With 0 prints all the constraints.
         * Values different of 0 reduce the # of constraints.
         **/

        ecoInfDAG.printInfDAG(model_file.c_str(),splitSYS,params);
        cout<<"Solving the problem..."<<endl;
        variables = new double[ecoInfDAG.nvar];
        int g_return = gurobi_solve((char*)model_file.c_str(), ecoInfDAG.nvar, &score, variables, verbose_mode, threads);
        if(verbose_mode == VB_MAX){
            cout<<"GUROBI finished with "<<g_return<<" return."<<endl;
            for(i=0; i<ecoInfDAG.nvar; i++)
                cout<<"x"<<i<<" = "<<variables[i]<<endl;
            cout<<"score = "<<score<<endl;
        }
        ecoInfDAG.splitsNUM = splitSYS.getNSplits();
        ecoInfDAG.totalSD = splitSYS.calcWeight();
        ecoInfDAG.dietConserved(variables);
        params.run_time = getCPUTime() - startTime;
        ecoInfDAG.printResults((char*)outFile.c_str(),variables, score,params);
        ecoInfDAG.printSubFoodWeb((char*)subFoodWeb.c_str(),variables);
        delete[] variables;
    }
}

void collapseLowBranchSupport(char *user_file, char *split_threshold_str) {
    DoubleVector minsup;
    convert_double_vec(split_threshold_str, minsup, '/');
    if (minsup.empty())
        outError("wrong -minsupnew argument, please use back-slash separated string");
    MExtTree tree;
    bool isrooted = false;
    tree.readTree(user_file, isrooted);
    tree.collapseLowBranchSupport(minsup);
    tree.collapseZeroBranches(NULL, NULL, -1.0);
    if (verbose_mode >= VB_MED)
        tree.drawTree(cout);
    string outfile = (string)user_file + ".collapsed";
    tree.printTree(outfile.c_str());
    cout << "Tree with collapsed branches written to " << outfile << endl;
}


/********************************************************
    main function
********************************************************/
/*
int main(){
    IQTree tree;
    char * str = "(1, (2, 345));";
    string k;
    tree.pllConvertTaxaID2IQTreeForm(str, k);
    cout << str << endl;
    cout << k << endl;
    cout << "WHAT" << endl;
    return 0;
}
*/

#ifndef BUILD_LIB
int main(int argc, char *argv[]) {

    /*
    Instruction set ID reported by vectorclass::instrset_detect
    0           = 80386 instruction set
    1  or above = SSE (XMM) supported by CPU (not testing for O.S. support)
    2  or above = SSE2
    3  or above = SSE3
    4  or above = Supplementary SSE3 (SSSE3)
    5  or above = SSE4.1
    6  or above = SSE4.2
    7  or above = AVX supported by CPU and operating system
    8  or above = AVX2
    9  or above = AVX512F
    */
    int instruction_set;

    MPIHelper::getInstance().init(argc, argv);
    
    atexit(funcExit);

    /*************************/
    { /* local scope */
        int found = FALSE;              /* "click" found in cmd name? */
        int n, dummyint;
        char *tmpstr;
        int intargc;
        char **intargv;
        intargc = 0;
        intargv = NULL;

        for (n = strlen(argv[0]) - 5;
             (n >= 0) && !found && (argv[0][n] != '/')
             && (argv[0][n] != '\\'); n--) {

            tmpstr = &(argv[0][n]);
            dummyint = 0;
            (void) sscanf(tmpstr, "click%n", &dummyint);
            if (dummyint == 5) found = TRUE;
            else {
                dummyint = 0;
                (void) sscanf(tmpstr, "CLICK%n", &dummyint);
                if (dummyint == 5) found = TRUE;
                else {
                    dummyint = 0;
                    (void) sscanf(tmpstr, "Click%n", &dummyint);
                    if (dummyint == 5) found = TRUE;
                }
            }
        }
        if (found) _exit_wait_optn = TRUE;

        if (_exit_wait_optn) { // get commandline parameters from keyboard
            getintargv(&intargc, &intargv);
            fprintf(stdout, "\n\n");
            if (intargc > 1) { // if there were option entered, use them as argc/argv
                argc = intargc;
                argv = intargv;
            }
        }
    } /* local scope */
    /*************************/

    parseArg(argc, argv, Params::getInstance());

    // 2015-12-05
    Checkpoint *checkpoint = new Checkpoint;
    string filename = (string)Params::getInstance().out_prefix +".ckp.gz";
    checkpoint->setFileName(filename);
    
    bool append_log = false;
    
    if (!Params::getInstance().ignore_checkpoint && fileExists(filename)) {
        checkpoint->load();
        if (checkpoint->hasKey("finished")) {
            if (checkpoint->getBool("finished")) {
                if (Params::getInstance().force_unfinished) {
                    if (MPIHelper::getInstance().isMaster())
                        cout << "NOTE: Continue analysis although a previous run already finished" << endl;
                } else {
                    delete checkpoint;
                    if (MPIHelper::getInstance().isMaster())
                        outError("Checkpoint (" + filename + ") indicates that a previous run successfully finished\n" +
                            "Use `-redo` option if you really want to redo the analysis and overwrite all output files.\n" +
                            "Use `--redo-tree` option if you want to restore ModelFinder and only redo tree search.\n" +
                            "Use `--undo` option if you want to continue previous run when changing/adding options."
                        );
                    else
                        exit(EXIT_SUCCESS);
                    exit(EXIT_FAILURE);
                } 
            } else {
                append_log = true;
            }
        } else {
            if (MPIHelper::getInstance().isMaster())
                outWarning("Ignore invalid checkpoint file " + filename);
            checkpoint->clear();
        }
    }

    if (MPIHelper::getInstance().isWorker())
        checkpoint->setFileName("");

    _log_file = Params::getInstance().out_prefix;
    _log_file += ".log";
    startLogFile(append_log);
    time_t start_time;

    if (append_log) {
        cout << endl << "******************************************************"
             << endl << "CHECKPOINT: Resuming analysis from " << filename << endl << endl;
    }

    MPIHelper::getInstance().syncRandomSeed();
    
    signal(SIGABRT, &funcAbort);
    signal(SIGFPE, &funcAbort);
    signal(SIGILL, &funcAbort);
    signal(SIGSEGV, &funcAbort);
#if !defined WIN32 && !defined _WIN32 && !defined __WIN32__ && !defined WIN64
    signal(SIGBUS, &funcAbort);
#endif
    printCopyright(cout);

    /*
    double x=1e-100;
    double y=1e-101;
    if (x > y) cout << "ok!" << endl;
    else cout << "shit!" << endl;
    */
    //FILE *pfile = popen("hostname","r");
    char hostname[100];
#if defined WIN32 || defined _WIN32 || defined __WIN32__ || defined WIN64
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    gethostname(hostname, sizeof(hostname));
    WSACleanup();
#else
    gethostname(hostname, sizeof(hostname));
#endif
    //fgets(hostname, sizeof(hostname), pfile);
    //pclose(pfile);

    instruction_set = instrset_detect();
#if defined(BINARY32) || defined(__NOAVX__)
    instruction_set = min(instruction_set, (int)LK_SSE42);
#endif
    if (instruction_set < LK_SSE2) outError("Your CPU does not support SSE2!");
    bool has_fma3 = (instruction_set >= LK_AVX) && hasFMA3();

#ifdef __FMA__
    bool has_fma =  has_fma3;
    if (!has_fma) {
        outError("Your CPU does not support FMA instruction, quiting now...");
    }
#endif

    cout << "Host:    " << hostname << " (";
    switch (instruction_set) {
    case 0: cout << "x86, "; break;
    case 1: cout << "SSE, "; break;
    case 2: cout << "SSE2, "; break;
    case 3: cout << "SSE3, "; break;
    case 4: cout << "SSSE3, "; break;
    case 5: cout << "SSE4.1, "; break;
    case 6: cout << "SSE4.2, "; break;
    case 7: cout << "AVX, "; break;
    case 8: cout << "AVX2, "; break;
    default: cout << "AVX512, "; break;
    }
    if (has_fma3) cout << "FMA3, ";
//    if (has_fma4) cout << "FMA4, ";
//#if defined __APPLE__ || defined __MACH__
    cout << (int)(((getMemorySize()/1024.0)/1024)/1024) << " GB RAM)" << endl;
//#else
//    cout << (int)(((getMemorySize()/1000.0)/1000)/1000) << " GB RAM)" << endl;
//#endif

    cout << "Command:";
    int i;
    for (i = 0; i < argc; i++)
        cout << " " << argv[i];
    cout << endl;

    checkpoint->get("iqtree.seed", Params::getInstance().ran_seed);
    cout << "Seed:    " << Params::getInstance().ran_seed <<  " ";
    init_random(Params::getInstance().ran_seed + MPIHelper::getInstance().getProcessID(), true);
    // initialize multiple random streams if needed
    if (Params::getInstance().multi_rstreams_used)
        init_multi_rstreams();

    time(&start_time);
    cout << "Time:    " << ctime(&start_time);

    // increase instruction set level with FMA
    if (has_fma3 && instruction_set < LK_AVX_FMA)
        instruction_set = LK_AVX_FMA;

    Params::getInstance().SSE = min(Params::getInstance().SSE, (LikelihoodKernel)instruction_set);

    cout << "Kernel:  ";

    if (Params::getInstance().lk_safe_scaling) {
        cout << "Safe ";
    }

    if (Params::getInstance().pll) {
#ifdef __AVX__
        cout << "PLL-AVX";
#else
        cout << "PLL-SSE3";
#endif
    } else {
        if (Params::getInstance().SSE >= LK_AVX512)
            cout << "AVX-512";
        else if (Params::getInstance().SSE >= LK_AVX_FMA) {
            cout << "AVX+FMA";
        } else if (Params::getInstance().SSE >= LK_AVX) {
            cout << "AVX";
        } else if (Params::getInstance().SSE >= LK_SSE2){
            cout << "SSE2";
        } else
            cout << "x86";
    }

#ifdef _OPENMP
    if (Params::getInstance().num_threads >= 1) {
        omp_set_num_threads(Params::getInstance().num_threads);
        Params::getInstance().num_threads = omp_get_max_threads();
    }
//    int max_threads = omp_get_max_threads();
    int max_procs = countPhysicalCPUCores();
    cout << " - ";
    if (Params::getInstance().num_threads > 0)
        cout << Params::getInstance().num_threads  << " threads";
    else
        cout << "auto-detect threads";
    cout << " (" << max_procs << " CPU cores detected)";
    if (Params::getInstance().num_threads  > max_procs) {
        cout << endl;
        outError("You have specified more threads than CPU cores available");
    }
    // omp_set_nested(false); // don't allow nested OpenMP parallelism
    omp_set_max_active_levels(1);
#else
    if (Params::getInstance().num_threads != 1) {
        cout << endl << endl;
        outError("Number of threads must be 1 for sequential version.");
    }
#endif

#ifdef _IQTREE_MPI
    cout << endl << "MPI:     " << MPIHelper::getInstance().getNumProcesses() << " processes";
#endif
    
    int num_procs = countPhysicalCPUCores();
#ifdef _OPENMP
    if (num_procs > 1 && Params::getInstance().num_threads == 1 && !Params::getInstance().alisim_active) {
        cout << endl << endl << "HINT: Use -nt option to specify number of threads because your CPU has " << num_procs << " cores!";
        cout << endl << "HINT: -nt AUTO will automatically determine the best number of threads to use.";
    }
#else
    if (num_procs > 1)
        cout << endl << endl << "NOTE: Consider using the multicore version because your CPU has " << num_procs << " cores!";
#endif

    //cout << "sizeof(int)=" << sizeof(int) << endl;
    cout << endl << endl;
    
    // show msgs which are delayed to show
    cout << Params::getInstance().delay_msgs;

    cout.precision(3);
    cout.setf(ios::fixed);
    
    // checkpoint general run information
    checkpoint->startStruct("iqtree");
    string command;
    
    if (CKP_RESTORE_STRING(command)) {
        // compare command between saved and current commands
        stringstream ss(command);
        string str;
        bool mismatch = false;
        for (i = 1; i < argc; i++) {
            if (!(ss >> str)) {
                outWarning("Number of command-line arguments differs from checkpoint");
                mismatch = true;
                break;
            }
            if (str != argv[i]) {
                outWarning((string)"Command-line argument `" + argv[i] + "` differs from checkpoint `" + str + "`");
                mismatch = true;
            }
        }
        if (mismatch) {
            outWarning("Command-line differs from checkpoint!");
        }
        command = "";
    }
    
    for (i = 1; i < argc; i++)
        command += string(" ") + argv[i];
    CKP_SAVE(command);
    int seed = Params::getInstance().ran_seed;
    CKP_SAVE(seed);
    CKP_SAVE(start_time);

    // check for incompatible version
    string version;
    stringstream sversion;
    sversion << iqtree_VERSION_MAJOR << "." << iqtree_VERSION_MINOR << iqtree_VERSION_PATCH;
    version = sversion.str();
    CKP_SAVE(version);
    checkpoint->endStruct();
    
    // load distributions from built-in file and user-specified file
    // load distributions from built-in file
    read_distributions();
    // load distributions from user-specified file
    if (Params::getInstance().alisim_distribution_definitions)
    {
        cout<<"Reading user-specified distributions/lists of random numbers."<<endl;
        read_distributions(Params::getInstance().alisim_distribution_definitions);
    }

    if (MPIHelper::getInstance().getNumProcesses() > 1) {
        if (Params::getInstance().alisim_active) {
            runAliSim(Params::getInstance(), checkpoint);
        } else if (Params::getInstance().aln_file || Params::getInstance().partition_file) {
            runPhyloAnalysis(Params::getInstance(), checkpoint);
        } else {
            outError("Please use one MPI process! The feature you wanted does not need parallelization.");
        }
    } else
    // call the main function
    if (Params::getInstance().alisim_active) {
        runAliSim(Params::getInstance(), checkpoint);
    } else if (Params::getInstance().tree_gen != NONE && Params::getInstance().start_tree!=STT_RANDOM_TREE) {
        generateRandomTree(Params::getInstance());
    } else if (Params::getInstance().do_pars_multistate) {
        doParsMultiState(Params::getInstance());
    } else if (Params::getInstance().rf_dist_mode != 0) {
        computeRFDist(Params::getInstance());
    } else if (Params::getInstance().test_input != TEST_NONE) {
        Params::getInstance().intype = detectInputFile(Params::getInstance().user_file);
        testInputFile(Params::getInstance());
    } else if (Params::getInstance().run_mode == RunMode::PRINT_TAXA) {
        printTaxa(Params::getInstance());
    } else if (Params::getInstance().run_mode == RunMode::PRINT_AREA) {
        printAreaList(Params::getInstance());
    } else if (Params::getInstance().run_mode == RunMode::SCALE_BRANCH_LEN || Params::getInstance().run_mode == RunMode::SCALE_NODE_NAME) {
        scaleBranchLength(Params::getInstance());
    } else if (Params::getInstance().run_mode == RunMode::PD_DISTRIBUTION) {
        calcDistribution(Params::getInstance());
    } else if (Params::getInstance().run_mode == RunMode::STATS){ /**MINH ANH: for some statistics on the input tree*/
        branchStats(Params::getInstance()); // MA
    } else if (Params::getInstance().branch_cluster > 0) {
        calcTreeCluster(Params::getInstance());
    } else if (Params::getInstance().ncbi_taxid) {
        processNCBITree(Params::getInstance());
    } else if (Params::getInstance().user_file && Params::getInstance().eco_dag_file) { /**ECOpd analysis*/
        processECOpd(Params::getInstance());
    } else if (Params::getInstance().gen_all_NNI){
        PhyloTree *tree = new PhyloTree();
        tree->readTree(Params::getInstance().user_file, Params::getInstance().is_rooted);
        tree->gen_all_nni_trees();
    } else if (Params::getInstance().terrace_analysis) { /**Olga: Terrace analysis*/
        runterraceanalysis(Params::getInstance());
    } else if ((Params::getInstance().aln_file || Params::getInstance().partition_file) &&
               Params::getInstance().consensus_type != CT_ASSIGN_SUPPORT_EXTENDED)
    {
        if ((Params::getInstance().siteLL_file || Params::getInstance().second_align) && !Params::getInstance().gbo_replicates)
        {
            if (Params::getInstance().siteLL_file)
                guidedBootstrap(Params::getInstance());
            if (Params::getInstance().second_align)
                computeMulProb(Params::getInstance());
        } else {
            runPhyloAnalysis(Params::getInstance(), checkpoint);
        }
//    } else if (Params::getInstance().ngs_file || Params::getInstance().ngs_mapped_reads) {
//        runNGSAnalysis(Params::getInstance());
//    } else if (Params::getInstance().pdtaxa_file && Params::getInstance().gene_scale_factor >=0.0 && Params::getInstance().gene_pvalue_file) {
//        runGSSAnalysis(Params::getInstance());
    } else if (Params::getInstance().consensus_type != CT_NONE) {
        MExtTree tree;
        switch (Params::getInstance().consensus_type) {
            case CT_CONSENSUS_TREE:
                computeConsensusTree(Params::getInstance().user_file, Params::getInstance().tree_burnin, Params::getInstance().tree_max_count, Params::getInstance().split_threshold,
                    Params::getInstance().split_weight_threshold, Params::getInstance().out_file, Params::getInstance().out_prefix, Params::getInstance().tree_weight_file, &Params::getInstance());
                break;
            case CT_CONSENSUS_NETWORK:
                computeConsensusNetwork(Params::getInstance().user_file, Params::getInstance().tree_burnin, Params::getInstance().tree_max_count, Params::getInstance().split_threshold,
                    Params::getInstance().split_weight_summary, Params::getInstance().split_weight_threshold, Params::getInstance().out_file, Params::getInstance().out_prefix, Params::getInstance().tree_weight_file);
                break;
            case CT_ASSIGN_SUPPORT:
                assignBootstrapSupport(Params::getInstance().user_file, Params::getInstance().tree_burnin, Params::getInstance().tree_max_count, 
                    Params::getInstance().second_tree, Params::getInstance().is_rooted, Params::getInstance().out_file,
                    Params::getInstance().out_prefix, tree, Params::getInstance().tree_weight_file, &Params::getInstance());
                break;
            case CT_ASSIGN_SUPPORT_EXTENDED:
                assignBranchSupportNew(Params::getInstance());
                break;
            case CT_NONE: break;
            /**MINH ANH: for some comparison*/
            case COMPARE: compare(Params::getInstance()); break; //MA
            case CT_ROOTSTRAP:
                runRootstrap(Params::getInstance()); break;
        }
    } else if (Params::getInstance().split_threshold_str) {
        // for Ricardo: keep those splits from input tree above given support threshold
        collapseLowBranchSupport(Params::getInstance().user_file, Params::getInstance().split_threshold_str);
    } else {
        Params::getInstance().intype = detectInputFile(Params::getInstance().user_file);
        if (Params::getInstance().intype == IN_NEWICK && Params::getInstance().pdtaxa_file && Params::getInstance().tree_gen == NONE) {
            if (Params::getInstance().budget_file) {
                //if (Params::getInstance().budget < 0) Params::getInstance().run_mode = PD_USER_SET;
            } else {
                if (Params::getInstance().sub_size < 1 && Params::getInstance().pd_proportion == 0.0)
                    Params::getInstance().run_mode = RunMode::PD_USER_SET;
            }
            // input is a tree, check if it is a reserve selection -> convert to splits
            if (Params::getInstance().run_mode != RunMode::PD_USER_SET) Params::getInstance().multi_tree = true;
        }


        if (Params::getInstance().intype == IN_NEWICK && !Params::getInstance().find_all && Params::getInstance().budget_file == NULL &&
            Params::getInstance().find_pd_min == false && Params::getInstance().calc_pdgain == false &&
            Params::getInstance().run_mode != RunMode::LINEAR_PROGRAMMING && Params::getInstance().multi_tree == false)
            runPDTree(Params::getInstance());
        else if (Params::getInstance().intype == IN_NEXUS || Params::getInstance().intype == IN_NEWICK) {
            if (Params::getInstance().run_mode == RunMode::LINEAR_PROGRAMMING && Params::getInstance().find_pd_min)
                outError("Current linear programming does not support finding minimal PD sets!");
            if (Params::getInstance().find_all && Params::getInstance().run_mode == RunMode::LINEAR_PROGRAMMING)
                Params::getInstance().binary_programming = true;
            runPDSplit(Params::getInstance());
        } else {
            outError("Unknown file input format");
        }
    }

    time(&start_time);
    cout << "Date and Time: " << ctime(&start_time);
    try{
    delete checkpoint;
    }catch(int err_num){}

    finish_random();
    // finish multiple random streams if used
    if (Params::getInstance().multi_rstreams_used)
        finish_multi_rstreams();
    
    return EXIT_SUCCESS;
}
#else
// library now
#include "libiqtree_fun.h"

#if defined WIN32 || defined _WIN32 || defined __WIN32__ || defined WIN64
#include <winsock2.h>
#endif

class input_options {
public:
    vector<string> flags;
    vector<string> values;

    void insert(string flag, string value="") {
        flags.push_back(flag);
        values.push_back(value);
    }
    
    // set Params according to the input options from PiQTREE
    // only invoke this function after the default values of parameters are set
    // this function defines which IQ-TREE options are availble for PiQTREE
    void set_params(Params& params);
};

void cleanup(Params& params) {
    if (params.state_freq_set != NULL) {
        delete[] params.state_freq_set;
        params.state_freq_set = NULL;
    }
}

void convertToVectorStr(StringArray& names, StringArray& seqs, vector<string>& names_vec, vector<string>& seqs_vec) {
    names_vec.clear();
    seqs_vec.clear();
    if (names.length != seqs.length)
        outError("The number of names must equal to the number of sequences");
    for (int i = 0; i < names.length; i++) {
        names_vec.push_back(string(names.strings[i]));
        seqs_vec.push_back(string(seqs.strings[i]));
    }
}

char* build_phylogenetic(StringArray& cnames, StringArray& cseqs, const char* cmodel, const char* cintree,
                          int rand_seed, string& prog, input_options* in_options, const char* other_options);

// Calculates the robinson fould distance between two trees
extern "C" IntegerResult robinson_fould(const char* ctree1, const char* ctree2) {
    IntegerResult output;
    output.errorStr = strdup("");

    try {
        string tree1 = string(ctree1);
        string tree2 = string(ctree2);

        MTree first_tree;
        bool is_rooted = false;
        std::vector<double> rfdist;

        // read in the first tree
        first_tree.read_TreeString(tree1, is_rooted);
        
        // second tree
        stringstream second_tree_str;
        second_tree_str << tree2;
        second_tree_str.seekg(0, ios::beg);
        
        // compute the RF distance
        first_tree.computeRFDist(second_tree_str, rfdist);
        
        output.value = (int)rfdist[0];
    } catch (const exception& e) {
        if (strlen(e.what()) > 0) {
            output.errorStr = new char[strlen(e.what())+1];
            strcpy(output.errorStr, e.what());
        }
        // reset the output and error buffers
        funcExit();
    }

    return output;
}

// Generates a set of random phylogenetic trees
// tree_gen_mode allows:"YULE_HARDING", "UNIFORM", "CATERPILLAR", "BALANCED", "BIRTH_DEATH", "STAR_TREE"
extern "C" StringResult random_tree(int num_taxa, const char* tree_gen_mode, int num_trees, int rand_seed) {
    StringResult output;
    output.errorStr = strdup("");
    
    try {
        ostringstream ostring;
        PhyloTree ptree;
        int seed = rand_seed;
        if (seed == 0)
            seed = make_new_seed();
        cout << "seed: " << seed << endl;
        init_random(seed);
        
        TreeGenType tree_mode;
        if (strcmp(tree_gen_mode, "YULE_HARDING")==0) {
            tree_mode = YULE_HARDING;
        } else if (strcmp(tree_gen_mode, "UNIFORM")==0) {
            tree_mode = UNIFORM;
        } else if (strcmp(tree_gen_mode, "CATERPILLAR")==0) {
            tree_mode = CATERPILLAR;
        } else if (strcmp(tree_gen_mode, "BALANCED")==0) {
            tree_mode = BALANCED;
        } else if (strcmp(tree_gen_mode, "BIRTH_DEATH")==0) {
            tree_mode = BIRTH_DEATH;
        } else if (strcmp(tree_gen_mode, "STAR_TREE")==0) {
            tree_mode = STAR_TREE;
        } else {
            outError("Unknown mode: " + string(tree_gen_mode));
        }
        
        Params params = Params::getInstance();
        params.setDefault();
        params.sub_size = num_taxa;
        params.tree_gen = tree_mode;
        params.repeated_time = num_trees;
        params.ignore_checkpoint = true; // overrid the output file if exists
        params.user_file = (char*) "";
        
        generateRandomTree(params, ostring);
        string s = ostring.str();
        
        if (s.length() > 0) {
            output.value = new char[s.length()+1];
            strcpy(output.value, s.c_str());
        }
        
    } catch (const exception& e) {
        if (strlen(e.what()) > 0) {
            output.errorStr = new char[strlen(e.what())+1];
            strcpy(output.errorStr, e.what());
        }
        // reset the output and error buffers
        funcExit();
    }
    return output;
}

// Perform phylogenetic analysis on the input alignment (in string format)
// With estimation of the best topology
// num_thres -- number of cpu threads to be used, default: 1; 0 - auto detection of the optimal number of cpu threads
extern "C" StringResult build_tree(StringArray& names, StringArray& seqs, const char* model, int rand_seed, int bootstrap_rep, int num_thres, const char* other_options) {
    StringResult output;
    output.errorStr = strdup("");
    
    try {
        const char* intree = "";
        input_options* in_options = NULL;
        if (bootstrap_rep > 0 || num_thres >= 0) {
            in_options = new input_options();
            if (bootstrap_rep > 0)
                in_options->insert("-bb", convertIntToString(bootstrap_rep));
            if (num_thres >= 0)
                in_options->insert("-nt", convertIntToString(num_thres));
        }
        string prog = "build_tree";
        output.value = build_phylogenetic(names, seqs, model, intree, rand_seed, prog, in_options, other_options);
        if (in_options != NULL)
            delete in_options;
    } catch (const exception& e) {
        if (strlen(e.what()) > 0) {
            output.errorStr = new char[strlen(e.what())+1];
            strcpy(output.errorStr, e.what());
        }
        // reset the output and error buffers
        funcExit();
    }
    return output;
}

// Perform phylogenetic analysis on the input alignment (in string format)
// With restriction to the input toplogy
// blfix -- whether to fix the branch length as those on the given tree, default: false
// num_thres -- number of cpu threads to be used, default: 1; 0 - auto detection of the optimal number of cpu threads
extern "C" StringResult fit_tree(StringArray& names, StringArray& seqs, const char* model, const char* intree, bool blfix, int rand_seed, int num_thres, const char* other_options) {
    StringResult output;
    output.errorStr = strdup("");
    
    try {
        input_options* in_options = NULL;
        if (num_thres >= 0 || blfix) {
            in_options = new input_options();
            if (num_thres >= 0)
                in_options->insert("-nt", convertIntToString(num_thres));
            if (blfix)
                in_options->insert("-blfix", "");
        }
        string prog = "fit_tree";
        output.value = build_phylogenetic(names, seqs, model, intree, rand_seed, prog, in_options, other_options);
        if (in_options != NULL)
            delete in_options;
    } catch (const exception& e) {
        if (strlen(e.what()) > 0) {
            output.errorStr = new char[strlen(e.what())+1];
            strcpy(output.errorStr, e.what());
        }
        // reset the output and error buffers
        funcExit();
    }
    return output;
}

// Perform phylogenetic analysis with ModelFinder
// on the input alignment (in string format)
// model_set -- a set of models to consider
// freq_set -- a set of frequency types
// rate_set -- a set of RHAS models
// num_thres -- number of cpu threads to be used, default: 1; 0 - auto detection of the optimal number of cpu threads
extern "C" StringResult modelfinder(StringArray& names, StringArray& seqs, int rand_seed, const char* model_set, const char* freq_set, const char* rate_set, int num_thres, const char* other_options) {
    StringResult output;
    output.errorStr = strdup("");
    
    try {
        input_options* in_options = NULL;
        const char* intree = "";
        const char* model = "MF"; // modelfinder
        int i;
        in_options = new input_options();
        // handle model_set, freq_set, rate_set
        if (strlen(model_set) > 0)
            in_options->insert("-mset", string(model_set));
        if (strlen(freq_set) > 0)
            in_options->insert("-mfreq", string(freq_set));
        if (strlen(rate_set) > 0)
            in_options->insert("-mrate", string(rate_set));
        if (num_thres >= 0)
            in_options->insert("-nt", convertIntToString(num_thres));
        string prog = "modelfinder";
        output.value = build_phylogenetic(names, seqs, model, intree, rand_seed, prog, in_options, other_options);
        
        delete in_options;
    } catch (const exception& e) {
        if (strlen(e.what()) > 0) {
            output.errorStr = new char[strlen(e.what())+1];
            strcpy(output.errorStr, e.what());
        }
        // reset the output and error buffers
        funcExit();
    }
    return output;
}

// Build pairwise JC distance matrix
// output: set of distances
// (n * i + j)-th element of the list represents the distance between i-th and j-th sequence,
// where n is the number of sequences
// num_thres -- number of cpu threads to be used, default: 1; 0 - use all available cpu threads on the machine
extern "C" DoubleArrayResult build_distmatrix(StringArray& cnames, StringArray& cseqs, int num_thres) {
    DoubleArrayResult output;
    output.errorStr = strdup("");
    
    try {
        string prog = "build_matrix";
        output.length = 0;
        output.value = NULL;
        vector<string> names;
        vector<string> seqs;
        convertToVectorStr(cnames, cseqs, names, seqs);
        int n = names.size();
        int n_sq = n * n;
        if (n_sq >= 1) {
            output.length = n_sq;
            output.value = new double[n_sq];
        }
        if (n == 1) {
            output.value[0] = 0.0;
        } else if (n > 1) {
            // extern VerboseMode verbose_mode;
            progress_display::setProgressDisplay(false);
            verbose_mode = VB_QUIET; // (quiet mode)
            Params params = Params::getInstance();
            params.setDefault();

            int rand_seed = make_new_seed();
            string out_prefix_str = prog + "_" + convertIntToString(rand_seed);
            _log_file = out_prefix_str + ".log";
            bool append_log = false;
            startLogFile(append_log);

        #ifdef _OPENMP
            int max_procs = countPhysicalCPUCores();
            if (num_thres > max_procs)
                num_thres = max_procs;
            if (num_thres > 0) {
                Params::getInstance().num_threads = num_thres;
                omp_set_num_threads(num_thres);
            } else if (num_thres == 0) {
                Params::getInstance().num_threads = max_procs;
                omp_set_num_threads(max_procs);
            }
         #endif
            
            PhyloTree ptree;
            ptree.aln = new Alignment(names, seqs, params.sequence_type, params.model_name);
            
            
            // compute the matrix
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
            #endif
            for (int i = 0; i < n; i++) {
                double* dmat = &output.value[i * n];
                dmat[i] = 0.0;
                for (int j = i+1; j < n; j++) {
                    dmat[j] = ptree.aln->computeJCDist(i, j);
                    output.value[j * n + i] = dmat[j];
                }
            }
            
            delete ptree.aln;
            funcExit();
        }
    } catch (const exception& e) {
        if (strlen(e.what()) > 0) {
            output.errorStr = new char[strlen(e.what())+1];
            strcpy(output.errorStr, e.what());
        }
        // reset the output and error buffers
        funcExit();
    }
    return output;
}

// Using Rapid-NJ to build tree from a distance matrix
extern "C" StringResult build_njtree(StringArray& cnames, DoubleArray& distances) {
    StringResult output;
    output.errorStr = strdup("");

    try {
        // convert to vector
        vector<string> names;
        for (int i = 0; i < cnames.length; i++) {
            names.push_back(string(cnames.strings[i]));
        }
        // check the size of names and distances
        if (names.size() < 3)
            outError("The size of names must be at least 3");
        size_t n = names.size();
        size_t sq_n = n * n;
        if (distances.length != sq_n)
            outError("The size of distances must equal to the square of the size of names");
        
        string prog = "build_njtree";
        // extern VerboseMode verbose_mode;
        progress_display::setProgressDisplay(false);
        verbose_mode = VB_QUIET; // (quiet mode)
        Params params = Params::getInstance();
        params.setDefault();
        
        int rand_seed = make_new_seed();
        string out_prefix_str = prog + "_" + convertIntToString(rand_seed);
        _log_file = out_prefix_str + ".log";
        bool append_log = false;
        startLogFile(append_log);
        
        string algn_name = "NJ-R"; // Rapid NJ
        StartTree::BuilderInterface* algorithm = StartTree::Factory::getTreeBuilderByName(algn_name);
        stringstream stree;
        if (!algorithm->constructTreeInMemory2(names, distances.doubles, stree)) {
            outError("Tree construction failed.");
        }
        string s = stree.str();
        if (s.length() > 0) {
            output.value = new char[s.length()+1];
            strcpy(output.value, s.c_str());
        }
        funcExit();
    } catch (const exception& e) {
        if (strlen(e.what()) > 0) {
            output.errorStr = new char[strlen(e.what())+1];
            strcpy(output.errorStr, e.what());
        }
        // reset the output and error buffers
        funcExit();
    }
    return output;
}

/*
 * Compute a consensus tree
 * trees -- a set of input trees
 * minsup -- a threshold to build the majority consensus, default is 0.0
 * output: the consensus tree of the set of input trees
 */
extern "C" StringResult consensus_tree(StringArray& trees, double minsup) {

    StringResult output;
    output.errorStr = strdup("");

    try {

        double max_split_threshold = 0.9999;

           // convert to vector
           vector<string> trees_vec;
        for (int i = 0; i < trees.length; i++) {
            trees_vec.push_back(string(trees.strings[i]));
        }
        if (trees.length < 1) {
            outError("Error! The number of input trees < 1");
        }

        Params params = Params::getInstance();
        params.setDefault();
        params.split_threshold = minsup;
        if (params.split_threshold > max_split_threshold)
            params.split_threshold = max_split_threshold;
        bool rooted = false;

        // // get a set of taxa names
        vector<string> taxa_names;
        MTree mtree;
        mtree.read_TreeString(trees_vec[0], rooted);
        mtree.getTaxaName(taxa_names);

        // read the trees
        MTreeSet boot_trees;
        boot_trees.init(trees_vec, taxa_names, rooted);

        SplitGraph sg;
        SplitIntMap hash_ss;
        double scale = 100.0;
        if (params.scaling_factor > 0)
            scale = params.scaling_factor;
        boot_trees.convertSplits(sg, params.split_threshold, SW_COUNT, params.split_weight_threshold);
        scale /= boot_trees.sumTreeWeights();
        cout << sg.size() << " splits found" << endl;

        if (params.scaling_factor < 0)
            sg.scaleWeight(scale, true);
        else {
            sg.scaleWeight(scale, false, params.numeric_precision);
        }

        MTree mytree;
        SplitGraph maxsg;
        sg.findMaxCompatibleSplits(maxsg);
        mytree.convertToTree(maxsg);
        if (!mytree.rooted) {
            string taxname;
            if (params.root)
                taxname = params.root;
            else
                taxname = sg.getTaxa()->GetTaxonLabel(0);
            Node *node = mytree.findLeafName(taxname);
            if (node)
                mytree.root = node;
        }

           ostringstream ostring;
        mytree.printTree(ostring, WT_BR_CLADE);
        string s = ostring.str();
        if (s.length() > 0) {
            output.value = new char[s.length()+1];
            strcpy(output.value, s.c_str());
        }

    } catch (const exception& e) {
        if (strlen(e.what()) > 0) {
            output.errorStr = new char[strlen(e.what())+1];
            strcpy(output.errorStr, e.what());
        }
        // reset the output and error buffers
        funcExit();
    }
    return output;
}

// verion number
extern "C" StringResult version() {
    StringResult output;
    output.errorStr = strdup("");
    try {
        stringstream ss;
        ss << iqtree_VERSION_MAJOR << "." << iqtree_VERSION_MINOR << iqtree_VERSION_PATCH;
        string s = ss.str();
        if (s.length() > 0) {
            output.value = new char[s.length()+1];
            strcpy(output.value, s.c_str());
        }
    } catch (const exception& e) {
        if (strlen(e.what()) > 0) {
            output.errorStr = new char[strlen(e.what())+1];
            strcpy(output.errorStr, e.what());
        }
    }
    return output;
}

// Execute AliSim Simulation
// output: results in YAML format that contains the simulated alignment and the content of the log file
// tree -- the NEWICK tree string
// subst_model -- the substitution model name
// seed -- the random seed
// partition_info -- partition information
// partition_type -- partition type is either ‘equal’, ‘proportion’, or ‘unlinked’
// seq_length -- the length of sequences
// insertion_rate -- the insertion rate
// deletion_rate -- the deletion rate
// root_seq -- the root sequence
// num_threads -- the number of threads
// insertion_size_distribution -- the insertion size distribution
// deletion_size_distribution -- the deletion size distribution
// population_size -- the population size
extern "C" StringResult simulate_alignment(const char* tree, const char* subst_model, int seed, const char* partition_info, const char* partition_type, int seq_length, double insertion_rate, double deletion_rate, const char* root_seq, int num_threads, const char* insertion_size_distribution, const char* deletion_size_distribution, int population_size) {
    
    // verbose_mode
    // extern VerboseMode verbose_mode;
    /*progress_display::setProgressDisplay(false);
    // verbose_mode = VB_MIN;
    verbose_mode = VB_QUIET; // (quiet mode)*/
    
    StringResult output;
    output.errorStr = strdup("");
    randstream = nullptr;
    
    try {
        Params& params = Params::getInstance();
        params.setDefault();
        
        params.alisim_active = true;
        params.multi_rstreams_used = true;
        params.alisim_output_filename = (char*) "AliSimAlignment";
        params.out_prefix = (char*) "AliSimTrees.nwk";
        params.aln_output_format = IN_FASTA;
        // set the population size, if specified
        if (population_size != -1)
        {
            // validate the input
            if (population_size <= 0)
                outError("Population size must be positive!");
            
            // set the scaling factor
            params.alisim_branch_scale = 0.5 / population_size;
        }
        // make sure seed must be positive
        seed = abs(seed);
        params.ran_seed = seed;
        init_random(params.ran_seed);
        // initialize multiple random streams if needed
        if (params.multi_rstreams_used)
            init_multi_rstreams();
        
        // load distributions from built-in file
        read_distributions();
        
        bool append_log = true;
        _log_file = params.out_prefix;
        _log_file += ".log";
        startLogFile(append_log);
        cout << "Start of the log file:" << endl; // This line seems to be vital...
        cout << "Seed: " << params.ran_seed << endl;
        
        params.user_file = params.out_prefix;
        ofstream trees_file(params.user_file);
        if (!trees_file.is_open())
            outError("Failed to create or open the trees file for writing.");
        if (!tree || tree[0] == '\0')
            outError("The input tree is null.");
        trees_file << tree << endl;
        trees_file.close();
        
        params.model_name = subst_model;
        
        if((partition_type == nullptr || strcmp(partition_type, "") == 0) && (partition_info && partition_info[0] != '\0'))
            outError("When partition info is provided, partition type must be provided.");
        else if(partition_type != nullptr && strcmp(partition_type, "") != 0) {
            if(strcmp(partition_type, "equal") == 0) {
                params.partition_type = BRLEN_FIX;
                params.optimize_alg_gammai = "Brent";
                params.opt_gammai = false;
            }
            else if(strcmp(partition_type, "proportion") == 0) {
                params.partition_type = BRLEN_SCALE;
                params.opt_gammai = false;
            }
            else if(strcmp(partition_type, "unlinked") != 0)
                outError("Partition type can be equal, proportion, or unlinked.");
            params.partition_file = (char*) "AliSimPartitionInfo.nex";
            ofstream partition_info_file(params.partition_file);
            if (!partition_info_file.is_open())
                outError("Failed to create or open the partition info file for writing.");
            partition_info_file << partition_info << std::endl;
            partition_info_file.close();
        }
        
        if (seq_length < 1)
            outError("Positive sequence please.");
        params.alisim_sequence_length = seq_length;
        
        if (insertion_rate < 0)
            outError("Insertion rate must not be negative.");
        params.alisim_insertion_ratio = insertion_rate;
        if (deletion_rate < 0)
            outError("Deletion rate must not be negative.");
        params.alisim_deletion_ratio = deletion_rate;
        
        if(root_seq != nullptr && strcmp(root_seq, "") != 0) {
            params.root_ref_seq_aln = "AliSimRootSequence.fasta";
            ofstream root_seq_file(params.root_ref_seq_aln);
            if (!root_seq_file.is_open())
                outError("Failed to create or open the root sequence file for writing.");
            root_seq_file << ">root" << endl;
            root_seq_file << root_seq << endl;
            root_seq_file.close();
            params.root_ref_seq_name = "root";
        }
        
        if (num_threads < 0)
            outError("Number of threads must not be negative.");
        params.num_threads = num_threads;
        
    #ifdef _OPENMP
        if (params.num_threads >= 1) {
            omp_set_num_threads(params.num_threads);
            params.num_threads = omp_get_max_threads();
        }
    //    int max_threads = omp_get_max_threads();
        int max_procs = countPhysicalCPUCores();
        cout << " - ";
        if (params.num_threads > 0)
            cout << params.num_threads  << " threads";
        else
            cout << "auto-detect threads";
        cout << " (" << max_procs << " CPU cores detected)";
        if (params.num_threads  > max_procs) {
            cout << endl;
            outError("You have specified more threads than CPU cores available.");
        }
        // omp_set_nested(false); // don't allow nested OpenMP parallelism
        omp_set_max_active_levels(1);
    #else
        if (params.num_threads != 1) {
            cout << endl << endl;
            outError("Number of threads must be 1 for sequential version.");
        }
    #endif
        cout << endl;
        
        if(insertion_size_distribution != nullptr && strcmp(insertion_size_distribution, "") != 0)
            params.alisim_insertion_distribution = parseIndelDis(insertion_size_distribution, "Insertion");
        if(deletion_size_distribution != nullptr && strcmp(deletion_size_distribution, "") != 0)
            params.alisim_deletion_distribution = parseIndelDis(deletion_size_distribution, "Deletion");
        
        IQTree* iqtree_ptr = nullptr;
        executeSimulation(params, iqtree_ptr);
                
        ostringstream yamlss;
        string line;
        yamlss << "alignment: |" << endl;
        ifstream in_alignment("AliSimAlignment.fa");
        if (!in_alignment)
            outError("Failed to open the alignment file.");
        while (safeGetline(in_alignment, line))
            yamlss << "  " << line << endl;
        in_alignment.close();
        yamlss << endl << "log: |" << endl;
        ifstream in_log(_log_file);
        if (!in_log)
            outError("Failed to open the log file.");
        while (safeGetline(in_log, line))
            yamlss << "  " << line << endl;
        in_log.close();
        
        string yamlstr = yamlss.str();
        if (yamlstr.length() > 0) {
            output.value = new char[yamlstr.length() + 1];
            strcpy(output.value, yamlstr.c_str());
        }
        
        finish_random();
        // finish multiple random streams if used
        if (params.multi_rstreams_used)
            finish_multi_rstreams();
        
        funcExit();
    } catch (const exception& e) {
        if (strlen(e.what()) > 0) {
            output.errorStr = new char[strlen(e.what()) + 1];
            strcpy(output.errorStr, e.what());
        }
        
        if (randstream != nullptr)
            finish_random();
        // finish multiple random streams if used
        if (Params::getInstance().multi_rstreams_used)
            finish_multi_rstreams();
        
        funcExit();
    }
    return output;
}

// ----------------------------------------------
// function for performing plylogenetic analysis
// ----------------------------------------------

// split the input string according to space
void split(const char* s, vector<char*>& out) {
    char* buf = strdup(s);
    char* token;
    
    out.clear();
    token = strtok(buf, " ");
    while (token != nullptr) {
        out.push_back(token);
        token = strtok(nullptr, " ");
    }
}

// Perform phylogenetic analysis on the input alignment (in string format)
// if intree exists, then the topology will be restricted to the intree
char* build_phylogenetic(StringArray& cnames, StringArray& cseqs, const char* cmodel, const char* cintree,
                          int rand_seed, string& prog, input_options* in_options, const char* other_options) {
    // perform phylogenetic analysis on the input sequences
    // all sequences have to be the same length

    int instruction_set;
    
    vector<string> names, seqs;
    
    extern VerboseMode verbose_mode;
    progress_display::setProgressDisplay(false);

    if (rand_seed == 0)
        rand_seed = make_new_seed();
    Params::getInstance().ran_seed = rand_seed;
    // cout << "Seed: " << Params::getInstance().ran_seed << endl << flush;
    init_random(Params::getInstance().ran_seed);

    string out_prefix_str = prog + "_" + convertIntToString(rand_seed);
    Params::getInstance().out_prefix = (char *) out_prefix_str.c_str();

    Checkpoint *checkpoint = new Checkpoint;
    string filename = (string)Params::getInstance().out_prefix +".ckp.gz";
    checkpoint->setFileName(filename);

    bool append_log = false;

    if (!Params::getInstance().ignore_checkpoint && fileExists(filename)) {
        checkpoint->load();
        if (checkpoint->hasKey("finished")) {
            if (checkpoint->getBool("finished")) {
                if (Params::getInstance().force_unfinished) {
                    if (MPIHelper::getInstance().isMaster())
                        cout << "NOTE: Continue analysis although a previous run already finished" << endl;
                } else {
                    delete checkpoint;
                    if (MPIHelper::getInstance().isMaster())
                        outError("Checkpoint (" + filename + ") indicates that a previous run successfully finished\n" +
                            "Use `-redo` option if you really want to redo the analysis and overwrite all output files.\n" +
                            "Use `--redo-tree` option if you want to restore ModelFinder and only redo tree search.\n" +
                            "Use `--undo` option if you want to continue previous run when changing/adding options."
                        );
                    else
                        exit(EXIT_SUCCESS);
                    exit(EXIT_FAILURE);
                }
            } else {
                append_log = true;
            }
        } else {
            if (MPIHelper::getInstance().isMaster())
                outWarning("Ignore invalid checkpoint file " + filename);
            checkpoint->clear();
        }
    }

    if (MPIHelper::getInstance().isWorker())
        checkpoint->setFileName("");

    _log_file = Params::getInstance().out_prefix;
    _log_file += ".log";
    startLogFile(append_log);
    
    char* oprefix = Params::getInstance().out_prefix;
    
    convertToVectorStr(cnames, cseqs, names, seqs);
    string model = string(cmodel);
    string intree = string(cintree);
    
    // checking whether all seqs are in the same length
    if (seqs.size() > 0) {
        int slen = seqs[0].length();
        for (int i=1; i<seqs.size(); i++) {
            if (seqs[i].length() != slen) {
                outError("The input sequences are not in the same length");
            }
        }
    }

    if (other_options != NULL && strlen(other_options) > 0) {
        vector<char*> tokens;
        split(other_options, tokens);
        if (tokens.size() > 0) {
            char** arr = new char*[tokens.size()+3];
            arr[0] = (char*) "";
            arr[1] = (char*) "-s";
            arr[2] = (char*)"dummy";
            for (size_t i = 0; i < tokens.size(); i++) {
                arr[i+3] = tokens[i];
            }
            parseArg(tokens.size()+3, arr, Params::getInstance());
            delete[] arr;
        }
    } else {
        Params::getInstance().setDefault();
    }
    verbose_mode = VB_QUIET; // (quiet mode)

    Params::getInstance().aln_file = (char*) "";
    Params::getInstance().model_name = model;
    Params::getInstance().ignore_identical_seqs = false; // keep the identical seqs
    
    if (intree != "") {
        // tree exists, then the resulting phylogenetic tree will be restricted to the input topology
        Params::getInstance().min_iterations = 0;
        Params::getInstance().stop_condition = SC_FIXED_ITERATION;
        Params::getInstance().start_tree = STT_USER_TREE;
        Params::getInstance().intree_str = intree;
    }

    if (in_options != NULL) {
        // assign the input options to Params
        in_options->set_params(Params::getInstance());
    }
    
    Params::getInstance().out_prefix = oprefix;

    time_t start_time;

    if (append_log) {
        cout << endl << "******************************************************"
             << endl << "CHECKPOINT: Resuming analysis from " << filename << endl << endl;
    }

    MPIHelper::getInstance().syncRandomSeed();

    signal(SIGABRT, &funcAbort);
    signal(SIGFPE, &funcAbort);
    signal(SIGILL, &funcAbort);
    signal(SIGSEGV, &funcAbort);
#if !defined WIN32 && !defined _WIN32 && !defined __WIN32__ && !defined WIN64
    signal(SIGBUS, &funcAbort);
#endif
    printCopyright(cout);

    char hostname[100];
#if defined WIN32 || defined _WIN32 || defined __WIN32__ || defined WIN64
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    gethostname(hostname, sizeof(hostname));
    WSACleanup();
#else
    gethostname(hostname, sizeof(hostname));
#endif

    instruction_set = instrset_detect();
#if defined(BINARY32) || defined(__NOAVX__)
    instruction_set = min(instruction_set, (int)LK_SSE42);
#endif
    if (instruction_set < LK_SSE2) outError("Your CPU does not support SSE2!");
    bool has_fma3 = (instruction_set >= LK_AVX) && hasFMA3();

#ifdef __FMA__
    bool has_fma =  has_fma3;
    if (!has_fma) {
        outError("Your CPU does not support FMA instruction, quiting now...");
    }
#endif

    cout << "Host:    " << hostname << " (";
    switch (instruction_set) {
    case 0: cout << "x86, "; break;
    case 1: cout << "SSE, "; break;
    case 2: cout << "SSE2, "; break;
    case 3: cout << "SSE3, "; break;
    case 4: cout << "SSSE3, "; break;
    case 5: cout << "SSE4.1, "; break;
    case 6: cout << "SSE4.2, "; break;
    case 7: cout << "AVX, "; break;
    case 8: cout << "AVX2, "; break;
    default: cout << "AVX512, "; break;
    }
    if (has_fma3) cout << "FMA3, ";
    cout << (int)(((getMemorySize()/1024.0)/1024)/1024) << " GB RAM)" << endl;

    time(&start_time);
    cout << "Time:    " << ctime(&start_time);

    // increase instruction set level with FMA
    if (has_fma3 && instruction_set < LK_AVX_FMA)
        instruction_set = LK_AVX_FMA;

    Params::getInstance().SSE = min(Params::getInstance().SSE, (LikelihoodKernel)instruction_set);

    cout << "Kernel:  ";

    if (Params::getInstance().lk_safe_scaling) {
        cout << "Safe ";
    }

    if (Params::getInstance().pll) {
#ifdef __AVX__
        cout << "PLL-AVX";
#else
        cout << "PLL-SSE3";
#endif
    } else {
        if (Params::getInstance().SSE >= LK_AVX512)
            cout << "AVX-512";
        else if (Params::getInstance().SSE >= LK_AVX_FMA) {
            cout << "AVX+FMA";
        } else if (Params::getInstance().SSE >= LK_AVX) {
            cout << "AVX";
        } else if (Params::getInstance().SSE >= LK_SSE2){
            cout << "SSE2";
        } else
            cout << "x86";
    }

#ifdef _OPENMP
    if (Params::getInstance().num_threads >= 1) {
        omp_set_num_threads(Params::getInstance().num_threads);
        Params::getInstance().num_threads = omp_get_max_threads();
    }
//    int max_threads = omp_get_max_threads();
    int max_procs = countPhysicalCPUCores();
    cout << " - ";
    if (Params::getInstance().num_threads > 0)
        cout << Params::getInstance().num_threads  << " threads";
    else
        cout << "auto-detect threads";
    cout << " (" << max_procs << " CPU cores detected)";
    if (Params::getInstance().num_threads  > max_procs) {
        cout << endl;
        outError("You have specified more threads than CPU cores available");
    }
    // omp_set_nested(false); // don't allow nested OpenMP parallelism
    omp_set_max_active_levels(1);
#else
    if (Params::getInstance().num_threads != 1) {
        cout << endl << endl;
        outError("Number of threads must be 1 for sequential version.");
    }
#endif

    int num_procs = countPhysicalCPUCores();

    //cout << "sizeof(int)=" << sizeof(int) << endl;
    cout << endl << endl;
    
    // show msgs which are delayed to show
    cout << Params::getInstance().delay_msgs;

    cout.precision(3);
    cout.setf(ios::fixed);
    
    // checkpoint general run information
    checkpoint->startStruct("iqtree");
    int seed = Params::getInstance().ran_seed;
    CKP_SAVE(seed);
    CKP_SAVE(start_time);

    // check for incompatible version
    string version;
    stringstream sversion;
    sversion << iqtree_VERSION_MAJOR << "." << iqtree_VERSION_MINOR << iqtree_VERSION_PATCH;
    version = sversion.str();
    CKP_SAVE(version);
    checkpoint->endStruct();
    
    Params params = Params::getInstance();
    IQTree *tree;
    Alignment *alignment = new Alignment(names, seqs, params.sequence_type, params.model_name);
    bool align_is_given = true;
    ModelCheckpoint* model_info = NULL;
    if (model == "MF") {
        model_info = new ModelCheckpoint;
    }

    runPhyloAnalysis(params, checkpoint, tree, alignment, align_is_given, model_info);
    
    stringstream ss;
    if (model_info != NULL) {
        // output the modelfinder results in YAML format
        model_info->dump(ss);
    } else {
        // output the checkpoint in YAML format
        checkpoint->dump(ss);
    }
    
    alignment = tree->aln;
    delete tree;
    delete alignment;
    cleanup(params);

    time(&start_time);
    cout << "Date and Time: " << ctime(&start_time);
    try{
        delete checkpoint;
        if (model_info != NULL)
            delete model_info;
    }catch(int err_num){}

    finish_random();
    funcExit();

    string output = ss.str();
    if (output.length() > 0) {
        char* out_cstr = new char[output.length()+1];
        strcpy(out_cstr, output.c_str());
        return out_cstr;
    } else {
        static char empty[] = "";
        return empty;
    }
}

/*
 * free the pointer
 */
extern "C" void iqtree_free(void *p) {
    if (p)
        free(p);
}

// --------------------------------------------------
// Handle the input options of PiQTREE
// --------------------------------------------------

void input_options::set_params(Params& params) {
    ASSERT(flags.size() == values.size());
    int n = flags.size();
    for (int i = 0; i < n; i++) {
        if (flags[i] == "-keep-indent") {
            params.ignore_identical_seqs = false;
        }
        else if (flags[i] == "-mset") {
            params.model_set = values[i];
        }
        else if (flags[i] == "-mfreq") {
            int clen = values[i].length();
            if (clen > 0) {
                params.state_freq_set = new char[clen + 1];
                strcpy(params.state_freq_set, values[i].c_str());
            }
        }
        else if (flags[i] == "-mrate") {
            params.ratehet_set = values[i];
        }
        else if (flags[i] == "-bb") {
            params.gbo_replicates = atoi(values[i].c_str());
            if (params.gbo_replicates < 1000)
                outError("#replicates must be >= 1000");
            params.consensus_type = CT_CONSENSUS_TREE;
            params.stop_condition = SC_BOOTSTRAP_CORRELATION;
        }
        else if (flags[i] == "-nt") {
            params.num_threads = atoi(values[i].c_str());
        }
        else if (flags[i] == "-blfix") {
            params.fixed_branch_length = BRLEN_FIX;
            params.optimize_alg_gammai = "Brent";
            params.opt_gammai = false;
            params.min_iterations = 0;
            params.stop_condition = SC_FIXED_ITERATION;
        }
    }
}
#endif // BUILD_LIB

