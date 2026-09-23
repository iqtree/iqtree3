//
// C++ Interface: pattern
//
// Description: 
//
//
// Author: BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef PATTERN_H
#define PATTERN_H

#include "phylo-yaml/statespace.h"

// using namespace std;
using namespace PML;

const int PAT_CONST       = 1; // const site pattern, e.g. AAAAAA, CC-C-CCCC
const int PAT_INVARIANT   = 2; // invariant site pattern, including const patterns and e.g., GS--G-GGG (S = G/C)
const int PAT_INFORMATIVE = 4; // parsimony informative sites
const int PAT_VARIANT     = 8; // variant site pattern

/**
	Site-patterns in a multiple sequence alignment
	@author BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>
*/
struct Pattern : StateVector {
    /**
     *  Default constructor
     */
    Pattern() {
        frequency = 1;
        group = -1;
        flag = 0;
        const_char = -1;
        num_chars = 0;
    }

    /**
     *  @param STATE_UNKNOWN The gap of state of the model
     *  @return TRUE if the pattern contains only gaps
     */
    bool isGapOnly(StateType STATE_UNKNOWN) const;

    /**
     *  @param num_states The number of states of the model
     *  @return The number of ambiguous character including gaps
     */
    size_t countAmbiguousChar(int num_states) const;

    /**
     *  @param STATE_UNKNOWN The gap of state of the model
     *  @return The number of gaps
     */
    size_t countGapChar(StateType STATE_UNKNOWN) const;

    inline bool operator==(const Pattern &other) const {
        return group == other.group &&
               static_cast<const StateVector &>(*this) == static_cast<const StateVector &>(other);
    }

    inline bool isConst() const {
        return (flag & PAT_CONST) != 0;
    }

    inline bool isInvariant() const {
        return (flag & PAT_INVARIANT) != 0;
    }

    inline bool isInformative() const {
        return (flag & PAT_INFORMATIVE) != 0;
    }

    /** number of sites with this pattern, default: 1 */
    int frequency;

    /** allows differentiating identical patterns, default: -1 */
    int group;

    /** encodes whether the pattern is const, invar, or inform */
    int flag;

    /** if the pattern is const, stores its const character, default: -1 */
    char const_char;

    /** number of different character states */
    int num_chars;

    // added by TD
    /** character frequencies indexed by StateType */
    vector<size_t> freqs;
};

#endif
