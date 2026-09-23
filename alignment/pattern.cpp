//
// C++ Implementation: pattern
//
// Description:
//
//
// Author: BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "pattern.h"
#include <vectorclass/vectorclass.h>

bool Pattern::isGapOnly(StateType STATE_UNKNOWN) const {
    for (StateVector::const_iterator i = begin(); i != end(); ++i) {
        if (*i != STATE_UNKNOWN) {
            return false;
        }
    }
    return true;
}

size_t Pattern::countAmbiguousChar(int num_states) const {
    size_t num = 0;
    for (StateVector::const_iterator i = begin(); i != end(); ++i) {
        if (*i >= num_states) {
            num++;
        }
    }
    return num;
}

#define VECTORIZE_GAPCHAR_COUNT 1
size_t Pattern::countGapChar(StateType STATE_UNKNOWN) const {
    size_t num = 0;
#if VECTORIZE_GAPCHAR_COUNT
    //This won't compile unless value_type is based on uint32_t
    //(nor should it! You'd need to use different vector types!)
    const uint32_t* dataStart = data();
    size_type  count   = size();
    size_type  vecSize = Vec8ui::size();
    Vec8ui     unknown = STATE_UNKNOWN;
    const uint32_t* dataStop   = dataStart + count;
    const uint32_t* blockStop  = dataStop - (count & (vecSize-1));
    for (const uint32_t* block=dataStart; block<blockStop; block+=vecSize) {
        Vec8ui a;
        a.load(block);
        num -= horizontal_add( Vec8ui(a == unknown) );
    }
    for (const uint32_t* single=blockStop; single<dataStop; ++single) {
        if (*single == STATE_UNKNOWN) {
            ++num;
        }
    }
#else
    for (StateVector::const_iterator i = begin(); i != end(); ++i) {
        if (*i == STATE_UNKNOWN) {
            num++;
        }
    }
#endif
    return num;
}
