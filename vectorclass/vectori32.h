//
// Created by Hashara Kumarasinghe on 22/4/2025.
//

#ifndef IQTREE_VECTORI32_H
#define IQTREE_VECTORI32_H

/*****************************************************************************
 *
 *           Vector of 32 1-bit unsigned integers or Booleans
 *
 *******************************************************************************/

class Vec32b {
public:
    uint32_t xmm; // 32-bit integer vector
public:
    // Default constructor:
    Vec32b() {
    }
    // Constructor to broadcast scalar value:
    Vec32b(uint32_t x) {
        xmm = x;
    }

    // Assignment operator to covert from type uint32_t to Vec32b:
    Vec32b & operator = (uint32_t const & x) {
        xmm = x;
        return *this;
    }

    // Type cast operator to convert to type uint32_t
    operator uint32_t() const {
        return xmm;
    }

    // Member function to load from array

    Vec32b & load(void const * p) {
        xmm = (uint32_t)(*(uint32_t const*)p);
        return *this;
    }

    // Member function to load from array
    void load_a(void const * p) {
        xmm = (uint32_t)(*(uint32_t const*)p);
    }

    // Member function to store into array (unaligned)
    void store(void * p) const {
        *(uint32_t*)p = xmm;
    }

    // Member function to store into array (aligned)
    void store_a(void * p) const {
        *(uint32_t*)p = xmm;
    }

    //Member function to change a single bit
    Vec32b const & set_bit(uint32_t index, int value){
        if (value)
            xmm |= (1 << index);
        else
            xmm &= ~(1 << index);
        return *this;
    }

    //Member function to get a single bit
    int get_bit(uint32_t index) const {
        return (xmm >> index) & 1;
    }

    // Extract a single element from vector
    bool operator [] (uint32_t index) const {
        return get_bit(index) != 0;
    }

    static int size() {
        return 32;
    }

};

// Define operators for this class

// vector operator & : bitwise and
static inline Vec32b operator & (Vec32b const & a, Vec32b const & b) {
    return Vec32b(a.xmm & b.xmm);
}

static inline Vec32b operator && (Vec32b const & a, Vec32b const & b) {
    return a && b;
}

// vector operator | : bitwise or
static inline Vec32b operator | (Vec32b const & a, Vec32b const & b) {
    return Vec32b(a.xmm | b.xmm);
}

static inline Vec32b operator || (Vec32b const & a, Vec32b const & b) {
    return a || b;
}

// vector operator ^ : bitwise xor
static inline Vec32b operator ^ (Vec32b const & a, Vec32b const & b) {
    return Vec32b(a.xmm ^ b.xmm);
}

// vector operator ~ : bitwise not
static inline Vec32b operator ~ (Vec32b const & a) {
    return Vec32b(~a.xmm);
}

// vector operator &= : bitwise and
static inline Vec32b & operator &= (Vec32b & a, Vec32b const & b) {
    a = a & b;
    return a;
}

// vector operator |= : bitwise or
static inline Vec32b & operator |= (Vec32b & a, Vec32b const & b) {
    a = a | b;
    return a;
}

// vector operator ^= : bitwise xor
static inline Vec32b & operator ^= (Vec32b & a, Vec32b const & b) {
    a = a ^ b;
    return a;
}

// Define functions for this class

// function and not: a & ~b
static inline Vec32b andnot(Vec32b const & a, Vec32b const & b) {
    return Vec32b(a.xmm & ~b.xmm);
}

/******************************************************************************
 *
 *          Generate compile-time constant vector
 *
 *******************************************************************************/
// Generate a constant vector of 1 integer stored in memory
template <int32_t i0>
static inline uint32_t constant1i(){
    static const Vec32b v = Vec32b(i0);
    return v.xmm;
}

template <uint32_t i0>
static inline uint32_t constant1ui(){
    return constant1i<int32_t(i0)>();
}

/*****************************************************************************
 *
 *          selectb function
 *
 *******************************************************************************/
// select between two sources
static inline uint32_t selectb(uint32_t const & s, uint32_t const & a, uint32_t const & b) {
    return (s & a) | (~s & b);
}

/*****************************************************************************
 *
 *          Horizontal Boolean functions
 *
 *******************************************************************************/

// horizontal and: returns true if all elements are 1
static inline bool horizontal_and(Vec32b const & a) {
    return (a.xmm == 0xFFFFFFFF);
}

// horizontal or: returns true if any element is 1
static inline bool horizontal_or(Vec32b const & a) {
    return (a.xmm != 0);
}

/**********************************************************************
 *
 *          Vector of 1 32-bit signed integer
 *
 *******************************************************************************/
class Vec1i: public Vec32b {
public:
    // Default constructor:
    Vec1i() {
    }

    // Constructor to broadcast the same value into all elements:
    Vec1i(int i) {
        xmm = (int32_t) i;
    }

//    // Constructor to build from all elements:
//    Vec1i(int32_t i0) {
//        xmm = i0;
//    }

    // Constructor to build from all elements:
    Vec1i(uint32_t i0) {
        xmm = (int32_t) i0;
    }

    // Assignment operator to convert from type int32_t to Vec1i:
    Vec1i & operator = (int32_t const & i) {
        xmm = i;
        return *this;
    }

    // Type cast operator to convert to type int32_t
    operator int32_t() const {
        return xmm;
    }

    // Member function to load from array (unaligned)
    Vec1i & load(void const * p) {
        xmm = (int32_t)(*(int32_t const*)p);
        return *this;
    }

    // Member function to load from array (aligned)
    Vec1i load_a(void const * p) {
        xmm = (int32_t)(*(int32_t const*)p);
        return *this;
    }

    // Partial load. Load n elements and set the rest to 0
    Vec1i & load_partial(int n, void const * p) {
        if (n >= 1) {
            xmm = *(int32_t const*)p;
        } else {
            xmm = 0;
        }
        return *this;
    }

    // Partial store. Store n elements
    void store_partial(int n, void * p) const {
        if (n >= 1) {
            *(int32_t*)p = xmm;
        }
    }

    // cut off vector to n elements
    Vec1i & cutoff(int n) {
        if (n == 0)
            xmm = 0;
        return *this;
    }

    // Member function to change a single element in vector
    Vec1i const & insert(uint32_t index, int32_t value) {
        if (index == 0)
            xmm = value;
        return *this;
    };

    // Member function extract a single element from vector
    int32_t extract(uint32_t index) const {
        return xmm;
    }

    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    int32_t operator [] (uint32_t index) const {
        return extract(index);
    }

    static int size() {
        return 1;
    }
};


/*****************************************************************************
 *
 *          Vec1ib: Vector of 1 Booleans for use with Vec1i and Vec1ui
 *
 *******************************************************************************/

class Vec1ib: public Vec1i {

public:
    // defult constructor:
    Vec1ib() {
    }

    // Constructor to build from all elements:
    Vec1ib(bool x0){
        xmm = Vec1i(-int32_t(x0));
    }

    // Constructor from int
    Vec1ib(int32_t x) {
        xmm = x;
    }

    // Assignment operator to convert from type int32_t to Vec1ib:
    Vec1ib & operator = (bool b) {
        xmm = -int32_t(b);
        return *this;
    }

private:
    Vec1ib & operator = (int x);

public:
    Vec1ib & insert (int index, bool a){
        Vec1i::insert(index, -(int)a);
        return *this;
    }

    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        return Vec1i::extract(index) != 0;
    }

    // Extract a single element. Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
};

/*****************************************************************************
 *
 *          Define operators for Vec1ib
 *
 *******************************************************************************/

// vector operator & : bitwise and
static inline Vec1ib operator & (Vec1ib const & a, Vec1ib const & b) {
    return Vec1ib(int32_t(Vec32b(a) & Vec32b(b)));
}

static inline Vec1ib operator && (Vec1ib const & a, Vec1ib const & b) {
    return a & b;
}

// vector operator &= : bitwise and
static inline Vec1ib & operator &= (Vec1ib & a, Vec1ib const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec1ib operator | (Vec1ib const & a, Vec1ib const & b) {
    return Vec1ib(int32_t(Vec32b(a) | Vec32b(b)));
}

static inline Vec1ib operator || (Vec1ib const & a, Vec1ib const & b) {
    return a | b;
}

// vector operator |= : bitwise or
static inline Vec1ib & operator |= (Vec1ib & a, Vec1ib const & b) {
    a = a | b;
    return a;
}

 // vector operator ^ : bitwise xor
static inline Vec1ib operator ^ (Vec1ib const & a, Vec1ib const & b) {
    return Vec1ib(int32_t(Vec32b(a) ^ Vec32b(b)));
}

// vector operator ^= : bitwise xor
static inline Vec1ib & operator ^= (Vec1ib & a, Vec1ib const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec1ib operator ~ (Vec1ib const & a) {
    return Vec1ib(int32_t(~Vec32b(a)));
}

// vector operator ! : element not
static inline Vec1ib operator ! (Vec1ib const & a) {
    return Vec1ib(int32_t(~Vec32b(a)));
}

// vector function and not: a & ~b
static inline Vec1ib andnot(Vec1ib const & a, Vec1ib const & b) {
    return Vec1ib(int32_t(andnot(Vec32b(a), Vec32b(b))));
}

// Horizontal Boolean functions for Vec1ib

// horizontal and: returns true if all elements are 1
static inline bool horizontal_and(Vec1ib const & a) {
    return (a.xmm == 0xFFFFFFFF);
}

// horizontal or: returns true if any element is 1
static inline bool horizontal_or(Vec1ib const & a) {
    return (a.xmm != 0);
}

/*****************************************************************************
 *
 *          Operators for Vec1i
 *
 *******************************************************************************/

// vector operator + : add element by element
static inline Vec1i operator + (Vec1i const & a, Vec1i const & b) {
    return Vec1i(a.xmm + b.xmm);
}

// Vec1i + int
static inline Vec1i operator + (Vec1i const & a, int b) {
    return Vec1i((int32_t)a + b);
}

// int + Vec1i
static inline Vec1i operator + (int a, Vec1i const & b) {
    return Vec1i(a + (int32_t)b);
}

// vector operator += : add
static inline Vec1i & operator += (Vec1i & a, Vec1i const & b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec1i operator ++ (Vec1i & a, int) {
    Vec1i a0 = a;
    a = a + 1;
    return a0;
}

// prefix operator ++
static inline Vec1i & operator ++ (Vec1i & a) {
    a = a + 1;
    return a;
}

// vector operator - : subtract element by element
static inline Vec1i operator - (Vec1i const & a, Vec1i const & b) {
    return Vec1i(a.xmm - b.xmm);
}

// vector operator - : unary minus
static inline Vec1i operator - (Vec1i const & a) {
    return Vec1i(-a.xmm);
}

// Vec1i + int
static inline Vec1i operator - (Vec1i const & a, int b) {
    return Vec1i((int32_t)a - b);
}

// int + Vec1i
static inline Vec1i operator - (int a, Vec1i const & b) {
    return Vec1i(a - (int32_t)b);
}


// vector operator -= : subtract
static inline Vec1i & operator -= (Vec1i & a, Vec1i const & b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec1i operator -- (Vec1i & a, int) {
    Vec1i a0 = a;
    a = a - 1;
    return a0;
}

// prefix operator --
static inline Vec1i & operator -- (Vec1i & a) {
    a = a - 1;
    return a;
}

// vector operator * : multiply element by element
static inline Vec1i operator * (Vec1i const & a, Vec1i const & b) {
    return Vec1i(a.xmm * b.xmm);
}

// vector operator *= : multiply
static inline Vec1i & operator *= (Vec1i & a, Vec1i const & b) {
    a = a * b;
    return a;
}

// vector operator << : shift left
static inline Vec1i operator << (Vec1i const & a, int b) {
    return Vec1i(a.xmm << b);
}

// vector operator <<= : shift left
static inline Vec1i & operator <<= (Vec1i & a, int b) {
    a = a << b;
    return a;
}

// vector operator >> : shift right arithmetic
static inline Vec1i operator >> (Vec1i const & a, int b) {
    return Vec1i(a.xmm >> b);
}

// vector operator >>= : shift right arithmetic
static inline Vec1i & operator >>= (Vec1i & a, int b) {
    a = a >> b;
    return a;
}

// vector operator ==: return true if all elements are equal
static inline Vec1ib operator == (Vec1i const & a, Vec1i const & b) {
    return Vec1ib(a.xmm == b.xmm);
}

// vector operator !=: return true if any elements are not equal
static inline Vec1ib operator != (Vec1i const & a, Vec1i const & b) {
    return Vec1ib(a.xmm != b.xmm);
}

// vector operator > : return true fo elements for which a > b
static inline Vec1ib operator > (Vec1i const & a, Vec1i const & b) {
    return Vec1ib(a.xmm > b.xmm);
}

// vector operator < : return true fo elements for which a < b
static inline Vec1ib operator < (Vec1i const & a, Vec1i const & b) {
    return Vec1ib(a.xmm < b.xmm);
}

// vector operator >= : return true fo elements for which a >= b (signed)
static inline Vec1ib operator >= (Vec1i const & a, Vec1i const & b) {
    return Vec1ib(a.xmm >= b.xmm);
}

// vector operator <= : return true fo elements for which a <= b (signed)
static inline Vec1ib operator <= (Vec1i const & a, Vec1i const & b) {
    return Vec1ib(a.xmm <= b.xmm);
}

// vector operator & : bitwise and
static inline Vec1i operator & (Vec1i const & a, Vec1i const & b) {
    return Vec1i(Vec32b(a) & Vec32b(b));
}

static inline Vec1i operator && (Vec1i const & a, Vec1i const & b) {
    return a & b;
}

// vector operator &= : bitwise and
static inline Vec1i & operator &= (Vec1i & a, Vec1i const & b) {
    a = a & b;
    return a;
}

// vector operator | : bitwise or
static inline Vec1i operator | (Vec1i const & a, Vec1i const & b) {
    return Vec1i(Vec32b(a) | Vec32b(b));
}
static inline Vec1i operator || (Vec1i const & a, Vec1i const & b) {
    return a | b;
}

// vector operator |= : bitwise or
static inline Vec1i & operator |= (Vec1i & a, Vec1i const & b) {
    a = a | b;
    return a;
}

// vector operator ^ : bitwise xor
static inline Vec1i operator ^ (Vec1i const & a, Vec1i const & b) {
    return Vec1i(Vec32b(a) ^ Vec32b(b));
}

// vector operator ^= : bitwise xor
static inline Vec1i & operator ^= (Vec1i & a, Vec1i const & b) {
    a = a ^ b;
    return a;
}

// vector operator ~ : bitwise not
static inline Vec1i operator ~ (Vec1i const & a) {
    return Vec1i(~Vec32b(a));
}

// Functions for this class

// select between two operands.
static inline Vec1i select(Vec1ib const & s, Vec1i const & a, Vec1i const & b) {
    return Vec1i(selectb(s, a, b));
}

// conditional add: for all vector elements i: result[i] = f[i] ? a[i] + b[i] : a[i]
static inline Vec1i if_add(Vec1ib const & f, Vec1i const & a, Vec1i const & b) {
    return a + (Vec1i(f) & b);
}

// horizontal add: returns sum of all elements
static inline int32_t horizontal_add(Vec1i const & a) {
    return a.xmm;
}

static inline Vec1i add_saturated(Vec1i const & a, Vec1i const & b) {
    int32_t av = int32_t(a);
    int32_t bv = int32_t(b);
    int32_t sum = av + bv;

    // Detect overflow
    if (((av ^ sum) & (bv ^ sum)) < 0) {
        // Overflow occurred
        if (av > 0) return Vec1i(0x7FFFFFFF);  // INT32_MAX
        else        return Vec1i(0x80000000);  // INT32_MIN
    }
    return Vec1i(sum);
}

static inline Vec1i sub_saturated(Vec1i const & a, Vec1i const & b) {
    int32_t av = int32_t(a);
    int32_t bv = int32_t(b);
    int32_t diff = av - bv;

    // Detect overflow
    if (((av ^ bv) & (av ^ diff)) < 0) {
        // Overflow occurred
        if (av > 0) return Vec1i(0x7FFFFFFF);  // INT32_MAX
        else        return Vec1i(0x80000000);  // INT32_MIN
    }
    return Vec1i(diff);
}


// function max: a > b ? a : b
static inline Vec1i max(Vec1i const & a, Vec1i const & b) {
    return Vec1i(a.xmm > b.xmm ? a.xmm : b.xmm);
}

// function min: a < b ? a : b
static inline Vec1i min(Vec1i const & a, Vec1i const & b) {
    return Vec1i(a.xmm < b.xmm ? a.xmm : b.xmm);
}

// function abs: absolute value
static inline Vec1i abs(Vec1i const & a) {
    return Vec1i(a.xmm < 0 ? -a.xmm : a.xmm);
}

static inline Vec1i abs_saturated(Vec1i const & a) {
    int32_t av = int32_t(a);
    if (av == INT32_MIN) {
        return Vec1i(INT32_MAX); // Saturate
    }
    return Vec1i(av < 0 ? -av : av);
}

// Use negative b to rotate right
static inline Vec1i rotate_left(Vec1i const & a, int b) {
    uint32_t av = uint32_t(int32_t(a));
    b &= 31; // Modulo 32 to keep shift amount valid
    return Vec1i(int32_t((av << b) | (av >> (32 - b))));
}

/***************************************************************************
 *
 *          Vec1ui: Vector of 1 unsigned 32-bit integer
 *
 *******************************************************************************/

class Vec1ui: public Vec1i {
public:
    // Default constructor:
    Vec1ui() {
    }

    // Constructor to broadcast the same value into all elements:
    Vec1ui(uint32_t i) {
        xmm = (int32_t) i;
    }

//    // Constructor to build from all elements:
//    Vec1ui(int32_t i0) {
//        xmm = i0;
//    }

    // Assignment operator to convert from type uint32_t to Vec1ui:
    Vec1ui & operator = (uint32_t const & i) {
        xmm = (int32_t)i;
        return *this;
    }

    // Type cast operator to convert to type uint32_t
    operator uint32_t() const {
        return (uint32_t)xmm;
    }

    // Member function to load from array (unaligned)
    Vec1ui & load(void const * p) {
        xmm = (int32_t)(*(uint32_t const*)p);
        return *this;
    }

    // Member function to load from array (aligned)
    Vec1ui load_a(void const * p) {
        xmm = (int32_t)(*(uint32_t const*)p);
        return *this;
    }

    // Member function to change a single element in vector
    Vec1ui const & insert(uint32_t index, uint32_t value) {
        Vec1i::insert(index, (int32_t)value);
        return *this;
    };

    // Member function extract a single element from vector
    uint32_t extract(uint32_t index) const {
        return Vec1i::extract(index);
    }

    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    uint32_t operator [] (uint32_t index) const {
        return extract(index);
    }

};

// Define operators for this class

// vector operator + : add element by element
static inline Vec1ui operator + (Vec1ui const & a, Vec1ui const & b) {
    return Vec1ui(Vec1i(a) + Vec1i(b));
}

// Vec1ui + uint32_t
inline Vec1ui operator+(const Vec1ui& a, uint32_t b) {
    return Vec1ui(a.xmm + b);
}

// uint32_t + Vec1ui
inline Vec1ui operator+(uint32_t a, const Vec1ui& b) {
    return Vec1ui(a + b.xmm);
}


// vector operator - : subtract element by element
static inline Vec1ui operator - (Vec1ui const & a, Vec1ui const & b) {
    return Vec1ui(Vec1i(a) - Vec1i(b));
}

// vector operator * : multiply element by element
static inline Vec1ui operator * (Vec1ui const & a, Vec1ui const & b) {
    return Vec1ui(Vec1i(a) * Vec1i(b));
}

// vector operator >> : shift right arithmetic
static inline Vec1ui operator >> (Vec1ui const & a, uint32_t b) {
    return Vec1ui(Vec1i(a.xmm >> b));
}

// vector operator >> : shift right arithmetic
static inline Vec1ui operator >> (Vec1ui const & a, int32_t b) {
    return Vec1ui(Vec1i(a.xmm >> b));
}

// vector operator >>= : shift right arithmetic
static inline Vec1ui & operator >>= (Vec1ui & a, int32_t b) {
    a = a >> b;
    return a;
}

// vector operator << : shift left
static inline Vec1ui operator << (Vec1ui const & a, int32_t b) {
    return Vec1ui(Vec1i(a.xmm << b));
}

static inline Vec1ui operator << (Vec1ui const & a, uint32_t b) {
    return Vec1ui(Vec1i(a.xmm << b));
}

// vector operator <<= : shift left
static inline Vec1ui & operator <<= (Vec1ui & a, int32_t b) {
    a = a << b;
    return a;
}

// vector operator > : return true fo elements for which a > b
static inline Vec1ib operator > (Vec1ui const & a, Vec1ui const & b) {
    return Vec1ib(Vec1i(a.xmm) > Vec1i(b.xmm));
}

// vector operator < : return true fo elements for which a < b
static inline Vec1ib operator < (Vec1ui const & a, Vec1ui const & b) {
    return Vec1ib(Vec1i(a.xmm) < Vec1i(b.xmm));
}

// vector operator >= : return true fo elements for which a >= b (unsigned)
static inline Vec1ib operator >= (Vec1ui const & a, Vec1ui const & b) {
    return Vec1ib(Vec1i(a.xmm) >= Vec1i(b.xmm));
}

// vector operator <= : return true fo elements for which a <= b (unsigned)
static inline Vec1ib operator <= (Vec1ui const & a, Vec1ui const & b) {
    return Vec1ib(Vec1i(a.xmm) <= Vec1i(b.xmm));
}

// vector operator &: bitwise and
static inline Vec1ui operator & (Vec1ui const & a, Vec1ui const & b) {
    return Vec1ui(Vec32b(a) & Vec32b(b));
}

static inline Vec1ui operator && (Vec1ui const & a, Vec1ui const & b) {
    return a & b;
}

// vector operator | : bitwise or
static inline Vec1ui operator | (Vec1ui const & a, Vec1ui const & b) {
    return Vec1ui(Vec32b(a) | Vec32b(b));
}

static inline Vec1ui operator || (Vec1ui const & a, Vec1ui const & b) {
    return a | b;
}

// vector operator ^ : bitwise xor
static inline Vec1ui operator ^ (Vec1ui const & a, Vec1ui const & b) {
    return Vec1ui(Vec32b(a) ^ Vec32b(b));
}

// vector operator ~ : bitwise not
static inline Vec1ui operator ~ (Vec1ui const & a) {
    return Vec1ui(~Vec32b(a));
}

// Functions for this class

// select between two operands.
static inline Vec1ui select(Vec1ib const & s, Vec1ui const & a, Vec1ui const & b) {
    return selectb(s, a, b);
}

// conditional add: for all vector elements i: result[i] = f[i] ? a[i] + b[i] : a[i]
static inline Vec1ui if_add(Vec1ib const & f, Vec1ui const & a, Vec1ui const & b) {
    return a + (Vec1ui(f) & b);
}

// horizontal add: returns sum of all elements
static inline uint32_t horizontal_add(Vec1ui const & a) {
    return a.xmm;
}

// function add_saturated: add element by element with saturation
static inline Vec1ui add_saturated(Vec1ui const & a, Vec1ui const & b) {
    return Vec1ui(Vec1i(a) + Vec1i(b));
}

// function sub_saturated: subtract element by element with saturation
static inline Vec1ui sub_saturated(Vec1ui const & a, Vec1ui const & b) {
    return Vec1ui(Vec1i(a) - Vec1i(b));
}

// function max: a > b ? a : b
static inline Vec1ui max(Vec1ui const & a, Vec1ui const & b) {
    return Vec1ui(a.xmm > b.xmm ? a.xmm : b.xmm);
}

// function min: a < b ? a : b
static inline Vec1ui min(Vec1ui const & a, Vec1ui const & b) {
    return Vec1ui(a.xmm < b.xmm ? a.xmm : b.xmm);
}







#endif //IQTREE_VECTORI32_H
