//
// Created by Hashara Kumarasinghe on 16/4/2025.
//

#ifndef IQTREE_VECTORF32_H
#define IQTREE_VECTORF32_H

/*****************************************************************************
 *
 *          Vec1fb: Vector of 1 Booleans for use with Vec1f
 *
 * *****************************************************************************/
#include <algorithm>

class Vec1fb {
public:
    bool xmm; // Float vector
    // Default constructor:
    Vec1fb() {
    }
    // Constructor to broadcast scalar value:
    Vec1fb(bool b) {
        xmm = b;
    }
    // Assignment operator to broadcast scalar value:
    Vec1fb & operator = (bool b) {
        *this = Vec1fb(b);
        return *this;
    }

private: // Prevent constructing from int, etc.
    Vec1fb(int b);
    Vec1fb & operator = (int x);

public:
    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec1fb const & insert(uint32_t index, bool value) {
        xmm = value;
        return *this;
    }
    // Member function extract a single element from vector
    bool extract(uint32_t index) const {
        return xmm;
    }
    // Extract a single element. Operator [] can only read an element, not write.
    bool operator [] (uint32_t index) const {
        return extract(index);
    }
    static int size() {
        return 1;
    }
};

/*****************************************************************************
 *
 *          Operators for Vec1fb
 *
 * *****************************************************************************/

// vector operator & : bitwise AND
static inline Vec1fb operator & (const Vec1fb &a, const Vec1fb &b) {
    return Vec1fb(a.xmm && b.xmm);
}

static inline Vec1fb operator && (const Vec1fb &a, const Vec1fb &b) {
    return Vec1fb(a.xmm && b.xmm);
}

// vector operator &= : bitwise AND
static inline Vec1fb & operator &= (Vec1fb &a, const Vec1fb &b) {
    a = a & b;
    return a;
}

/*****************************************************************************
 *
 *          Vec1f: Vector of 1 Float for use with Vec1fb
 *
 * *****************************************************************************/

class Vec1f {
public:
    float xmm; // Float vector
    // Default constructor:
    Vec1f() {
    }
    // Constructor to broadcast scalar value:
    Vec1f(float f) {
        xmm = f;
    }

    // Member function to load from an array (unaligned)
    Vec1f & load(const float *p) {
        xmm = *p;
        return *this;
    }

    // Member function to load from an array (aligned)
    Vec1f & load_a(const float *p) {
        xmm = *p;
        return *this;
    }

    // Partial load. load n elements and set the rest to zero
    Vec1f & load_partial(int n, const float *p) {
        switch (n) {
            case 1:
                xmm = *p;
                break;
            case 2:
                xmm = 0.0f;
        }
        return *this;
    }

    // Member function to store to an array (unaligned)
    void store(float *p) const {
        *p = xmm;
    }

    // Member function to store to an array (aligned)
    void store_a(float *p) const {
        *p = xmm;
    }

    // cut off vector to n elements. The last 4-n elements are set to zero
    Vec1f & cutoff(int n){
        if (n == 0)
            xmm = 0.0f;
        return *this;
    }

    // Member function to change a single element in vector
    // Note: This function is inefficient. Use load function if changing more than one element
    Vec1f const & insert(uint32_t index, float value) {
        xmm = value;
        return *this;
    };

    // Member function extract a single element from vector
    float extract(uint32_t index) const {
        return xmm;
    }

    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    float operator [] (uint32_t index) const {
        return extract(index);
    }

    static int size() {
        return 1;
    }
};

/*****************************************************************************
 *
 *          Operators for Vec1f
 *
 * *****************************************************************************/

// vector operator + : addition
static inline Vec1f operator + (const Vec1f &a, const Vec1f &b) {
    return Vec1f(a.xmm + b.xmm);
}

// vector operator + : addition with scalar
static inline Vec1f operator + (const Vec1f &a, float b) {
    return a + Vec1f(b);
}

static inline Vec1f operator + (float a, const Vec1f &b) {
    return Vec1f(a) + b;
}

// vector operator += : addition
static inline Vec1f & operator += (Vec1f &a, const Vec1f &b) {
    a = a + b;
    return a;
}

// postfix operator ++
static inline Vec1f operator ++ (Vec1f &a, int) {
    Vec1f a0 = a;
    a = a + 1.0f;
    return a0;
}

// prefix operator ++
static inline Vec1f & operator ++ (Vec1f &a) {
    a = a + 1.0f;
    return a;
}

// vector operator - : subtraction
static inline Vec1f operator - (const Vec1f &a, const Vec1f &b) {
    return Vec1f(a.xmm - b.xmm);
}

// vector operator - : subtraction with scalar
static inline Vec1f operator - (const Vec1f &a, float b) {
    return a - Vec1f(b);
}

static inline Vec1f operator - (float a, const Vec1f &b) {
    return Vec1f(a) - b;
}

// vector operator -: unary minus
// Change sign bit, even for 0, INF and NAN
static inline Vec1f operator - (const Vec1f &a) {
    return Vec1f(-a.xmm);
}

// vector operator -=: subtraction
static inline Vec1f & operator -= (Vec1f &a, const Vec1f &b) {
    a = a - b;
    return a;
}

// postfix operator --
static inline Vec1f operator -- (Vec1f &a, int) {
    Vec1f a0 = a;
    a = a - 1.0f;
    return a0;
}

// prefix operator --
static inline Vec1f & operator -- (Vec1f &a) {
    a = a - 1.0f;
    return a;
}

// vector operator * : multiply element by element
static inline Vec1f operator * (const Vec1f &a, const Vec1f &b) {
    return Vec1f(a.xmm * b.xmm);
}

// vector operator * : multiply vector and scalar
static inline Vec1f operator * (const Vec1f &a, float b) {
    return a * Vec1f(b);
}
static inline Vec1f operator * (float a, const Vec1f &b) {
    return Vec1f(a) * b;
}

// vector operator *= : multiply
static inline Vec1f & operator *= (Vec1f &a, const Vec1f &b) {
    a = a * b;
    return a;
}

// vector operator / : divide element by element
static inline Vec1f operator / (const Vec1f &a, const Vec1f &b) {
    return Vec1f(a.xmm / b.xmm);
}

// vector operator / : divide vector and scalar
static inline Vec1f operator / (const Vec1f &a, float b) {
    return a / Vec1f(b);
}
static inline Vec1f operator / (float a, const Vec1f &b) {
    return Vec1f(a) / b;
}

// vector operator /= : divide
static inline Vec1f & operator /= (Vec1f &a, const Vec1f &b) {
    a = a / b;
    return a;
}

// vector operator == : returns true for elements for which a == b
static inline Vec1fb operator == (const Vec1f &a, const Vec1f &b) {
    return Vec1fb(a.xmm == b.xmm);
}

// vector operator != : returns true for elements for which a != b
static inline Vec1fb operator != (const Vec1f &a, const Vec1f &b) {
    return Vec1fb(a.xmm != b.xmm);
}

// vector operator < : returns true for elements for which a < b
static inline Vec1fb operator < (const Vec1f &a, const Vec1f &b) {
    return Vec1fb(a.xmm < b.xmm);
}

// vector operator <= : returns true for elements for which a <= b
static inline Vec1fb operator <= (const Vec1f &a, const Vec1f &b) {
    return Vec1fb(a.xmm <= b.xmm);
}

// vector operator > : returns true for elements for which a > b
static inline Vec1fb operator > (const Vec1f &a, const Vec1f &b) {
    return Vec1fb(a.xmm > b.xmm);
}

// vector operator >= : returns true for elements for which a >= b
static inline Vec1fb operator >= (const Vec1f &a, const Vec1f &b) {
    return Vec1fb(a.xmm >= b.xmm);
}


// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
static inline float horizontal_add(const Vec1f &a) {
    return a.xmm;
}

// function max: a > b ? a : b
static inline Vec1f max(const Vec1f &a, const Vec1f &b) {
    return std::max(a.xmm,b.xmm);
}

// function min: a < b ? a : b
static inline Vec1f min(const Vec1f &a, const Vec1f &b) {
    return std::min(a.xmm,b.xmm);
}

// function abs: absolute value
// removes sign bit, even for 0, INF and NAN
static inline Vec1f abs(const Vec1f &a) {
    return Vec1f(fabs(a.xmm));
}

// function log: logarithm
static inline Vec1f log(const Vec1f &a) {
    return Vec1f(logf(a.xmm));
}

// Fused multiply and add functions

// Multiply and add
static inline Vec1f mul_add(const Vec1f &a, const Vec1f &b, const Vec1f &c) {
    return Vec1f(a.xmm * b.xmm + c.xmm);
}

// Multiply and subtract
static inline Vec1f mul_sub(const Vec1f &a, const Vec1f &b, const Vec1f &c) {
    return Vec1f(a.xmm * b.xmm - c.xmm);
}

// Multiply and inverse subtract
static inline Vec1f nmul_sub(const Vec1f &a, const Vec1f &b, const Vec1f &c) {
    return Vec1f(c.xmm - a.xmm * b.xmm);
}

/************************************************************************
 *
 *          Horizontal Boolean functions
 *
 * *****************************************************************************/

// horizontal and: Returns true if all elements are true
static inline bool horizontal_and(const Vec1fb &a) {
    return a.xmm;
}

// horizontal or: Returns true if any element is true
static inline bool horizontal_or(const Vec1fb &a) {
    return a.xmm;
}

// instances of exp_f template
static inline Vec1f exp(const Vec1f &a) {
    return Vec1f(expf(a.xmm));
}




#endif //IQTREE_VECTORF32_H

