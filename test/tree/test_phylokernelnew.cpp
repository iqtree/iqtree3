#include <gtest/gtest.h>
#include "tree/phylokernelnew.h"
#include "vectorclass/vectorclass.h"

#define KERNEL_FIX_STATES
#include "tree/phylokernelnew.h"
// test SumVec in phylokernelnew.h
TEST(PhylokernelNewTest, sumVec) {
    Vec4d A[4] = {
            Vec4d(1.0, 2.0, 3.0, 4.0),
            Vec4d(1.0, 2.0, 3.0, 4.0),
            Vec4d(1.0, 2.0, 3.0, 4.0),
            Vec4d(1.0, 2.0, 3.0, 4.0)
    };



    Vec4d X(4.0, 8.0, 12.0, 16.0);
    Vec4d R(0.0, 0.0, 0.0, 0.0);

    sumVec<Vec4d, true>(A, R, 4);


    for (int i = 0; i < 8; ++i) {
        EXPECT_DOUBLE_EQ(R[i], X[i]);
    }
}

// test computeBounds in phylokernelnew.h
/*
TEST(ComputeBoundsRealVectorClass, Vec4dBasicTest) {
    vector<size_t> limits;
    int threads = 2;
    int packets = 4;
    size_t elements = 18;  // not a multiple of 4 (Vec4d size)

    computeBounds<Vec4d>(threads, packets, elements, limits);

    // Vec4d::size() == 4, so elements rounded up to 20
    EXPECT_EQ(limits.size(), packets + 1);
    EXPECT_EQ(limits.front(), 0);
    EXPECT_EQ(limits.back(), 20);

    // Check limits are sorted non-decreasing
    for (size_t i = 1; i < limits.size(); ++i) {
        EXPECT_GE(limits[i], limits[i - 1]);
    }

    // Optional: print limits for manual verification
    for (size_t i = 0; i < limits.size(); ++i) {
        std::cout << "limits[" << i << "] = " << limits[i] << "\n";
    }
}*/
