//
// Created by Hashara Kumarasinghe on 25/5/2025.
//

#include "kernelParam.h"

KernelParam& KernelParam::getInstance() {
    static KernelParam instance;
    return instance;
}

// Getter
bool KernelParam::isX86() const {
    return X86;
}

// Setter
void KernelParam::setX86(bool value) {
    X86 = value;
}