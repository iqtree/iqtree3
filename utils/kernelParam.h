//
// Created by Hashara Kumarasinghe on 25/5/2025.
//

#ifndef IQTREE_KERNELPARAM_H
#define IQTREE_KERNELPARAM_H


class KernelParam {
public:
    // Singleton access
    static KernelParam& getInstance();

    // Getter for X86
    bool isX86() const;

    // Setter for X86
    void setX86(bool value);

private:
    // Private constructor
    KernelParam() = default;

    // Disable copy and assignment
    KernelParam(const KernelParam&) = delete;
    KernelParam& operator=(const KernelParam&) = delete;

private:
    bool X86 = false;
};

#endif //IQTREE_KERNELPARAM_H
