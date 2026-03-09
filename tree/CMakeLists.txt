add_library(tree
constrainttree.cpp
constrainttree.h
candidateset.cpp candidateset.h
iqtree.cpp
iqtree.h
iqtreemix.cpp
iqtreemix.h
iqtreemixhmm.cpp
iqtreemixhmm.h
matree.cpp
matree.h
memslot.cpp memslot.h
mexttree.cpp
mexttree.h
mtree.cpp
mtree.h
mtreeset.cpp
mtreeset.h
ncbitree.cpp
ncbitree.h
node.cpp
node.h
genomenode.h
genomenode.cpp
genometree.h
genometree.cpp
phylokernel.h
phylokernelnew.h
phylokernelnonrev.h
phylonode.cpp
phylonode.h
phylonodemixlen.cpp
phylonodemixlen.h
phylosuperhmm.cpp
phylosuperhmm.h
phylosupertree.cpp
phylosupertree.h
phylosupertreeplen.cpp
phylosupertreeplen.h
phylosupertreeunlinked.cpp
phylosupertreeunlinked.h
phylotree.cpp
phylotree.h
phylotreemixlen.cpp
phylotreemixlen.h
phylotreepars.cpp
phylotreesse.cpp
quartet.cpp
supernode.cpp
supernode.h
tinatree.cpp
tinatree.h
parstree.cpp
parstree.h
discordance.cpp
phylohmm.cpp
phylohmm.h
phylokernelgpu.h
phylotreegpu.cpp
)

# Conditionally add OpenACC source files
if (USE_OPENACC)
    target_sources(tree PRIVATE
        phylokernel_openacc.h
        phylokernel_openacc.cpp
        phylokernel_openacc_test.h
        phylokernel_openacc_test.cpp
    )
    message("tree library: OpenACC sources added")
endif()

target_link_libraries(tree pll model alignment)