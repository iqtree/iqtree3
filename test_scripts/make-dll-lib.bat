:: @echo off
setlocal enabledelayedexpansion

:: Build the DLL file and the DEF file
g++ -shared -o iqtree.dll %1 -Wl,--output-def,iqtree.def -liqtree -L. -lws2_32 -lz -lpthread

:: Create the LIB file
lib /def:iqtree.def
