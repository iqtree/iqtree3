#!/bin/bash
if [ $# -eq 0 ];
then
   echo "Syntax: $0 [intel.zip] [arm.zip]"
   exit
fi

intelname=`basename $1 .zip`
armname=`basename $2 .zip`
universalname=`basename $1 -intel.zip`

unzip $1
unzip $2
mkdir $universalname
mkdir $universalname/bin
cp $intelname/*.* $universalname
lipo -create -output $universalname/bin/iqtree3 $intelname/bin/iqtree3 $armname/bin/iqtree3
zip -vr $universalname.zip $universalname -x "*.DS_Store"
