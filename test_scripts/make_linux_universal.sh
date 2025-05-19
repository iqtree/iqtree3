#!/bin/bash
if [ $# -eq 0 ];
then
   echo "Syntax: $0 [*-intel.tar.gz] [*-arm.tar.gz]"
   exit
fi

intelname=`basename $1 .tar.gz`
armname=`basename $2 .tar.gz`
universalname=`basename $1 -intel.tar.gz`

tar -xvf $1
tar -xvf $2
mkdir -p $universalname
mkdir -p $universalname/bin
cp $intelname/*.* $universalname

cp $intelname/bin/iqtree3 $universalname/bin/iqtree3_intel
cp $armname/bin/iqtree3 $universalname/bin/iqtree3_arm

# Make the final file executable
chmod +x test_scripts/linux_universal.sh
cp test_scripts/linux_universal.sh $universalname/bin/iqtree3


rm -rf $intelname
rm -rf $armname

tar czf $universalname.tar.gz $universalname

rm -rf $universalname