#!/usr/bin/env python

import os
import sys
import shutil
import ROOT

path, out_dir = sys.argv[1:3]

thisdir = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro('%s/from_triggerntuple_ehbinary.cc+' % thisdir)

fname = os.path.basename(path)
tmp_out = '%s/%s' % (os.getenv('TMPDIR', '/tmp'), fname)

chain = ROOT.TChain('hgcalTriggerNtuplizer/HGCalTriggerNtuple')
chain.Add(path)
ROOT.convert(chain, tmp_out, 5.)

out_path = '%s/%s' % (out_dir, fname)
if tmp_out != out_path:
    shutil.move(tmp_out, out_path)
