#!/usr/bin/env python
#-*- coding:utf-8 -*-
from auto_phase_diagram import phase_diagram
import shutil,os

if __name__ == '__main__':
    # Constant
    quality_2d = (500,500) # the quality for 2D contour map
    import sys
    args = sys.argv
    lprobability = False
    p_threshold = 0.05
    filename = None
    usage = """
        usage: plot_phase_diagram.py xls_file [--probability threshold] [--example] [--template] 
        xls_file: input excel file. 
        --probability threshold: consider the mix phase distribution. If p > threshold, means the phase is possible. 
        --example: copy example.xlsx into work dir.
        --template: copy template.vsz into work dir.
    """
    
    if ((len(args)==1) or (len(args)>4)):
        print(len(args))
        print(usage)
        sys.exit()
    else:
        for idx,arg in enumerate(args):
            if arg == '--probability':
                try:
                    p_threshold = float(args[idx+1])
                    if p_threshold <= 0.0:
                        print("Use default threshold 0.05.")
                        p_threshold = 0.05
                except:
                    print("Use default threshold 0.05.")
                    pass
                lprobability = True
            elif arg == '--example':
                mod_dir = os.path.dirname(os.path.abspath(sys.modules.get(phase_diagram.__module__).__file__))
                shutil.copy2(os.path.join(mod_dir,'input.xlsx'),'example.xlsx')
                sys.exit()
            elif arg == '--template':
                mod_dir = os.path.dirname(os.path.abspath(sys.modules.get(phase_diagram.__module__).__file__))
                shutil.copy2(os.path.join(mod_dir,'template.vsz'),'template.vsz')
                sys.exit()
            elif '.xls' in arg:
                filename = arg
            else:
                if idx != 0:
                    print(usage)
                    sys.exit()
    if not filename:
        print(usage)
        sys.exit(".xls file should be provided!")
    phase_diagram(filename,quality_2d=quality_2d,lprobability=lprobability,p_threshold=p_threshold)