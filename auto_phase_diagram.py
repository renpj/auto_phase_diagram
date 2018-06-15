# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import parser
import shutil

# blue_darkred
# copy from veusz/utils/colormap
color_map = ( # order: b,g,r
    (216, 0, 36, 255),
    (247, 28, 24, 255),
    (255, 87, 40, 255),
    (255, 135, 61, 255),
    (255, 176, 86, 255),
    (255, 211, 117, 255),
    (255, 234, 153, 255),
    (255, 249, 188, 255),
    (255, 255, 234, 255),
    (234, 255, 255, 255),
    (188, 241, 255, 255),
    (153, 214, 255, 255),
    (117, 172, 255, 255),
    (86, 120, 255, 255),
    (61, 61, 255, 255),
    (53, 39, 247, 255),
    (47, 21, 216, 255),
    (33, 0, 165, 255)
)

def get_color(n,start=0,stop=1):
    '''
    Return a dict of color in hex with length of n.
    The color list is interpolate of colormap.
    '''
    cmap = np.array(color_map)
    x0 = np.linspace(start,stop,len(color_map))
    x = np.linspace(start,stop,n).astype(np.intc)
    b = np.interp(x,x0,cmap[:,0]).astype(np.intc)
    g = np.interp(x,x0,cmap[:,1]).astype(np.intc)
    r = np.interp(x,x0,cmap[:,2]).astype(np.intc)
    clist = ['#%02x%02x%02x' % tuple(rgb) for rgb in zip(r,g,b)]
    cdict = {}
    for i,c in enumerate(clist):
        cdict[x[i]] = c
    return cdict

def start_veusz():
    # for ploting by veusz 
    print("Preparing plotting environment ...")
    global isveusz
    try:
        import veusz.embed
        isveusz = True
        print("The data will be plotted by Veusz!")
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb()
        vdisplay.start()
        embed = veusz.embed.Embedded(hidden=True)
        return embed,vdisplay
    except:
        isveusz = False
        print("Can't import Veusz, the data will not be plotted!")
        return None,None

def close_veusz(embed,vdisplay):
    if isveusz:
        embed.Close()
        # vdisplay.stop()

def data_from_xls(filename):    
    print('Reading input excel file ...')
    allsheet = pd.read_excel(filename,sheet_name=None)
    input_data = allsheet['data']
    ref_data = allsheet['ref']
    ref_detail = {}
    for s in allsheet:
        if s.startswith('ref_'):
            name = s[4:]
            ref_detail[name] = allsheet[s]
    print('Initialize data ...')
    data = input_data.dropna(subset=['Nads','E_Total']) # remove rows that are NaN for Nads and E_Total
    data = data.dropna(axis=0,how='all')
    return data,ref_data,ref_detail
    
def check_data(data,ref):
    # data is a pandas.DataFrame
    require_col = set((u'Nads', u'E_Slab', u'E_Total','Formula'))
    if not require_col.issubset(set(data.columns)):
        print('Error: Required Columns are ', ', '.join(require_col))

    for icol in ('Name','E_Slab','ZPE_Slab','ZPE_Total','Formula'): # fill NaN with i[0], else 0
        v = data[icol][0]
        if pd.isnull(data[icol][0]):
            v = 0
        data[icol] = data[icol].fillna(v)
            
    data = data.groupby(by=['Name','Nads'],as_index=False).agg(min) # 聚合相同的Nads, 取最小值。注意，没有作为新的index。

    data['G_Slab'] = data['E_Slab'] + data['ZPE_Slab']
    data['G_Total'] = data['E_Total'] + data['ZPE_Total']
    
    return data

def check_formula(s):
    co_names = parser.expr(s).compile().co_names
    for i in co_names:
        exec(i+'=1.0') # assign name with value
    try:
        eval(s)
    except Exception as e:
        msg = " ".join([e,s])
        print(msg)
        exit(0)
    
def parse_formula(s):
    '''
    Parse the string formula to a list.
    '''
    check_formula(s)
    result = []
    var = ''
    for i in s:
        if i in '+-*/()':
            if var != '':
                result.append(var)
            result.append(i)
            var = ''
        elif i in ' ': # 去除所有空格
            pass
        else:
            var += i
    if var != '':
        result.append(var)
    return result

def rebuild_formula(s,mapping):
    '''
    Replace the vars according to map and rebuild the list to new formula string.
    mapping = {'old_var':'new_var'}
    '''
    l = parse_formula(s)
    nl = [mapping.pop(il,il) for il in l] # dict.pop is great!
    ns = ''.join(nl)
    for k in mapping: # for the item not in formula, let them multiply 0
        ns += '+0*' + mapping[k]
    return ns
 
def new_formula(ref,formula,name):
    mapping = {}
    for k in ref[name].keys():
        mapping[k] = 'ref["'+name+'"]["'+k+'"]'
        if name in ('S','HT'):
            mapping[k] += '(T)'
    return rebuild_formula(formula,mapping)

def get_ref(ref_data,ref_detail,formula):
    '''
    ref_data is pandas.DataFrame, formula is pandas.Series.
    '''
    variable = {} # if T or p are variables, store them
    ref = {}
    # get T
    t = ref_data['Temperature']
    t = t[pd.notnull(t)]
    if len(t) > 1:
        print("Error: Pls Check Temperature Input!")
        exit(0)
    elif len(t)==0:
        ref['T'] = 0 # default value for ref['T']
    else:
        t = t.iloc[0]
        if type(t) in (np.float64,np.int64,int,float): # t is a number, type is from pandas
            ref['T'] = t
        else: # t is a variable
            try:
                ref['T'] = np.array(eval(t))
                if ref['T'][0] != ref['T'][1]:
                    variable['T'] = ref['T']
                else:
                    ref['T'] = ref['T'][0]
            except Exception as e:
                print("Error: Please check the temperature format!")
                print(e)
                exit(0)
    
    ref['S'] = {}
    ref['p'] = {}
    ref['HT'] = {}
    ref['E'] = {}
    ref['dZPE'] = {}
    ref['u'] = {}
    for nf in formula:
        co_names = set([name for name in parser.expr(nf).compile().co_names])-set(('Total','Slab','Nads'))
        for iname in co_names:
            # assign HT,E,dZPE,u,S
            row = ref_data[ref_data.Name == iname]
            if row.shape[0] != 1:
                print ("Error: Duplicated or NO row for "+iname)
                break
            
            for r in ('E','dZPE',):
                rd = row[r]
                if rd.isnull().iloc[0]:
                    print ("Error: NO E or dZPE for "+iname)
                    break
                else:
                    ref[r][iname] = rd.iloc[0]
            
            for r in ('S','HT',):
                rd = row[r]
                if rd.notnull().iloc[0]:
                    def func(x,c=rd.iloc[0]):
                        return np.ones(len(x))*c if hasattr(x,'__iter__') else c
                else:
                    if iname in ref_detail: # use S(T) and H(T)
                        v = ref_detail[iname]
                        if r in v.columns:
                            if np.all(pd.notnull(v[r])):
                                def func(x,vt=v['T'],vr=v[r]):
                                    return np.interp(x,vt,vr)
                            else:
                                print("Error: pls check ref_"+iname)
                                break
                        else:
                            def func(x):
                                return np.zeros(len(x)) if hasattr(x,'__iter__') else 0.0
                    else:
                        print ("Error: No "+r+" vaule for "+iname)
                        break
                ref[r][iname] = func
            # assign pressure
            p = row['Press']

            if p.isnull().iloc[0]:
                ref['p'][iname] = None # unit ln(bar)
            else:
                if type(p.iloc[0]) in (np.float64,np.int64,int,float):
                    ref['p'][iname] = np.log(p.iloc[0])
                else:
                    try:
                        ref['p'][iname] = np.array(np.log(eval(p.iloc[0]))) # ln(p)
                        if ref['p'][iname][0] != ref['p'][iname][1]:
                            if 'p' not in variable:
                                variable['p'] = {} 
                            variable['p'][iname] = ref['p'][iname]
                        else:
                            ref['p'][iname] = ref['p'][iname][0]
                    except Exception as e:
                        print("Error: Please check the Press format!")
                        print(e)
                        break
            # get u
            u = row['u']
            if u.isnull().iloc[0]:
                ref['u'][iname] = None # default 0
            else:
                if  type(u.iloc[0]) in (np.float64,int,float):
                    ref['u'][iname] = u.iloc[0]
                else:
                    try:
                        ref['u'][iname] = np.array(eval(u.iloc[0])) # ln(p)
                        if ref['u'][iname][0] != ref['u'][iname][1]:
                            if 'u' not in variable:
                                variable['u'] = {}                         
                            variable['u'][iname] = ref['u'][iname]
                        else:
                            ref['u'][iname] = ref['u'][iname][0]
                    except Exception as e:
                        print("Error: Please check the u format!",e)
                        print(u)
                        break
    return ref,variable

def plot_1D(plot_dict):
    """
    plot_dict keys:
        embed: veusz.embed.Embedded or None
        xlabel: str
        xdata: numpy.array
        ydata: dict, {Nads:value}
        output: str
    """
    xdata = plot_dict['xdata']
    xlabel = plot_dict['xlabel']
    veusz_set = []
    veusz_set.append("SetData('x',"+str(xdata.tolist())+")")
    ydata = plot_dict['ydata']
    ymin = []
    ymax = []
    for nads in ydata:
        dG = ydata[nads].tolist()
        name = 'G' + nads
        path = '/data/graph1/' + name
        veusz_set.append("CloneWidget('/data/graph1/template','/data/graph1','"+name+"')")
        veusz_set.append("Set('"+path+"/key', '"+nads+"')")
        veusz_set.append("Set('"+path+"/xData','x')")
        veusz_set.append("SetData('" + name + "', " +str(dG)+")")
        veusz_set.append("Set('"+path+"/yData','"+name+"')")
        ymin.append(min(dG))
        ymax.append(max(dG))
    veusz_set.append("Set('/data/graph1/x/min',"+str(float(min(xdata)))+")")
    veusz_set.append("Set('/data/graph1/x/max',"+str(float(max(xdata)))+")")
    veusz_set.append("Set('/data/graph1/x/label','"+xlabel+"')")
    ymin = min(ymin)
    ymax = max(ymax)
    veusz_set.append("Set('/data/graph1/y/min',"+str(float(ymin-(ymax-ymin)*0.2))+")")
    veusz_set.append("Set('/data/graph1/y/max',"+str(float(ymax+(ymax-ymin)*0.2))+")")
    veusz_set.append("Remove('/data/graph1/template')")
    veusz_set.append("Remove('/contour')")
    # save to vsz
    output_filename = plot_dict['output']
    shutil.copy2('template.vsz',output_filename+'.vsz')
    veusz_file = open(output_filename+'.vsz','a')
    for  i in veusz_set:
        veusz_file.write(i+'\n')
    veusz_file.close()
    # save data to .dat file
    print('Save data to '+output_filename+'.csv')
    ydata[xlabel] = xdata 
    data_df = pd.DataFrame(ydata)
    data_df.set_index(xlabel,inplace=True)
    data_df.to_csv(output_filename+'.csv',index=True,float_format='%5.3f')
    embed = plot_dict['embed']
    if embed is not None:
        embed.Load(output_filename+'.vsz')
        print('Export to '+output_filename+'.jpg')
        embed.Export(output_filename+'.jpg',dpi=300)

def plot_2D(plot_dict):
    # 生成等值面图
    ngrid = plot_dict['ngrid']
    xdata = plot_dict['xdata']
    ydata = plot_dict['ydata']
    nmin = plot_dict['nmin']
    nmax = plot_dict['nmax']
    xlabel = plot_dict['xlabel']
    ylabel = plot_dict['ylabel']
    label = plot_dict['label']
    output_filename = plot_dict['output']
    embed = plot_dict['embed']
    print('Generate 2D contour '+output_filename)
    veusz_set = []
    veusz_set.append("SetData2D('grid',"
        +str(ngrid.tolist())
        +",xcent="
        +str(xdata.tolist())
        +",ycent="
        +str(ydata.tolist())
        +")")
    veusz_set.append("Set('/contour/graph1/image1/min',"+str(nmin)+")")
    veusz_set.append("Set('/contour/graph1/image1/max',"+str(nmax)+")")
    ncolormap = str(nmax-nmin)
    veusz_set.append("Set('/contour/graph1/image1/colorMap', u'blue-darkred-step"+ncolormap+"')")
    veusz_set.append("Set('/contour/graph1/colorbar1/MajorTicks/number', "+ncolormap+")")
    level = np.unique(ngrid).tolist()
    veusz_set.append("Set('/contour/graph1/contour1/manualLevels', "+str(level)+")")
    xmin = min(xdata)
    xmax = max(xdata)
    veusz_set.append("Set('/contour/graph1/x/label','"+ xlabel+"')")
    veusz_set.append("Set('/contour/graph1/x/min',"+str(float(xmin))+")")
    veusz_set.append("Set('/contour/graph1/x/max',"+str(float(xmax))+")")
    ymin = min(ydata)
    ymax = max(ydata)
    veusz_set.append("Set('/contour/graph1/y/label','"+ ylabel+"')")
    veusz_set.append("Set('/contour/graph1/y/min',"+str(float(ymin))+")")
    veusz_set.append("Set('/contour/graph1/y/max',"+str(float(ymax))+")")
    # add label and rect
    cmap = get_color(ncolormap,nmin,nmax-1)
    label_file=open(output_filename+'_labelname.dat','w')
    print("Label name:")
    for ilabel in label:
        if ilabel != 0:
            label_name = 'label'+str(ilabel+1)
            rect_name = 'rect' + str(ilabel+1)
            veusz_set.append("CloneWidget('/contour/graph1/label1','/contour/graph1','"+label_name+"')")
            veusz_set.append("CloneWidget('/contour/graph1/rect1','/contour/graph1','"+rect_name+"')")
        else:
            label_name = 'label1'
            rect_name = 'rect1'
        # set label prop
        veusz_set.append("Set('/contour/graph1/"+label_name+"/label','"+str(ilabel)+"')")
        yPos = 0.96-ilabel*0.07
        veusz_set.append("Set('/contour/graph1/"+label_name+"/yPos',["+str(yPos)+"])")
        # set rect prop
        yPos = 0.97-ilabel*0.07
        veusz_set.append("Set('/contour/graph1/"+rect_name+"/yPos',["+str(yPos)+"])")
        veusz_set.append("Set('/contour/graph1/"+rect_name+"/Fill/color','"+cmap[ilabel]+"')")
        print(str(ilabel)+': '+label[ilabel])
        label_file.write("%4i\t%s\n" %(ilabel,label[ilabel]))
    label_file.close()
    print("Label names were saved in "+output_filename+'_labelname.dat')
    veusz_set.append("Remove('/data')")
    shutil.copy2('template.vsz',output_filename+'.vsz')
    veusz_file = open(output_filename+'.vsz','a')
    for i in veusz_set:
        veusz_file.write(i+'\n')
    veusz_file.close()

    # save data to .dat file
    dim = quality_2d[0]*quality_2d[1]
    xyz = np.asarray([xgrid.reshape(dim),ygrid.reshape(dim),ngrid.reshape(dim)]).T
    np.savetxt(output_filename+'.dat',xyz,fmt='%.5f',header=" ".join((xlabel,ylabel,"N")))
    if embed is not None:
        embed.Load(output_filename+'.vsz')
        print('Export to '+output_filename+'.jpg')
        embed.Export(output_filename+'.jpg',dpi=300)
        
if __name__ == '__main__':
    # Constant
    quality_2d = (500,500) # the quality for 2D contour map
    
    import sys
    args = sys.argv
    if not (len(args) == 2):
        print("usage: auto_phase_diagram.py xls_file")
        exit(0)
    filename = args[1]
    input_data,ref_data,ref_detail = data_from_xls(filename)
    formula = input_data['Formula'] # formula is pd.Series
    ref,variable =  get_ref(ref_data,ref_detail,formula)
    data = check_data(input_data,ref)
    nvar = len(variable)
    if nvar == 1:
        k = list(variable)[0]
        if k in ('p','u'):
            nvar = len(variable[k])
    print("Number of variable is "+str(nvar))
    if nvar > 0:
        print(variable)
        embed,vdisplay = start_veusz()
    
    # get u for all ref
    for name in ref['u']:
        T = ref['T']
        try:
            # get u directly
            print("Try to use u for "+name+" directly...")
            ref['u'][name] = ref['u'][name]+ref['E'][name]+ref['dZPE'][name]+ref['HT'][name](T)
            print("Done!")
        except Exception as e1:
            print(e1)
            # get u from T and p
            print("Try to get u from T and p ...")
            try:
                ref['u'][name] = ref['E'][name]
                ref['u'][name] += ref['dZPE'][name]
                ref['u'][name] += ref['HT'][name](T)
                ref['u'][name] += 8.314*T*ref['p'][name]/1000/96.4853 
                ref['u'][name] -= T*ref['S'][name](T) 
                print("Done!")
            except Exception as e2:
                print(e2)
                print("Error: Pls provide enough ref data: p, T or u!")
                print(ref)
                exit(0)

    if nvar == 1:
        vk,vv = list(variable.items())[0]       
        if vk == 'T':
            T = np.linspace(vv[0],vv[1],quality_2d[0])
            xdata = T
            output = 'G_'+vk
            xlabel = 'Temperature (K)'
            # recalculate u for new T
            for name in ref['u']:
                ref['u'][name] = ref['E'][name]
                ref['u'][name] += ref['dZPE'][name]
                ref['u'][name] += ref['HT'][name](T)
                ref['u'][name] += 8.314*T*ref['p'][name]/1000/96.4853 
                ref['u'][name] -= T*ref['S'][name](T) 
        elif vk == 'p':
            xlabel = 'ln(p('+ list(vv.keys())[0] + ')/p0)'
            xdata = vv.values()[0]
            output = 'G_'+vk+'_'+list(vv.keys())[0]
            # no required for recalculate u
        elif vk == 'u':
            xlabel = 'u('+ list(vv.keys())[0] + ') (eV)'
            xdata = list(vv.values())[0]
            output = 'G_'+vk+'_'+list(vv.keys())[0]
            # no required for recalculate u
        else:
            print('Unsupport variable!')
            exit(0)

    elif nvar == 2:      
        """
        Three cases: (T,p), (p1,p2), (u1,u2)
        """     
        keys = variable.keys()
        if ('T' in keys) and ('p' in keys):
            xlabel = 'T(K)'
            pk = list(variable['p'].keys())[0]
            pv = list(variable['p'].values())[0]
            ylabel = 'ln(p('+ pk+ ')/p0)'
            output = "_".join(['T','p',pk,'2D'])
            xdata = np.linspace(variable['T'][0],variable['T'][1],quality_2d[0])
            ydata = np.linspace(pv[0],pv[1],quality_2d[1])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['p'][pk] = ygrid.reshape(quality_2d[0]*quality_2d[1])
            T = xgrid.reshape(quality_2d[0]*quality_2d[1])
            # recalculate u for new T
            for name in ref['u']:
                ref['u'][name] = ref['E'][name]
                ref['u'][name] += ref['dZPE'][name]
                ref['u'][name] += ref['HT'][name](T)
                ref['u'][name] += 8.314*T*ref['p'][name]/1000/96.4853 
                ref['u'][name] -= T*ref['S'][name](T) 
        elif ('p' in keys) and len(keys)==1:
            pk = list(variable['p'].keys())
            pv = list(variable['p'].values())
            xlabel = 'ln(p('+ pk[0] + ')/p0)'
            ylabel = 'ln(p('+ pk[1] + ')/p0)'
            output = "_".join(['p',pk[0],'p',pk[1],'2D'])
            xdata = np.linspace(pv[0][0],pv[0][1],quality_2d[0])
            ydata = np.linspace(pv[1][0],pv[1][1],quality_2d[1])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['p'][pk[0]] = xgrid.reshape(quality_2d[0]*quality_2d[1])
            ref['p'][pk[1]] = ygrid.reshape(quality_2d[0]*quality_2d[1])
            # recalculate u for new p
            T = ref['T']
            for name in pk: # not for all u
                ref['u'][name] = ref['E'][name]
                ref['u'][name] += ref['dZPE'][name]
                ref['u'][name] += ref['HT'][name](T)
                ref['u'][name] += 8.314*T*ref['p'][name]/1000/96.4853 
                ref['u'][name] -= T*ref['S'][name](T) 
        elif ('u' in keys) and len(keys)==1:
            uk = list(variable['u'].keys())
            uv = list(variable['u'].values())
            xlabel = 'u('+ uk[0] + ') (eV)'
            ylabel = 'u('+ uk[1] + ') (eV)'
            output = "_".join(['u',uk[0],'u',uk[1],'2D'])
            xdata = np.linspace(uv[0][0],uv[0][1],quality_2d[0])
            ydata = np.linspace(uv[1][0],uv[1][1],quality_2d[1])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['u'][uk[0]] = xgrid.reshape(quality_2d[0]*quality_2d[1])
            ref['u'][uk[1]] = ygrid.reshape(quality_2d[0]*quality_2d[1])
            # recalculate u for new u
            T = ref['T']
            for name in uk: # not for all u
                ref['u'][name] += ref['E'][name]
                ref['u'][name] += ref['dZPE'][name]
                ref['u'][name] += ref['HT'][name](T) 
        else:
            print("Unsupport 2D plot for: "+str(keys))
            exit(0)
        # Get 2D data
        zgrid = []
        zgrid.append(np.zeros(xgrid.shape)) # all grid should compare to 0!
    elif nvar > 2:
        print("Number of variables must less than 2!")
        exit(0)

    # eval dG
    dG = []
    for irow in range(len(data)):
        idata = data.iloc[irow]
        Nads = idata['Nads']
        iformula = idata['Formula']
        Total = idata['G_Total']
        Slab = idata['G_Slab']
        dG.append(eval(new_formula(ref,iformula,'u')))

    # output
    if nvar == 0:
        # 这意味着p和T (or u) 是一个值, 不做图
        data['dG'] = dG
        data['dG_avg'] = data['dG']/data['Nads'] # 平均吸附能
        print('Save data to excel file G_result.xslx')
        data = data[['Name','Nads','Formula','E_Slab','ZPE_Slab','E_Total','ZPE_Total','dG','dG_avg',]]
        data.to_excel('G_result.xlsx',float_format='%.4f',index=False)
    elif nvar == 1:
        ydata = {}
        for irow in range(len(data)):
            idata = data.iloc[irow]
            Nads = idata['Nads']
            name = idata['Name']
            ydata[name+'(N='+str(Nads)+')'] = dG[irow]
        plot_dict = {
            'xdata':xdata,
            'ydata': ydata,
            'xlabel':xlabel,
            'embed': embed,
            'output':output,
        }
        plot_1D(plot_dict)
    elif nvar == 2:
        # get Gmin
        dG = np.array([[0]*(quality_2d[0]*quality_2d[1])]+dG)
        Gmin = dG.min(0) # column min
        ddG = dG - Gmin
        # calculate partition function
        if type(T) not in (np.ndarray,):
            if T==0:
                T = 298.15
                print("Use T=298.15 instead of 0!")
        q = np.exp(-ddG*1000*96.4853/8.314/T) # note: T can be a array or number
        # calculate probability
        P = q/q.sum(0)
        # get logical array
        LP = P >= 0.05 # the probability bigger than 0.02 can exists
        LPset = list(set(map(tuple,LP.T))) # note: column mode
        # get index array
        Narray = np.ones(quality_2d[0]*quality_2d[1])*-1 # default value is -1
        for idx,iLP in enumerate(LPset):
            Narray[np.all(LP.T==iLP,1)] = idx
        # make it grid like 
        ngrid = Narray.reshape(quality_2d)
        # get the labels
        label = {}
        label_id = np.unique(Narray).astype(np.intc)
        namelist = np.array([0]+list(data['Name']))
        nadslist = np.array([0]+list(data['Nads']))
        for idx in label_id:
            if idx != -1:
                iLP = LPset[idx]
                name = namelist[np.array(iLP)]
                nads = nadslist[np.array(iLP)]
                label_name = []
                if hasattr(name ,'__iter__'):
                    for iname,inads in zip(name,nads):
                        label_name.append(iname+'(N='+str(inads)+')')
                    label_name = str(tuple(label_name))
                else:
                    label_name = iname+'(N='+str(inads)+')'
                label[idx] = label_name   
            else:
                label[idx] = None
        nmax,nmin = len(LPset),-1
        plot_dict = {
            'ngrid':ngrid,
            'xdata':xdata,
            'ydata':ydata,
            'nmin':nmin,
            'nmax':nmax,
            'xlabel':xlabel,
            'ylabel':ylabel,
            'output':output,
            'embed':embed,
            'label':label,
        }
        plot_2D(plot_dict)

#    close_veusz(embed,vdisplay)