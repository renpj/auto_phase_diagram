# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import parser
import shutil

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
    data = input_data.dropna(subset=['Nads','E_total']) # remove rows that are NaN for Nads and E_total
    data = data.dropna(axis=0,how='all')
    return data,ref_data,ref_detail
    
def check_data(data,ref):
    # data is a pandas.DataFrame
    require_col = set((u'Nads', u'E_slab', u'E_total','Formula_ads'))
    if not require_col.issubset(set(data.columns)):
        print('Error: Required Columns are ', ', '.join(require_col))

    for icol in ('Name','E_slab','ZPE_slab','ZPE_total','Formula_ads'): # fill NaN with i[0], else 0
        v = data[icol][0]
        if pd.isnull(data[icol][0]):
            v = 0
        data[icol] = data[icol].fillna(v)
            
    for icol in ('dZPE','E'):
        e_ads = [eval(new_formula(ref,f,icol)) for f in data['Formula_ads']]
        data[icol+'_ads'] = e_ads

    data = data.groupby(by=['Name','Nads'],as_index=False).agg(min) # 聚合相同的Nads, 取最小值。注意，没有作为新的index。

    data['G_slab'] = data['E_slab'] + data['ZPE_slab']
    data['G_total'] = data['E_total'] + data['ZPE_total']
    data['G_ads'] = (data['E_ads'] + data['dZPE_ads'])*data['Nads']

    data['dG'] = data['G_total'] - data['G_slab'] - data['G_ads']
    data['dG_avg'] = data['dG']/data['Nads'] # 平均吸附能
    data['dG_step'] = data['dG']- data['dG'].shift().fillna(0)  # 分布吸附能
    
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
        ref['T'] = None # default value for ref['T']
    else:
        t = t.iloc[0]
        if type(t) in (np.float64,int,float): # t is a number, type is from pandas
            ref['T'] = t
        else: # t is a variable
            try:
                ref['T'] = np.array(eval(t))
                if ref['T'][0] != ref['T'][1]:
                    variable['T'] = ref['T']
                else:
                    ref['T'] = ref['T'][0]
            except:
                print("Error: Please check the temperature format!")
                exit(0)
    
    ref['S'] = {}
    ref['p'] = {}
    ref['HT'] = {}
    ref['E'] = {}
    ref['dZPE'] = {}
    ref['u'] = {}
    for nf in formula:
        co_names = set([name for name in parser.expr(nf).compile().co_names])
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
                    c = str(rd.iloc[0])
                    ref[r][iname] = lambda x:np.ones(len(x))*eval(c) if hasattr(x,'__iter__') else eval(c)  # 形式一致性
                else:
                    if iname in ref_detail: # use S(T) and H(T)
                        v = ref_detail[iname]
                        if r in v.columns:
                            if np.all(pd.notnull(v[r])):
                                ref[r][iname] = lambda x: np.interp(x,v['T'],v[r]) # 形式一致性
                            else:
                                print("Error: pls check ref_"+iname)
                                break
                        else:
                            ref[r][iname] = lambda x: np.zeros(len(x)) if hasattr(x,'__iter__') else 0.0   # 形式一致性
                    else:
                        print ("Error: No "+r+" vaule for "+iname)
                        break
                
            # assign pressure
            p = row['Press']

            if p.isnull().iloc[0]:
                ref['p'][iname] = None # unit ln(bar)
            else:
                if type(p.iloc[0]) in (np.float64,int,float):
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
        name = 'G' + str(nads)
        path = '/data/graph1/' + name
        veusz_set.append("CloneWidget('/data/graph1/template','/data/graph1','"+name+"')")
        veusz_set.append("Set('"+path+"/key', 'N="+str(nads)+"')")
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
    ncolormap = str(max(nmax-nmin,2))
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
    formula = input_data['Formula_ads'] # formula is pd.Series
    ref,variable =  get_ref(ref_data,ref_detail,formula)
    data = check_data(input_data,ref)
    try:
        # pressure part
        pf = [new_formula(ref,f,'p') for f in formula] # for pressure
        u_p = np.array([8.314*ref['T']*eval(ipf)/1000/96.4853 for ipf in pf])
        # entropy part
        sf = [new_formula(ref,f,'S') for f in formula] # for entropy
        hf = [new_formula(ref,f,'HT') for f in formula] # for HT
        u_ts = []
        u_HT = []
        for i in range(len(sf)):
            isf = sf[i]
            ihf = hf[i]
            if 'T' in variable:
                u_ts.append(isf) # return a list of function(just entropy)
                u_HT.append(ihf) # return a list of function
            else:
                T = ref['T']
                u_ts.append(-T*eval(isf))
                u_HT.append(eval(ihf))
        u_ts = np.array(u_ts)
        u_HT = np.array(u_HT)
        if 'T' in variable:
            u = None # not required
        else:
            u = np.array([u_p[i]+u_ts[i]+u_HT[i] for i in range(len(u_ts))]) # cannot add them directly!
    except Exception as e1:
        print(e1)
        print("Use u directly.")
        try:
            # directly for u
            uf = [new_formula(ref,f,'u') for f in formula] # for u
            u = np.array([eval(iuf) for iuf in uf])
        except Exception as e2:
            print("Error: Pls provide enough ref data: p, T or u!")
            print(e2)
            exit(0)
    nvar = len(variable)
    if nvar == 1:
        k = list(variable)[0]
        if k in ('p','u'):
            nvar = len(variable[k])
    print("Number of variable is "+str(nvar))
    print variable
    embed,vdisplay = start_veusz()
    
    if nvar == 0:
        # 这意味着p和T都是一个值, 不做图
        data['u_ads'] = u
        data['G_ads'] += data['u_ads']*data['Nads']
        data['dG'] -= data['u_ads']*data['Nads']
        data['dG_avg'] = data['dG']/data['Nads'] # 平均吸附能
        data['dG_step'] = data['dG']- data['dG'].shift().fillna(0)  # 分布吸附能

    elif nvar == 1:
        plot_dict = {}
        vk,vv = list(variable.items())[0]       
        if vk == 'T':
            plot_dict['xlabel'] = 'Temperature (K)'
            T = np.linspace(vv[0],vv[1],quality_2d[0])
            S = np.array([eval(fs) for fs in u_ts])
            u_ts = -(T*S)
            HT = np.array([eval(fh) for fh in u_HT])
            u_HT = HT
            u_p = np.array([8.314*T*eval(ipf)/1000/96.4853 for ipf in pf]) # recalculate u_p, due to T has changed
            u = u_ts + u_p + u_HT
            plot_dict['xdata'] = T
            plot_dict['output'] = 'G_'+vk
        elif vk == 'p':
            plot_dict['xlabel'] = 'ln(p('+ vv.keys()[0] + ')/p0)'
            plot_dict['xdata'] = vv.values()[0]
            plot_dict['output'] = 'G_'+vk+'_'+vv.keys()[0]
        elif vk == 'u':
            plot_dict['xlabel'] = 'u('+ vv.keys()[0] + ') (eV)'
            plot_dict['xdata'] = vv.values()[0]
            plot_dict['output'] = 'G_'+vk+'_'+vv.keys()[0]
        else:
            print('Unsupport variable!')
            exit(0)
        ydata = {}
        for irow in range(len(data)):
            nads = int(data.iloc[irow]['Nads'])
            name = data.iloc[irow]['Name']
            dG = data.iloc[irow]['dG']
            dG -= nads*u[irow]
            ydata[name+'_'+str(nads)] = dG
        plot_dict['ydata'] = ydata
        plot_dict['embed'] = embed
        
        plot_1D(plot_dict)

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
            xdata = np.linspace(variable['T'][0],variable['T'][1],quality_2d[0])
            ydata = np.linspace(pv[0],pv[1],quality_2d[1])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['p'][pk] = ygrid.reshape(quality_2d[0]*quality_2d[1])
            T = xgrid.reshape(quality_2d[0]*quality_2d[1])
            S = np.array([eval(fs) for fs in u_ts]) # too slow
            u_ts = -(T*S)
            HT = np.array([eval(fh) for fh in u_HT]) # too slow
            u_HT = HT
            u_p = np.array([8.314*T*eval(ipf)/1000/96.4853 for ipf in pf]) # recalculate u_p, due to T has changed
            #print u_ts.shape,u_HT.shape,u_p.shape
            u = u_ts + u_HT + u_p
            output = "_".join(['T','p',pk,'2D'])
        elif ('p' in keys) and len(keys)==1:
            pk = list(variable['p'].keys())
            pv = list(variable['p'].values())
            xlabel = 'ln(p('+ pk[0] + ')/p0)'
            ylabel = 'ln(p('+ pk[1] + ')/p0)'
            xdata = np.linspace(pv[0][0],pv[0][1],quality_2d[0])
            ydata = np.linspace(pv[1][0],pv[1][1],quality_2d[1])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['p'][pk[0]] = xgrid.reshape(quality_2d[0]*quality_2d[1])
            ref['p'][pk[1]] = ygrid.reshape(quality_2d[0]*quality_2d[1])
            u_p = np.array([8.314*T*eval(ipf)/1000/96.4853 for ipf in pf]) # recalculate u_p, due to T has changed
            #print u_ts.shape,u_HT.shape,u_p.shape
            u = (u_ts + u_HT + u_p.T).T
            output = "_".join(['p',pk[0],'p',pk[1],'2D'])
        elif ('u' in keys) and len(keys)==1:
            uk = list(variable['u'].keys())
            uv = list(variable['u'].values())
            xlabel = 'u('+ uk[0] + ') (eV)'
            ylabel = 'u('+ uk[1] + ') (eV)'
            xdata = np.linspace(uv[0][0],uv[0][1],quality_2d[0])
            ydata = np.linspace(uv[1][0],uv[1][1],quality_2d[1])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['u'][uk[0]] = xgrid.reshape(quality_2d[0]*quality_2d[1])
            ref['u'][uk[1]] = ygrid.reshape(quality_2d[0]*quality_2d[1])
            u = np.array([eval(iuf) for iuf in uf])
            output = "_".join(['u',uk[0],'u',uk[1],'2D'])
        else:
            print("Unsupport 2D plot for: "+str(keys))
            exit(0)

        # Get 2D data
        zgrid = []
        zgrid.append(np.zeros(xgrid.shape)) # all grid should compare to 0!
        for idx in range(len(data)): # the index of zgrid
            dG = data['dG'].iloc[idx]
            nads = data['Nads'].iloc[idx]
            dG -= nads*u[idx]
            zgrid.append(dG.reshape(quality_2d))
        zgrid = np.array(zgrid)
        ngrid = zgrid.argmin(0)
        nmax,nmin = len(data)+1,0
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
        }
        plot_2D(plot_dict)

    else:
        print("Number of variables must less than 2!")
        exit(0)

    print('Save data to excel file G_result.xslx')
    data = data[['Name','Nads','Formula_ads','E_slab','ZPE_slab','E_ads','dZPE_ads','E_total','ZPE_total','dG','dG_avg','dG_step']]
    data.to_excel('G_result.xlsx',float_format='%.4f',index=False)
#    close_veusz(embed,vdisplay)