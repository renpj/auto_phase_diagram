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
    allsheet = pd.read_excel(filename,sheetname=None)
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

def rebuild_formula(s,map):
    '''
    Replace the vars according to map and rebuild the list to new formula string.
    map = {'old_var':'new_var'}
    '''
    l = parse_formula(s)
    for k,v in map.items():
        nl = [v if il==k else il for il in l]
    ns = ''.join(nl)
    return ns

def get_ref(ref_data,ref_detail,formula):
    '''
    ref_data is pandas.DataFrame, formula is string.
    '''
    variable = {} # if T or p are variables, store them
    ref = {}
    co_names = set([name for nf in formula for name in parser.expr(nf).compile().co_names])
    # get T
    t = ref_data['Temperature']
    t = t[pd.notnull(t)]
    if len(t) > 1:
        print("Error: Pls Check Temperature Input!")
        exit(0)
    elif len(t)==0:
        ref['T'] = 298.15 # default value for ref['T']
    else:
        t = t.iloc[0]
        if type(t) == type(np.float64(0)): # t is a number, type is from pandas
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
    for iname in co_names:
        # assign HT,E,dZPE,u,S
        row = ref_data[ref_data.Ref == iname]
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
                ref[r][iname] = lambda x:eval(c) # 形式一致性
            else:
                if iname in ref_detail: # use S(T) and H(T)
                    v = ref_detail[iname]
                    if r in v.columns:
                        if np.all(pd.notnull(v[r])):
                            ref[r][iname] = lambda x: np.interp(x,v['T'],v[r])
                        else:
                            print("Error: pls check ref_"+iname)
                            break
                    else:
                        ref[r][iname] = lambda x: 0.0
                else:
                    print ("Error: No "+r+" vaule for "+iname)
                    break
            
        # assign pressure
        p = row['Press']

        if p.isnull().iloc[0]:
            ref['p'][iname] = 0 # unit ln(bar)
        else:
            if type(p.iloc[0]) == type(np.float64(0)):
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
                    print("Error: Please check the Press format!",e)
                    break
        # get u
        u = row['u']
        if u.isnull().iloc[0]:
            ref['u'][iname] = None 
        else:
            if type(u.iloc[0]) == type(np.float64(0)):
                ref['u'][iname] = u.iloc[0]
            else:
                try:
                    ref['u'][iname] = np.array(np.log(eval(u.iloc[0]))) # ln(p)
                    if ref['u'][iname][0] != ref['u'][iname][1]:
                        if 'u' not in variable:
                            variable['u'] = {}                         
                        variable['u'][iname] = ref['u'][iname]
                    else:
                        ref['p']['u'][iname] = ref['u'][iname][0]
                except Exception as e:
                    print("Error: Please check the u format!",e)
                    break
    return ref,variable
 
def new_formula(ref,formula,name):
    map = {}
    for k in ref[name].keys():
        map[k] = 'ref["'+name+'"]["'+k+'"]'
    return rebuild_formula(formula,map)

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
    ref,variable = {},{}
    for nf in formula:
        iref,iv = get_ref(ref_data,ref_detail,nf)
        ref.update(iref)
        variable.update(iv)
    data = check_data(input_data,ref)
    pf = [new_formula(ref,f,'p') for f in formula] # for pressure
    sf = [new_formula(ref,f,'S') for f in formula] # for entropy
    hf = [new_formula(ref,f,'HT') for f in formula] # for HT
    u_p = np.array([8.314*ref['T']*eval(ipf)/1000/96.4853 for ipf in pf])
    u_ts = []
    for isf in sf:
        if 'T' in variable:
            u_ts.append(eval(isf)) # return a list of function(just entropy)
        else:
            u_ts.append(-ref['T']*eval(isf)(ref['T']))
    u_ts = np.array(u_ts)
    u_HT = []
    for ihf in hf:
        if 'T' in variable:
            u_HT.append(eval(ihf)) # return a list of function
        else:
            u_HT.append(eval(ihf)(ref['T']))
    u_HT = np.array(u_HT)
    nvar = len(variable)
    print("Number of variable is "+str(nvar))
    embed,vdisplay = start_veusz()
    
    if nvar == 0:
        # 这意味着p和T都是一个值, 不做图
        data['u_ads'] = (u_p + u_ts + u_HT)
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
            S = np.array([[fs(it) for it in T] for fs in u_ts])
            u_ts = -(T*S)
            HT = np.array([[fh(it) for it in T] for fh in u_HT])
            u_HT = HT
            u_p = np.array([8.314*T*eval(ipf)/1000/96.4853 for ipf in pf]) # recalculate u_p, due to T has changed
            plot_dict['xdata'] = T
            plot_dict['output'] = 'G_'+vk
        else:
            plot_dict['xlabel'] = 'ln(p('+ vv.keys()[0] + ')/p0)'
            plot_dict['xdata'] = vv.values()[0]
            plot_dict['output'] = 'G_'+vk+'_'+vv.keys()[0]
        ydata = {}
        for irow in range(len(data)):
            nads = int(data.iloc[irow]['Nads'])
            dG = data.iloc[irow]['dG']
            dG -= nads*(u_ts[irow]+u_p[irow]+u_HT[irow])
            ydata[nads] = dG
        plot_dict['ydata'] = ydata
        plot_dict['embed'] = embed
        
        plot_1D(plot_dict)

    elif nvar == 2:      
        """
        Three cases: (T,p), (p1,p2), (u1,u2)
        """     
        keys = variable.keys()
        plot_dict = {}
        if ('T' in keys) and ('p' in keys):
            plot_dict['xlabel'] = 'T(K)'
            pk = variable['p'].keys()[0]
            pv = variable['p'].values()[0]
            xdata = np.linspace(variable['T'][0],variable['T'][1],quality_2d[0])
            ydata = np.linspace(pv[0],pv[1],quality_2d[1])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['T'] = xgrid
            ref['p'][pk] = ygrid
        elif ('p' in keys) and len(set(keys))==1:
            pk = variable['p'].keys()
            pv = variable['p'].values()
            plot_dict['xlabel'] = 'ln(p('+ pk[0] + ')/p0)'
            plot_dict['ylabel'] = 'ln(p('+ pk[1] + ')/p0)'
            xdata = np.linspace(pv[0][0],pv[0][1],quality_2d[0])
            ydata = np.linspace(pv[1][0],pv[1][1],quality_2d[1])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['p'][pk[0]] = xgrid
            ref['p'][pk[1]] = ygrid
        elif ('u' in keys) and len(set(keys))==1:
            uk = variable['u'].keys()
            uv = variable['u'].values()
            plot_dict['xlabel'] = 'u('+ uk[0] + ') (eV)'
            plot_dict['ylabel'] = 'u('+ uk[1] + ') (eV)'
            xdata = np.linspace(uv[0][0],uv[0][1],quality_2d[0])
            ydata = np.linspace(uv[1][0],uv[1][1],quality_2d[1])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['u'][uk[0]] = xgrid
            ref['u'][uk[1]] = ygrid
        # Get 2D contour data
        k_notT = [i for i in variable if i!= 'T'] # get var that isnot T
        ylabel = 'ln(p('+ k_notT[0] + ')/p0)'
        ydata = np.linspace(variable[k_notT[0]][0],variable[k_notT[0]][1],quality_2d[1])
        if len(k_notT) == 1:
            xlabel = 'T(K)'
            xdata = np.linspace(variable['T'][0],variable['T'][1],quality_2d[0])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['T'] = xgrid
            ref['p'][k_notT[0]] = ygrid
        else:
            xlabel = 'ln(p('+ k_notT[1] + ')/p0)'
            xdata = np.linspace(variable[k_notT[1]][0],variable[k_notT[1]][1],quality_2d[0])
            xgrid,ygrid = np.meshgrid(xdata,ydata)
            ref['p'][k_notT[1]] = xgrid
            ref['p'][k_notT[0]] = ygrid
        u_p = 8.314*ref['T']*eval(pf)/1000/96.4853
        u_ts = -ref['T']*eval(sf)
        u = u_p + u_ts
        zgrid = []
        for nads in range(int(data['Nads'].max())+1): # nads as the index of zgrid,if nads not exists, dG == 0
            if nads in map(int,data['Nads']):
                dG = data['dG'][data['Nads']==nads].iloc[0]
                dG -= nads*u
                zgrid.append(dG)
            else:
                zgrid.append(np.zeros(xgrid.shape))
        zgrid = np.array(zgrid)
        ngrid = zgrid.argmin(0)
        nmax,nmin = ngrid.max(),ngrid.min()
        # 生成等值面图
        print('Generate 2D contour '+'_'.join(variable.keys())+'_2D.vsz')
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
        output_filename = '_'.join(variable.keys())+'_2D'
        shutil.copy2('template.vsz',output_filename+'.vsz')
        veusz_file = open(output_filename+'.vsz','a')
        for i in veusz_set:
            veusz_file.write(i+'\n')
        veusz_file.close()
        # save data to .dat file
        dim = quality_2d[0]*quality_2d[1]
        xyz = np.asarray([xgrid.reshape(dim),ygrid.reshape(dim),ngrid.reshape(dim)]).T
        np.savetxt(output_filename+'.dat',xyz,fmt='%.5f',header=" ".join((xlabel,ylabel,"N")))
        if isveusz:
            embed.Load(output_filename+'.vsz')
            print('Export to '+output_filename+'.jpg')
            embed.Export(output_filename+'.jpg',dpi=300)
            
        # Plot G_u
        xdata = np.array([u.min(),u.max()])
        plot_data = {}
        for irow in data.index:
            nads = int(data.iloc[irow]['Nads'])
            dG = data.iloc[irow]['dG']
            dG -= nads*xdata
            plot_data[nads] = dG
        print('Generate G vs u plot G_u.vsz')
        xlabel = 'u(eV)'
        ymin = []
        ymax = []
        veusz_set = []
        veusz_set.append("SetData('x',"+str(xdata.tolist())+")")
        for nads in plot_data:
            dG = plot_data[nads].tolist()
            name = 'G' + str(nads)
            path = '/data/graph1/' + name
            veusz_set.append("CloneWidget('/data/graph1/template','/data/graph1','"+name+"')")
            veusz_set.append("Set('"+path+"/key', 'N="+str(nads)+"')")
            veusz_set.append("Set('"+path+"/xData','x')")
            veusz_set.append("SetData('" + name + "', " +str(dG)+")")
            veusz_set.append("Set('"+path+"/yData','"+name+"')")
            ymin.append(min(dG))
            ymax.append(max(dG))
        xmin,xmax = xdata
        veusz_set.append("Set('/data/graph1/x/min',"+str(float(min(xdata)))+")")
        veusz_set.append("Set('/data/graph1/x/max',"+str(float(max(xdata)))+")")
        ymin = min(ymin)
        ymax = max(ymax)
        veusz_set.append("Set('/data/graph1/y/min',"+str(float(ymin-(ymax-ymin)*0.2))+")")
        veusz_set.append("Set('/data/graph1/y/max',"+str(float(ymax+(ymax-ymin)*0.2))+")")
        veusz_set.append("Remove('/data/graph1/template')")
        veusz_set.append("Remove('/contour')")
        # save to vsz
        output_filename = 'G_u'
        shutil.copy2('template.vsz',output_filename+'.vsz')
        veusz_file = open(output_filename+'.vsz','a')
        for  i in veusz_set:
            veusz_file.write(i+'\n')
        veusz_file.close()
        # save data to .dat file
        print('Save data to '+output_filename+'.csv')
        plot_data[xlabel] = xdata 
        plot_df = pd.DataFrame(plot_data)
        plot_df.set_index(xlabel,inplace=True)
        plot_df.to_csv(output_filename+'.csv',index=True,float_format='%5.3f')
        if isveusz:
            embed.Load(output_filename+'.vsz')
            print('Export to '+output_filename+'.jpg')
            embed.Export(output_filename+'.jpg',dpi=300)
    else:
        print("Number of variables must less than 2!")
        exit(0)
    print('Save data to excel file G_result.xslx')
    data = data[['Nads','E_slab','ZPE_slab','E_ads','ZPE_ads','E_total','ZPE_total','dG','dG_avg','dG_step']]
    data.to_excel('G_result.xlsx',float_format='%.4f',index=False)
#    close_veusz(embed,vdisplay)