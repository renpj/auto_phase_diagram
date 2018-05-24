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
    print('Initialize data ...')
    data = input_data.dropna(subset=['Nads','E_total']) # remove rows that are NaN for Nads and E_total
    data = data.dropna(axis=1,how='all')
    return data,ref_data
    
def check_data(data):
    # data is a pandas.DataFrame
    require_col = set((u'Nads', u'E_slab', u'E_total',))
    if not require_col.issubset(set(data.columns)):
        print('Error: Required Columns are ', ', '.join(list(require_col)))

    for icol in ('E_slab','E_ads','ZPE_slab','ZPE_ads','ZPE_total'): # fill NaN with i[0]
        v = data[icol][0]
        if pd.isnull(data[icol][0]):
            v = 0.0
        data[icol] = data[icol].fillna(v)
        if not np.all(data[icol] == v): # these cols should have the same values!
            print('Error: This column should have the same values: ', icol)
            
    data = data.groupby(by='Nads',as_index=False).agg(min) # 聚合相同的Nads, 取最小值。注意，没有作为新的index。

    data['G_slab'] = data['E_slab'] + data['ZPE_slab']
    data['G_total'] = data['E_total'] + data['ZPE_total']
    data['G_ads'] = (data['E_ads'] + data['ZPE_ads'])*data['Nads']

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

def get_ref(ref_data,formula):
    '''
    ref_data is pandas.DataFrame, formula is string.
    '''
    variable = {} # if T or p are variables, store them
    ref = {}
    co_names = parser.expr(formula).compile().co_names
    # get T
    t = ref_data['Temperature'][0]
    if pd.isnull(t):
        ref['T'] = 298.15 # default value for ref['T']
    elif type(t) == type(np.float64(0)): # t is a number, type is from pandas
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

    ref['S'] = {}
    ref['p'] = {}
    for iname in co_names:
        # assign entropy
        S = ref_data[ref_data.Ref == iname]['S']
        if S.size > 1:
            print ("Error: Duplicated S for "+iname)
            break
        elif S.size == 0 or S.isnull().iloc[0]:
            print ("Error: No S vaule for "+iname)
            break
        ref['S'][iname] = S.iloc[0]
    
        # assign pressure
        p = ref_data[ref_data.Ref == iname]['Press']
        if p.size > 1:
            print ("Error: Duplicated press for "+iname)
            break
        elif p.size == 0 or p.isnull().iloc[0]:
            ref['p'][iname] = 0 # unit ln(bar)
        else:
            if type(p.iloc[0]) == type(np.float64(0)):
                ref['p'][iname] = p.iloc[0]
            else:
                try:
                    ref['p'][iname] = np.array(np.log(eval(p.iloc[0]))) # ln(p)
                    if ref['p'][iname][0] != ref['p'][iname][1]:
                        variable[iname] = ref['p'][iname]
                    else:
                        ref['p'][iname] = ref['p'][iname][0]
                except Exception as e:
                    print("Error: Please check the Press format!",e)
    return ref,variable
 
def p_formula(ref,formula):
    map = {}
    for k in ref['p'].keys():
        map[k] = 'ref["p"]["'+k+'"]'
    return rebuild_formula(formula,map)
def s_formula(ref,formula):
    map = {}
    for k in ref['S'].keys():
        map[k] = 'ref["S"]["' + k + '"]'
    return rebuild_formula(formula,map)
    
if __name__ == '__main__':
    # Constant
    quality_2d = (500,500) # the quality for 2D contour map
    
    import sys
    args = sys.argv
    if not (len(args) == 2):
        print("usage: auto_phase_diagram.py xls_file")
        exit(0)
    filename = args[1]
    input_data,ref_data = data_from_xls(filename)
    data = check_data(input_data)
    formula = input_data['Formula_ads'][0]
    ref,variable = get_ref(ref_data,formula)
    pf = p_formula(ref,formula)
    sf = s_formula(ref,formula)
    u_p = 8.314*ref['T']*eval(pf)/1000/96.4853
    u_ts = -ref['T']*eval(sf)
    nvar = len(variable)
    print("Number of variable is "+str(nvar))
    embed,vdisplay = start_veusz()
    
    if nvar == 0:
        # 这意味着p和T都是一个值, 不做图
        data['G_ads'] = (data['E_ads'] + data['ZPE_ads'] + u_p + u_ts)*data['Nads']
        data['dG'] = data['G_total'] - data['G_slab'] - data['G_ads']
        data['dG_avg'] = data['dG']/data['Nads'] # 平均吸附能
        data['dG_step'] = data['dG']- data['dG'].shift().fillna(0)  # 分布吸附能

    elif nvar == 1:
        plot_data = {}
        vk,vv = list(variable.items())[0]       
        for irow in data.index:
            nads = int(data.iloc[irow]['Nads'])
            dG = data.iloc[irow]['dG']
            dG -= nads*(u_ts+u_p)
            plot_data[nads] = dG
        xdata = vv
        print('Generate G vs '+vk+' plot G_'+vk+'.vsz')
        if vk =='T':
            xlabel = 'Temperature (K)'
        else:
            xlabel = 'ln(p('+ vk + ')/p0)'
        veusz_set = []
        veusz_set.append("SetData('x',"+str(xdata.tolist())+")")
        ymin = []
        ymax = []
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
        output_filename = 'G_'+vk
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

    elif nvar == 2:           
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