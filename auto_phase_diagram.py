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
    input_data = pd.read_excel(filename,sheetname=0)
    ref_data = pd.read_excel(filename,sheetname=1)
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
        print(' '.join(e,s))
        exit(0)
    
def parse_formula(s):
    '''
    Parse the string formula to a list.
    '''
    check_formula(s)
    result = []
    var = ''
    for id,i in enumerate(s):
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
    for k,v in map.iteritems():
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
    elif type(t) == np.float64: # t is a number, type is from pandas
        ref['T'] = t
    else: # t is a variable
        try:
            ref['T'] = np.array(eval(t))
            variable['T'] = ref['T']
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
            if type(p.iloc[0]) == np.float64:
                ref['p'][iname] = p.iloc[0]
            else:
                try:
                    ref['p'][iname] = np.array(np.log(eval(p.iloc[0]))) # ln(p)
                    variable[iname] = ref['p'][iname]
                except Exception as e:
                    print("Error: Please check the Press format!",e)
    return ref,variable
 
def p_formula(ref,formula):
    map = {}
    for k in ref['p'].iterkeys():
        map[k] = 'ref["p"]["'+k+'"]'
    return rebuild_formula(formula,map)
def s_formula(ref,formula):
    map = {}
    for k in ref['S'].iterkeys():
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
        vk,vv = variable.items()[0]       
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
        # 使用点线图更方便？YES！ 
        #embed.Load('template.vsz')
        #all_widget = [i.name for i in embed.Root.WalkWidgets()] 
        all_widget = []
        veusz_set = []
        veusz_set.append("SetData('x',"+str(xdata.tolist())+")")
        #embed.SetData('x', xdata)
        ymin = []
        ymax = []
        for nads in plot_data:
            dG = plot_data[nads].tolist()
            name = 'G' + str(nads)
            path = '/data/graph1/' + name
            if name not in all_widget:
                #embed.CloneWidget('/data/graph1/template','/data/graph1',name)
                veusz_set.append("CloneWidget('/data/graph1/template','/data/graph1','"+name+"')")
            veusz_set.append("Set('"+path+"/key', 'N="+str(nads)+"')")
            veusz_set.append("Set('"+path+"/xData','x')")
            veusz_set.append("SetData('" + name + "', " +str(dG)+")")
            veusz_set.append("Set('"+path+"/yData','"+name+"')")
            #embed.Set(path+'/key', 'N='+str(nads))
            #embed.Set(path+'/xData','x')
            #embed.SetData(name, dG)
            #embed.Set(path+'/yData',name)
            #print function
            all_widget.append(name)
            ymin.append(min(dG))
            ymax.append(max(dG))
        veusz_set.append("Set('/data/graph1/x/min',"+str(float(min(xdata)))+")")
        veusz_set.append("Set('/data/graph1/x/max',"+str(float(max(xdata)))+")")
        veusz_set.append("Set('/data/graph1/x/label','"+xlabel+"')")
        #embed.Set('/data/graph1/x/min',float(min(xdata)))
        #embed.Set('/data/graph1/x/max',float(max(xdata)))
        #embed.Set('/data/graph1/x/label', xlabel)
        ymin = min(ymin)
        ymax = max(ymax)
        veusz_set.append("Set('/data/graph1/y/min',"+str(float(ymin-(ymax-ymin)*0.2))+")")
        veusz_set.append("Set('/data/graph1/y/max',"+str(float(ymax+(ymax-ymin)*0.2))+")")
        #embed.Set('/data/graph1/y/min',float(ymin-(ymax-ymin)*0.2))
        #embed.Set('/data/graph1/y/max',float(ymax+(ymax-ymin)*0.2))
        veusz_set.append("Remove('/data/graph1/template')")
        veusz_set.append("Remove('/function')")
        veusz_set.append("Remove('/contour')")
        #embed.Remove('/data/graph1/template')
        #embed.Remove('/function')
        #embed.Remove('/contour')
        #print('Export to G_'+vk+'.jpg')
        #embed.Export('G_'+vk+'.jpg',dpi=300)
        #embed.Save('G_'+vk+'.vsz')
        veusz_filename = 'G_'+vk+'.vsz'
        shutil.copy2('template.vsz',veusz_filename)
        veusz_file = open(veusz_filename,'a')
        for  i in veusz_set:
            veusz_file.write(i+'\n')
        veusz_file.close()
        # save data to .dat file
        print('Save data to G_'+vk+'.csv')
        plot_data['ln(p('+vk+'))'] = vv 
        plot_df = pd.DataFrame(plot_data)
        plot_df.set_index('ln(p('+vk+'))',inplace=True)
        plot_df.to_csv('G_'+vk+'.csv',sep='\t',index=True,float_format='%5.3f')
        if isveusz:
            embed.Load(veusz_filename)
            print('Export to G_'+vk+'.jpg')
            embed.Export('G_'+vk+'.jpg',dpi=300)

    elif nvar == 2:
        u = np.array([u_p.min()+u_ts.min(),u_p.max()+u_ts.max()])
        if isveusz:
            # 作图 G vs u
            print('Generate G vs u plot G_u.vsz')
            xmin,xmax = u
            ymin = []
            ymax = []
            embed.Load('template.vsz')
            all_widget = [i.name for i in embed.Root.WalkWidgets()] 
            xdata = u
            embed.SetData('x', xdata)
            for irow in data.index:
                nads = int(data.iloc[irow]['Nads'])
                dG = data.iloc[irow]['dG']
                dG -= nads*u
                name = 'G' + str(nads)
                path = '/data/graph1/' + name
                if name not in all_widget:
                    embed.CloneWidget('/data/graph1/template','/data/graph1',name)
                embed.Set(path+'/xData','x')
                embed.SetData(name, dG)
                embed.Set(path+'/yData',name)
                embed.Set(path+'/key', 'N='+str(nads))
                all_widget.append(name)   
                ymin.append(min(dG))
                ymax.append(max(dG))
            embed.Set('/data/graph1/x/min',float(xmin))
            embed.Set('/data/graph1/x/max',float(xmax))
            ymin = min(ymin)
            ymax = max(ymax)
            embed.Set('/data/graph1/y/min',float(ymin-(ymax-ymin)*0.2))
            embed.Set('/data/graph1/y/max',float(ymax+(ymax-ymin)*0.2))
            embed.Remove('/data/graph1/template')
            embed.Remove('/function')
            embed.Remove('/contour')
            print("Export to G_u.jpg")
            embed.Export('G_u.jpg',dpi=300)
            embed.Save('G_u.vsz')
        else:
            # save data to .dat file
            pass
        k_notT = [i for i in variable if i!= 'T'] # get var that isnot T
        ylabel = 'ln(p('+ k_notT[0] + ')/p0)'
        ydata = np.linspace(variable[k_notT[0]][0],variable[k_notT[0]][1],quality_2d[1])
        if len(k_notT) == 1:
            xlabel = 'Temperature (K)'
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
        if isveusz:
            # 生成等值面图
            print('Generate 2D contour ','_'.join(variable.keys())+'_2D.vsz')
            embed.Load('template.vsz')
            all_widget = [i.name for i in embed.Root.WalkWidgets()] 
            embed.SetData2D('grid',ngrid,xcent=xdata,ycent=ydata)
            embed.Set('/contour/graph1/image1/data','grid')
            embed.Set('/contour/graph1/image1/colorMap', u'blue-darkorange')
            embed.Set('/contour/graph1/x/label', xlabel)
            xmin = min(xdata)
            xmax = max(xdata)
            embed.Set('/contour/graph1/x/min', float(xmin))
            embed.Set('/contour/graph1/x/max', float(xmax))
            embed.Set('/contour/graph1/y/label', ylabel)
            ymin = min(ydata)
            ymax = max(ydata)
            embed.Set('/contour/graph1/y/min', float(ymin))
            embed.Set('/contour/graph1/y/max', float(ymax))
            embed.Remove('/data')
            embed.Remove('/function')
            print('Export jpg image to ','_'.join(variable.keys())+'_2D.jpg')
            embed.Export('_'.join(variable.keys())+'_2D.jpg',dpi=300)
            embed.Save('_'.join(variable.keys())+'_2D.vsz')
        else:
            pass
            # save data to .dat file
            
    print('Save data to excel file G_result.xslx')
    data = data[['Nads','E_slab','ZPE_slab','E_ads','ZPE_ads','E_total','ZPE_total','dG','dG_avg','dG_step']]
    data.to_excel('G_result.xlsx',float_format='%.4f',index=False)
#    close_veusz(embed,vdisplay)