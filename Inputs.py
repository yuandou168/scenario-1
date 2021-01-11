from preprocess.XMLProcess import XML2DAG
from random import choice
import networkx as nx
from matplotlib import pyplot as plt
import json, csv
import pandas as pd
import numpy as np
from random import choice
import openpyxl
import math

def get_node(jobs, id):
    for job in jobs:
        # print(job)
        if id==job['id']:
            return job

'''Unify the Inputs for IC-PCP'''

n = '29'
wfname = './datasets/Sipht_'+n+'.xml'
desfolder = './inputs/sipht/'

deadline = str(2300000)

# 写入inputs文件 f1. dag, f2. profile, f3. data source, f4. performance, f5. price, f6. deadline
dagfilename = '1.'+n+'.1/1.'+n+'.1.dag' 
profilename = '1.'+n+'.1/1.'+n+'.1.propfile' 
visname = '1.'+n+'.1/1.'+n+'.1.png'
inputdatasource =  '1.'+n+'.1/1.'+n+'.1.datasource'
perfilename = '1.'+n+'.1/performance'
pricefilename = '1.'+n+'.1/price'
deadlinefilename = '1.'+n+'.1/deadline'


in_tmp = pd.read_excel("./datasets/WSP_dataset.xlsx", sheet_name="infrastructures")
cloudproviders = list(set(in_tmp.loc[:, 'Cloud providers']))
geolocation = list(set(in_tmp.loc[:, 'Geolocation']))
vm_instances = list(in_tmp.loc[:, 'vm instances'])
bandwidth = list(in_tmp.loc[:, 'Bandwidth (Gbps)'])
res_ratios = list(in_tmp.loc[:, 'Resource Ratio'])
p_r = list(in_tmp.loc[:, 'P_r ($/hour)'])
p_b = list(in_tmp.loc[:, 'P_b ($/GB)'])

print(cloudproviders, geolocation, vm_instances, res_ratios, bandwidth)


WF = XML2DAG(wfname)
WF.get_dag()                # 矩阵形式的DAG nxn
edges = WF.print_graph()    # 边集 
jobs = WF.jobs()            # 结点集, 0~n-1
x = WF.get_precursor()      # start 节点的children
y = WF.get_successor()      # exit 节点的parent
print('precursor', x, 'successor', y)
addEdge1 = []
# addEdge2 = []
for i in x:
    addEdge1.append((0, i+1))       # 添加start节点的边
# for j in y:
    # addEdge2.append((j+1, len(jobs)+1))   # 添加exit/end节点的边
# print(addEdge1, addEdge2)
# 更新jobs，更新edges
newOriginEdges = []         # 新的边集
for edge in edges:
    newOriginEdges.append((edge[0]+1, edge[1]+1))

newOriginEdges = addEdge1+newOriginEdges
# newOriginEdges = addEdge1+addEdge2+newOriginEdges
# print(newOriginEdges)


startnode = {'id': 0, 'name': 'start', 'namespace': '', 'runtime': float(0), 'inputfilesize': 0, 'outputfilzesize': 0, 'imagesize': 0}      # start节点
# exitnode = {'id': len(jobs)+1, 'name': 'exit', 'namespace': '', 'runtime': float(0), 'inputfilesize': 0, 'outputfilzesize': 0, 'imagesize': 0}
newJobs = [startnode]                # 新的点集
for i, job in enumerate(jobs):
    newnode={ 'id': i+1, 'name': job['name'], 'namespace': job['namespace'], 'runtime': job['runtime'], 'inputfilesize': job['inputfilesize'], 
            'outputfilesize': job['outputfilesize'], 'imagesize': job['imagesize']}
    newJobs.append(newnode)
# newJobs.append(exitnode)

print(newJobs)
# types = WF.types()          # 结点的类型集合
# jobtypes = WF.typeRTimeDicts(types[0], jobs)  


dag = {'nodes': [], 'links': []}
datasource = {'nodes': []}
geos = ['europe-west4', 'west europe', 'europe, Paris']
getAccessContext = [[1,0,0], [0,1,0], [1,0,1]]      # data access policy--选数据中心-选vm types()-

# dag['nodes'].append({'order': 0, 'name': 't0'})
for id, job in enumerate(newJobs):
    dag['nodes'].append({'order': id, 'name': 't'+str(id)})
# dag['nodes'].append({'order': len(jobs)+1, 'name': 't'+str(len(jobs)+1)})

# weight ==> data and image preparation: communication time and deployment time
with open(desfolder + profilename, 'w') as f2:
    f2.writelines('digraph dag {\n')
    for edge in newOriginEdges:
        comm = get_node(newJobs, edge[0])['inputfilesize']/(bandwidth[0]*1024*1024)
        depl = get_node(newJobs, edge[1])['imagesize']/(bandwidth[0]*1024*1024)
        f2.writelines('\t'+str(edge[0]) +' -> '+ str(edge[1])+'\t[weight='+ str(math.ceil(comm+depl))+'];\n')
        dag['links'].append({'source': 't'+str(edge[0]), 'target': 't'+str(edge[1]), 'throughput': math.ceil(comm+depl)})
    f2.writelines('}')
f2.close()

with open(desfolder+dagfilename, 'w') as f1:
    f1.write(json.dumps(dag,indent=4,separators=(',', ': ')))
f1.close()

# original input data
for i, id in enumerate(x):
    datasource['nodes'].append({'order':id+1, 'name': 't'+str(id+1), 'geo': choice(geos)})
with open(desfolder+inputdatasource, 'w') as f3:
    f3.write(json.dumps(datasource,indent=4,separators=(',', ': ')))
f3.close()

with open(desfolder +pricefilename, 'w') as f4:
    price = str(p_r).strip('[]')
    f4.writelines(price)
f4.close

with open(desfolder+ deadlinefilename, 'w') as f5:
    f5.writelines(deadline)
f5.close()
# print(in_tmp)

exetime = []
for j, res_ratio in enumerate(res_ratios): 
    tmp = [] 
    for i, job in enumerate(newJobs):
        executetime = job['runtime']/res_ratio
        time = math.ceil(executetime)      # node performance without the limitation of data policy
        tmp.append(time)
    exetime.append(tmp)
matrix = np.asarray(exetime)

df = pd.DataFrame(data=matrix, index=[vmtype for vmtype in vm_instances], columns=[job['name'] for job in newJobs])
# df.to_excel(desfolder+".performance.xlsx",  sheet_name="execution time")
df.to_csv(desfolder+perfilename, index=False, header=False)

graph = nx.DiGraph()
graph.add_edges_from(newOriginEdges) # 添加边集
# orderedjobs = list(nx.topological_sort(graph))  # 拓扑排序

print(wfname)
print(nx.is_directed_acyclic_graph(graph)) # 判断给定的workflow是否是有向无环图 => True
    
plt.tight_layout()
nx.draw_networkx(graph, arrows=True)
plt.savefig(desfolder+visname, format="PNG")
plt.clf()

# matching ids and sizes
sizes = []
for id in list(nx.topological_sort(graph)):
    sizes.append(newJobs[id]['inputfilesize'])
        
# print(graph.nodes()) # => NodeView((a', 'b', 'e', 'c', 'd'))
# print(orderedjobs)
# print(sizes)

