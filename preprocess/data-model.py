from __future__ import print_function
from ortools.linear_solver import pywraplp
from XMLProcess import XML2DAG
from random import choice
import networkx as nx


def create_data_model():
    """Create the job data for the example."""
    
    graph = nx.DiGraph()
    wfname = './datasets/Montage_1000.xml'
    WF = XML2DAG(wfname)
    WF.get_dag()
    edges = WF.print_graph()
    graph.add_edges_from(edges)
    orderedjobs = list(nx.topological_sort(graph))

    print(nx.is_directed(graph)) # => True
    print(nx.is_directed_acyclic_graph(graph)) # => True
    
    # matching ids and sizes
    sizes = []
    for id in list(nx.topological_sort(graph)):
        sizes.append(WF.jobs()[id]['inputfilesize'])
        
    print(graph.nodes()) # => NodeView((a', 'b', 'e', 'c', 'd'))
    print(orderedjobs)
    print(sizes)
    jobData = {}
    jobData['sizes'] = sizes
    jobData['jobs'] = orderedjobs
    jobData['vms'] = [j for j in range(len(sizes))]
    vmTypeSize = [50*1024*1024*1024,100*1024*1024,200*1024*1024]
    jobData['vm_capacity'] = choice(vmTypeSize)
    return jobData
    

def main():
    data = create_data_model()

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')


    # Variables
    # x[i, j] = 1 if job i is packed in vm j.
    x = {}
    for i in data['jobs']:
        for j in data['vms']:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # y[j] = 1 if vm j is used.
    y = {}
    for j in data['vms']:
        y[j] = solver.IntVar(0, 1, 'y[%i]' % j)

    # Constraints
    # Each job must be in exactly one vm.
    for i in data['jobs']:
        solver.Add(sum(x[i, j] for j in data['vms']) == 1)

    # The amount shipped in each vm cannot exceed its capacity.
    for j in data['vms']:
        solver.Add(
            sum(x[(i, j)] * data['sizes'][i] for i in data['jobs']) <= y[j] *
            data['vm_capacity'])        

    # Objective: minimize the number of vms used.
    solver.Minimize(solver.Sum([y[j] for j in data['vms']]))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        num_vms = 0.
        for j in data['vms']:
            if y[j].solution_value() == 1:
                vm_jobs = []
                vm_size = 0 
                for i in data['jobs']:
                    if x[i, j].solution_value() > 0:
                        vm_jobs.append(i)
                        vm_size += data['sizes'][i]
                if vm_size > 0:
                    num_vms += 1
                    print('VM number', j)
                    print('  Jobs shipped:', vm_jobs)
                    print('  Total size:', vm_size)
                    print('  VM capacity:', data['vm_capacity'])
                    print()
        print()
        print('Number of vms used:', num_vms)
        print('Time = ', solver.WallTime(), ' milliseconds')
    else:
        print('The problem does not have an optimal solution.')


if __name__ == '__main__':
    main()    


