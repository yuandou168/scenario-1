
from xml.etree.ElementTree import ElementTree
import numpy as np
import re, math
import xml.dom.minidom
from random import choice


class XML2DAG:
    """resolve a scientific workflow (XML file)：the set of jobs and its dependencies with a dag (an adjaceny matrix)"""
    adag_tag = "{http://pegasus.isi.edu/schema/DAX}adag"
    job_tag = "{http://pegasus.isi.edu/schema/DAX}job"
    child_tag = "{http://pegasus.isi.edu/schema/DAX}child"
    parent_tag = "{http://pegasus.isi.edu/schema/DAX}parent"
    uses_tag = "{http://pegasus.isi.edu/schema/DAX}uses"
    

    def __init__(self, file):
        self.xmlFile = file
        self.n_job = int(re.compile(r'\d+').findall(file)[0])
        self.DAG = np.zeros((self.n_job, self.n_job), dtype=int)  # structure of the workflow, an adjacent matrix
        self.jobType = np.zeros((self.n_job), dtype=int)          # types of the jobs in a workflow


    def get_dag(self):          # via minidom solver
        domtree = xml.dom.minidom.parse(self.xmlFile)
        collection = domtree.documentElement
        childrens = collection.getElementsByTagName("child")

        for child in childrens:
            child_id = child.getAttribute('ref')
            child_id = int(child_id[2:])
            # print('Child: ', child_id)
            parents = child.getElementsByTagName('parent')
            for parent in parents:
                parent_id = parent.getAttribute('ref')
                parent_id = int(parent_id[2:])
                # print(parent_id)
                self.DAG[parent_id, child_id] = 1
        return self.DAG


    def get_precursor(self):    # get the set of precursor nodes of a workflow
        precursor = []
        for i in range(self.n_job):
            temp = self.DAG[:, i]
            if np.sum(temp) == 0:
                precursor.append(i)
        return precursor


    def get_successor(self):    # get the successor node of a workflow
        successor = []
        for i in range(self.n_job):
            temp = self.DAG[i, :]
            if np.sum(temp) == 0:
                successor.append(i)
        return successor


    def print_graph(self):  # print the adjacency matrix
        # print(self.DAG)
        edges = []
        for i in range(self.n_job):
            for j in range(self.n_job):
                if self.DAG[i, j] != 0:
                    # print(i, ' -> ', j)
                    edge = (i, j)
                    edges.append(edge)
        # print(edges)
        return edges


    def jobs(self):  # the set of jobs in a workflow
        """attributes: id, name(job type), namespace (workflow), runtime, size """
        tree = ElementTree(file=self.xmlFile)
        root = tree.getroot()
        simple_jobs = []
        pattern = re.compile(r'\+?[1-9][0-9]*$|0$')    # 匹配第一个不为0的数字或者以0结尾的数字
        imagetypes = [200*1024*1024, 400*1024*1024, 1000*1024*1024, 2000*1024*1024]             # 镜像文件的大小， unit: *B
        for job in root.iter(tag=self.job_tag):
            input_size = []
            # print(len(job.findall(self.uses_tag)))
            for use in job.findall(self.uses_tag):
                if use.get('link')=='input':
                    use_input_file_size = int(use.get('size'))    # the usage files' size (unit: B)
                    # input_speed.append(round(use_file_size/(10*1024*1024),6))    # Network I/O bandwidth=10M/s
                    input_size.append(use_input_file_size)
                if use.get('link')=='output':
                    output_size = int(use.get('size'))
    
            simple_job = {'id': int(pattern.findall(job.attrib['id'])[0]), 
                            'name': job.attrib['name'], 
                            'namespace': job.attrib['namespace'],
                            'runtime': float(job.attrib['runtime']),    
                            'inputfilesize': sum(input_size),    # the total size of a job:  * B
                            'outputfilesize': output_size,      # the output sie of a job: *B
                            'imagesize': choice(imagetypes)   # the image size of the containerized task
                          }
            simple_jobs.append(simple_job)
        return simple_jobs


    def get_node(self, id):
        for job in self.jobs():
            # print(job)
            if id==job['id']:
                return job

    def types(self):  # the set of types of jobs
        types = []
        res = []
        for job in self.jobs(): 
            types.append(job['name'])
        for i, type in enumerate({}.fromkeys(types).keys()):
            res.append(type)
        for i, type in enumerate(types):
            self.jobType[i]=res.index(type)
            # print(self.taskType[i])
        return res, self.jobType


    def typeRTimeDicts(self, types, jobs):  # the set of runtime for each type of jobs
        typeRTimeDict = {}
        for typ in types:
            lst = []
            for job in jobs:
                if job['name'] == typ:
                    lst.append(job['runtime'])
            print(typ, lst)
            typeRTimeDict[typ] = lst
        return typeRTimeDict


    def typeTTimeDicts(self, types, jobs):  # the set of transfer time for each type of jobs
        typTTimeDict = {}
        for typ in types:
            lst = []
            for job in jobs:
                if job['name'] == typ:
                    lst.append(job['transtime'])
            typTTimeDict[typ] = lst
        return typTTimeDict



