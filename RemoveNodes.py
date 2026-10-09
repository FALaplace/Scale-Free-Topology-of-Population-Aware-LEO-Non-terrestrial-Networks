import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import random
import sys

# expontial func
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c

#power-law func
def func(x, a, b, c):
    return a*x ** (-b) + c

#power-law func(2-para)
def func2para(x, a, b):
    return a*x ** (-b)

def largestConnectedGraph(Graph):
    C = sorted(nx.connected_components(Graph), key=len, reverse=True)
    ConGraph = Graph.subgraph(C[0]) # this step makes the graph frozen
    return ConGraph

def DegreeDistribution(Graph,plotDistriution=True,plotGraphTopology=True,save=False):
    degreeFreq = nx.degree_histogram(Graph)
    degrees = list(range(len(degreeFreq)))
    degreeFreqSum = sum(degreeFreq)
    degreeFreqDist = [deg / degreeFreqSum for deg in degreeFreq]
    degreeFreq=degreeFreqDist

    if plotDistriution:
        plt.figure(figsize=(8, 6))
        plt.loglog(degrees, degreeFreq, 'go')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        if save:
            plt.savefig('FigureOutput/plotDistriution')
        plt.show()

    if plotGraphTopology:
        ps = nx.spring_layout(Graph)  # layout
        nx.draw(Graph, ps, with_labels=False, node_size=10)
        if save:
            plt.savefig('FigureOutput/SatTerrainTopology')
        plt.show()

    return degrees,degreeFreq

#CurveFit
def CurveFitPowerLaw(degrees,degreeFreq,plotFigure=True,save=False):#(xdata,ydata)
    xdata=degrees

    popt, pcov = curve_fit(func, xdata, degreeFreq)
    # popt数组中，三个值分别是待求参数a,b,c
    y2 = [func(i, popt[0], popt[1], popt[2]) for i in xdata]

    if save:
        plt.savefig('FigureOutput/Curvefit')
    if plotFigure:
        plt.loglog(xdata, y2, 'r--', label='CurveFit(powerlaw)')
        plt.loglog(xdata, degreeFreq, 'b.-', label='real')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.legend()

        tex = popt
        tex[0] = round(popt[0], 6)
        tex[1] = round(popt[1], 6)
        tex[2] = round(popt[2], 6)

        text = str(tex[0]) + '*exp(-' + str(tex[1]) + '*x)+' + str(tex[2])
        plt.text(10, 0.1, text, fontsize=12, color='black')

        plt.show()

    return popt

def CurveFitPowerLaw2Para(degrees,degreeFreq,plotFigure=True,save=False):#(xdata,ydata)
    xdata=degrees

    popt, pcov = curve_fit(func2para, xdata, degreeFreq)
    # popt数组中，三个值分别是待求参数a,b,c
    y2 = [func2para(i, popt[0], popt[1]) for i in xdata]

    if save:
        plt.savefig('FigureOutput/Curvefit')
    if plotFigure:
        plt.loglog(xdata, y2, 'r--', label='CurveFit(powerlaw)')
        plt.loglog(xdata, degreeFreq, 'b.-', label='real')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.legend()

        tex = popt
        tex[0] = round(popt[0], 6)
        tex[1] = round(popt[1], 6)

        text = str(tex[0]) + '*exp(-' + str(tex[1]) + '*x)'
        plt.text(10, 0.1, text, fontsize=12, color='black')
        plt.show()

    return popt

def xdataTranslation(xlist,degreeFreqNo0,startDegree,endDgree,xbias):
    startDegreeIndex=xlist.index(startDegree)
    endDgreeIndex=xlist.index(endDgree)+1
    xlist=xlist[startDegreeIndex:endDgreeIndex]
    degreeFreqTrans=degreeFreqNo0[startDegreeIndex:endDgreeIndex]
    xdata=[]
    for i in xlist:
        i=i-xbias
        xdata.append(i)
    return xdata,degreeFreqTrans

def listRemoveZero(degrees,degreeFreq):
    degreesnum = []
    degreesNo0 = degrees  # degrees without zero
    degreeFreqNo0 = degreeFreq
    for i in degreesNo0:
        if degreeFreqNo0[i] == 0:
            degreesnum.append(i)
    degreesnum.sort(reverse=True)
    for i in degreesnum:
        del degreeFreqNo0[i]
        del degreesNo0[i]

    return degreesNo0,degreeFreqNo0



'''
def GraphRandomlyRemoveNodes(Graph,p,returnConnected=True): #the return is connected graph
    # to unfreeze a graph you must make a copy by creating a new graph object
    #print(SatTerNodeList[1])
    unfrozen_graph = nx.Graph(Graph)
    GraphNodeList = list(unfrozen_graph.nodes)
    #
    # print(GraphNodeList)
    RemoveNodeNum = round(Graph.number_of_nodes()*p)
    # print(RemoveNodeNum)
    RemoveNodeList=random.sample(GraphNodeList,RemoveNodeNum)
    # print(RemoveNodeList)
    # print(len(unfrozen_graph.nodes))
    for nodeRemove in RemoveNodeList:
        unfrozen_graph.remove_node(nodeRemove)
    # print(unfrozen_graph.nodes)
    # print(len(unfrozen_graph.nodes))
    if returnConnected:
        unfrozen_graph = largestConnectedGraph(unfrozen_graph)
    return unfrozen_graph
'''
def GraphRandomlyRemoveNodes(Graph,p,returnLargestConnected=True): #the return is connected graph
    # to unfreeze a graph you must make a copy by creating a new graph object
    #print(SatTerNodeList[1])
    random.seed(123456789)
    unfrozen_graph = nx.Graph(Graph)
    GraphNodeList = list(unfrozen_graph.nodes)
    #
    # print(GraphNodeList)
    RemoveNodeNum = round(Graph.number_of_nodes()*p)
    # print(RemoveNodeNum)
    RemoveNodeList=random.sample(GraphNodeList,RemoveNodeNum)

    # print(len(unfrozen_graph.nodes))
    for nodeRemove in RemoveNodeList:
        unfrozen_graph.remove_node(nodeRemove)
    # print(unfrozen_graph.nodes)
    # print(len(unfrozen_graph.nodes))
    if returnLargestConnected:
        unfrozen_graph = largestConnectedGraph(unfrozen_graph)
    return unfrozen_graph



def GraphAttackRemoveNodesWithDegree(Graph,p,returnLargestConnected=True): #the return is connected graph

    unfrozen_graph = nx.Graph(Graph)
    DegreeSortList = sorted(unfrozen_graph.degree, key=lambda x: x[1], reverse=True)
    RemoveNodeNum = round(unfrozen_graph.number_of_nodes() * p)
    for i in range(RemoveNodeNum):
        nodeRemove=DegreeSortList[0][0]
        unfrozen_graph.remove_node(nodeRemove)
        DegreeSortList = sorted(unfrozen_graph.degree, key=lambda x: x[1], reverse=True)

    if returnLargestConnected:
        unfrozen_graph = largestConnectedGraph(unfrozen_graph)

    return unfrozen_graph

def GraphRandomlyRemoveRegionNodes(Graph,RemoveNodeNum,regionNodeList,returnLargestConnected=True): #the return is connected graph
    # to unfreeze a graph you must make a copy by creating a new graph object
    #print(SatTerNodeList[1])
    random.seed(123456789)
    unfrozen_graph = nx.Graph(Graph)
    GraphNodeList = regionNodeList
    print('GraphNodeList=',GraphNodeList)
    #
    # print(GraphNodeList)
    # print(RemoveNodeNum)
    RemoveNodeList=random.sample(GraphNodeList,RemoveNodeNum)

    for nodeRemove in RemoveNodeList:
        unfrozen_graph.remove_node(nodeRemove)
    # print(unfrozen_graph.nodes)
    # print(len(unfrozen_graph.nodes))
    if returnLargestConnected:
        unfrozen_graph = largestConnectedGraph(unfrozen_graph)
    return unfrozen_graph

def GraphAttackRemoveRegionNodes(Graph,RemoveNodeNum,sourceNode,targetNode,returnLargestConnected=True): #the return is connected graph
    # to unfreeze a graph you must make a copy by creating a new graph object
    #print(SatTerNodeList[1])
    random.seed(123456789)
    unfrozen_graph = nx.Graph(Graph)
    #
    # print(GraphNodeList)
    # print(RemoveNodeNum)
    for i in range(RemoveNodeNum):
        ShortestPathList=nx.shortest_path(unfrozen_graph, source=sourceNode, target=targetNode)
        nodeRemove=ShortestPathList[2]
        unfrozen_graph.remove_node(nodeRemove)

    if returnLargestConnected:
        unfrozen_graph = largestConnectedGraph(unfrozen_graph)
    return unfrozen_graph

def writeTwoLineList(xList,yList,writefilename):
    xnum=len(xList)
    ynum=len(yList)
    if xnum != ynum:
        print('xnum != ynum')
        sys.exit(1)
    with open(writefilename, "w") as f:
        for i in range(xnum):
            line=str(xList[i])+','+str(yList[i])
            f.write(line+'\n')

def readTwoLineList(readfilename):
    xdata = []
    ydata = []
    with open(readfilename, "r") as f:
        for line in f:
            line = line.split('\n')
            line = line[0]
            line = str(line).split(',')
            xdata.append(float(line[0]))
            ydata.append(float(line[1]))
    return xdata,ydata

def readOneLineList(readfilename):
    xdata = []
    with open(readfilename, "r") as f:
        for line in f:
            line = line.split('\n')
            line = line[0]
            xdata.append(line)
    return xdata

def writeMatrixList(GSLLink,writefilename):
    with open(writefilename, "w") as f:
        for line in GSLLink:
            for number in line:
                f.writelines(str(number)+',')
            f.writelines('\n')

def readMatrixList(readfilename):
    readlist=[]
    numberlist = []
    with open(readfilename, "r") as f:
        for line in f:
            line = line.split('\n')
            line = line[0]
            line = line.split(',')
            line = line[:-1]
            for str1 in line:
                numberlist.append(int(str1))
            readlist.append(numberlist)
            numberlist = []
    return readlist

def readStrMatrixList(readfilename):
    readlist=[]
    numberlist = []
    with open(readfilename, "r") as f:
        for line in f:
            line = line.split('\n')
            line = line[0]
            line = line.split(',')
            line = line[:-1]
            for str1 in line:
                numberlist.append(str(str1))
            readlist.append(numberlist)
            numberlist = []
    return readlist

def readFloatMatrixList(readfilename):
    readlist=[]
    numberlist = []
    with open(readfilename, "r") as f:
        for line in f:
            line = line.split('\n')
            line = line[0]
            line = line.split(',')
            line = line[:-1]
            for str1 in line:
                numberlist.append(float(str1))
            readlist.append(numberlist)
            numberlist = []
    return readlist

def readFloatMatrixList(readfilename):
    readlist=[]
    numberlist = []
    with open(readfilename, "r") as f:
        for line in f:
            line = line.split('\n')
            line = line[0]
            line = line.split(',')
            line = line[:-1]
            for str1 in line:
                numberlist.append(float(str1))
            readlist.append(numberlist)
            numberlist = []
    return readlist

def LogBinning(degreeFreq):
    degrees = list(range(1, len(degreeFreq) + 1))

    'processing'
    zeroList = []
    degreeFreq = [freq / sum(degreeFreq) for freq in degreeFreq]  # 试一下看看

    binsEnd = np.log2(max(degrees))
    binsEnd = int(binsEnd) + 1
    lenDegFreq = len(degreeFreq)
    # print(lenDegFreq)
    zeroList = [0 for x in range(0, 2 ** binsEnd - max(degrees))]
    degreeFreq = degreeFreq + zeroList

    logBinningSpace = []
    for i in range(0, binsEnd):
        p = 2 ** i
        binStart = p
        binend = binStart + p - 1
        logBinningSpace.append([binStart, binend])

    # print(logBinningSpace)
    xdataBinning = []
    ydataBinning = []

    for binList in logBinningSpace:
        Start = binList[0]
        End = binList[1]
        xdataAvg = (End + Start) / 2
        xdataBinning.append(xdataAvg)
        binningList = degreeFreq[Start - 1:End]
        nodeNumber = End - Start + 1
        # print(nodeNumber)
        binListSum = sum(binningList)
        ydataBinning.append(binListSum / nodeNumber)

    return xdataBinning,ydataBinning