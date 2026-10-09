from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import random
from geopy.distance import geodesic
import networkx as nx
import powerlaw

'''
satLlaDist={'Name':[Lat,Long,attitude]}
satLla=['Lat','Long','attitude','Name'] #satInfo
synGsInfo = ['Num', 'Name', 'Lat'(y), 'Long(x)']
'''

def readGroundStations(gsfilename):
    gsInfo = []
    with open(gsfilename, "r") as f:
        for line in f:
            line = line.split('\n')
            line = line[0]
            line = str(line).split(',')
            line[1] = float(line[1])
            line[2] = float(line[2])
            line[4] = float(line[4])
            gsInfo.append(line)

    return gsInfo


def ISL_Calculate(satLlaDist, orbitNum, satPerOrbit, earthRadius, orbitAltitude):
    ISLink = []
    ISLinkSet = set()
    ISLinkSetbuf = ()
    ISLdist = {}

    for orbit in range(orbitNum):
        for sat in range(satPerOrbit):
            satStr = 'Sat' + str(orbit) + '_' + str(sat)

            ISLbuf = [[], [], [], [], []]
            ISLlengthbuf = [[], [], [], [], []]
            ISLbuf[0] = satStr
            ISLbuf[1] = 'Sat' + str(orbit) + '_' + str(sat - 1)
            ISLbuf[2] = 'Sat' + str(orbit + 1) + '_' + str(sat)
            ISLbuf[3] = 'Sat' + str(orbit) + '_' + str(sat + 1)
            ISLbuf[4] = 'Sat' + str(orbit - 1) + '_' + str(sat)

            if orbit == 0 or orbit == (orbitNum - 1) or sat == 0 or sat == (satPerOrbit - 1):
                if sat == 0:
                    ISLbuf[1] = 'Sat' + str(orbit) + '_' + str(satPerOrbit - 1)

                if orbit == orbitNum - 1:
                    ISLbuf[2] = 'Sat' + str(0) + '_' + str(sat)

                if sat == (satPerOrbit - 1):
                    ISLbuf[3] = 'Sat' + str(orbit) + '_' + str(0)

                if orbit == 0:
                    ISLbuf[4] = 'Sat' + str(orbitNum - 1) + '_' + str(sat)
            # calculate the length of ISL
            satLocation0 = satLlaDist[ISLbuf[0]]
            satLocation1 = satLlaDist[ISLbuf[1]]
            satLocation2 = satLlaDist[ISLbuf[2]]
            satLocation3 = satLlaDist[ISLbuf[3]]
            satLocation4 = satLlaDist[ISLbuf[4]]
            ISLlengthbuf[0] = 0

            ISLlengthbuf[1] = geodesic((satLocation0[0], satLocation0[1]), (satLocation1[0], satLocation1[1])).m * (
                    earthRadius + orbitAltitude) / earthRadius
            ISLlengthbuf[2] = geodesic((satLocation0[0], satLocation0[1]), (satLocation2[0], satLocation2[1])).m * (
                    earthRadius + orbitAltitude) / earthRadius
            ISLlengthbuf[3] = geodesic((satLocation0[0], satLocation0[1]), (satLocation3[0], satLocation3[1])).m * (
                    earthRadius + orbitAltitude) / earthRadius
            ISLlengthbuf[4] = geodesic((satLocation0[0], satLocation0[1]), (satLocation4[0], satLocation4[1])).m * (
                    earthRadius + orbitAltitude) / earthRadius

            for i in range(1, 5):
                ISLink.append([ISLbuf[0], ISLbuf[i], ISLlengthbuf[i]])
                ISLinkSetbuf = (ISLbuf[0], ISLbuf[i], ISLlengthbuf[i])
                ISLinkSet.add(ISLinkSetbuf)

    return ISLink
    # print(ISLink)
    # print(ISLinkSet)


def satellite_Graph_Generate(satlla, ISlink):
    satgraph = nx.Graph()
    # add the node
    for satinfo in satlla:
        satgraph.add_node(satinfo[3], Lat=satinfo[0], Long=satinfo[1], Alt=satinfo[2])  # 添加节点１

    # add the edge
    for isl in ISlink:
        # print(isl)
        satgraph.add_edge(isl[0], isl[1], length=isl[2])

    # plt.show()
    return satgraph


def GSL_Graph_Generate(gsllink):
    gslGraph = nx.Graph()
    # add the node and edge
    i = 0
    for gslinfo in gsllink:
        gsnode = gslinfo[1]
        satnode = gslinfo[2]
        gslGraph.add_node(gsnode)
        gslGraph.add_node(satnode)
        gslGraph.add_edge(gsnode, satnode)
        i = i + 1

    return gslGraph


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
            output_dir = Path(__file__).resolve().parent / 'FigureOutput'
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'plotDistriution')
        plt.show()

    if plotGraphTopology:
        ps = nx.spring_layout(Graph)  # layout
        nx.draw(Graph, ps, with_labels=False, node_size=10)
        if save:
            output_dir = Path(__file__).resolve().parent / 'FigureOutput'
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'SatTerrainTopology')
        plt.show()

    return degrees,degreeFreq


def LogBinning(degreefreq):
    degreefreq = [freq / sum(degreefreq) for freq in degreefreq]  # 试一下看看
    binsEnd = np.log2(max(degrees))
    binsEnd = int(binsEnd) + 1
    lenDegFreq = len(degreefreq)
    print(lenDegFreq)
    zeroList = [0 for x in range(0, 2 ** binsEnd - max(degrees))]
    degreefreq = degreefreq + zeroList

    logBinningSpace = []
    for i in range(0, binsEnd):
        p = 2 ** i
        binStart = p
        binend = binStart + p - 1
        logBinningSpace.append([binStart, binend])

    print(logBinningSpace)
    xdatabinning = []
    ydatabinning = []

    for binList in logBinningSpace:
        Start = binList[0]
        End = binList[1]
        xdataAvg = (End + Start) / 2
        xdatabinning.append(xdataAvg)
        binningList = degreefreq[Start - 1:End]
        nodeNumber = End - Start + 1
        print(nodeNumber)
        binListSum = sum(binningList)
        ydatabinning.append(binListSum / nodeNumber)

    return xdatabinning, ydatabinning


if __name__ == '__main__':

    random.seed(12345678910)

    # yLength = 1

    gsfileName = Path(__file__).resolve().parent / 'worldcities.txt'
    gsInfo = readGroundStations(gsfileName)
    gsNumber = 1000

    synDataNum = 30000

    # latitude
    latSlide = 37
    latRange = np.linspace(-90, 90, latSlide)
    latDelta = 180 / (latSlide - 1)
    xLat=[-90+latDelta*i for i in range(1,len(latRange))]


    numLatItr = len(latRange)
    popLat = [0 for x in range(0, numLatItr - 1)]
    cityLat = [0 for x in range(0, numLatItr - 1)]
    citySynLat = [0 for x in range(0, numLatItr - 1)]

    for cityInfo in gsInfo[0:gsNumber]:
        for i in range(numLatItr - 1):
            if latRange[i] <= cityInfo[1] < latRange[i + 1]:
                popLat[i] = popLat[i] + cityInfo[4]
    popSum = sum(popLat)
    popLat = [Popu / popSum for Popu in popLat]

    for cityInfo in gsInfo[0:gsNumber]:
        for i in range(numLatItr - 1):
            if latRange[i] <= cityInfo[1] < latRange[i + 1]:
                cityLat[i] = cityLat[i] + 1
    citySum = sum(cityLat)
    cityLat = [cityN / citySum for cityN in cityLat]




    # longitude
    longSlide = 37
    longRange = np.linspace(-180, 180, longSlide)

    longDelta = 360 / (longSlide - 1)
    xLong = [-180 + longDelta * i for i in range(1, len(longRange))]

    numLongItr = len(longRange)
    popLong = [0 for x in range(0, numLongItr - 1)]
    cityLong = [0 for x in range(0, numLongItr - 1)]
    citySynLong = [0 for x in range(0, numLongItr - 1)]
    # print(gsInfo[0:gsNumber])
    for cityInfo in gsInfo[0:gsNumber]:
        for i in range(numLongItr - 1):
            if longRange[i] <= cityInfo[2] < longRange[i + 1]:
                popLong[i] = popLong[i] + cityInfo[4]

    popSum = sum(popLong)
    popLong = [Popu / popSum for Popu in popLong]

    for cityInfo in gsInfo[0:gsNumber]:
        for i in range(numLongItr - 1):
            if longRange[i] <= cityInfo[2] < longRange[i + 1]:
                cityLong[i] = cityLong[i] + 1
    citySum = sum(cityLong)
    cityLong = [cityN / citySum for cityN in cityLong]

    # plt.show()

    '2. Scale-model'

    '2.1. create latitude synthetic data'
    cumPopLat = np.cumsum(popLat)
    probLatDict = {}

    print(cumPopLat)

    startLat = 0
    endLat = 0
    for i in cumPopLat:
        if i == 0:
            startLat = startLat + 1
        if i < 0.9999:
            endLat = endLat + 1
        else:
            break

    for i in range(startLat, endLat + 1):
        probLatDict[latRange[i]] = cumPopLat[i]

    latSynDataRange = np.linspace(latRange[startLat], latRange[endLat + 1], endLat - startLat + 2)

    synLat = []
    for i in range(synDataNum):
        synProb = random.random()
        # print(startLat, endLat)
        # print('synProb=', synProb)
        for p in range(startLat, endLat + 1):
            if synProb < cumPopLat[p]:
                # print(' cumPopLat[p]=', cumPopLat[p])
                # print('lessRange', latRange[p])
                syndata = latRange[p] + latDelta * random.random()
                # print('syndata=', syndata)
                synLat.append(syndata)
                break

    for syn in synLat:
        for i in range(numLatItr - 1):
            if latRange[i] <= syn < latRange[i + 1]:
                citySynLat[i] = citySynLat[i] + 1
    citySynSumLat = sum(citySynLat)
    citySynLat = [citySynN / citySynSumLat for citySynN in citySynLat]

    print('xLat=',xLat)
    print('popLat=', popLat)
    print('cityLat=', cityLat)
    print('citySynLat=', citySynLat)
    'figure'
    # plt.plot(xLat, popLat,color='#D4232A', marker='^',linestyle='-',markersize=7, markeredgewidth=1.3, fillstyle='none' ,label='population')
    # plt.plot(xLat, cityLat, color='#E98B91',marker='D',linestyle='-',markersize=6.5, markeredgewidth=1.3, fillstyle='none', label='city')
    # plt.plot(xLat, citySynLat, color='#60301E',marker='s',linestyle='-',markersize=6.5, markeredgewidth=1.3, fillstyle='none', label='synthetic data')
    # plt.legend()
    # plt.xlabel('latitude (degree)')
    # plt.ylabel('proportion (%)')
    # plt.xticks(latRange[0::4])
    # plt.show()



    '2.2. create longitude synthetic data'
    cumPopLong = np.cumsum(popLong)
    probLongDict = {}
    print(cumPopLong)

    startLong = 0
    endLong = 0

    for i in cumPopLong:
        if i == 0:
            startLong = startLong + 1
        if i < 0.9999:
            endLong = endLong + 1
        else:
            break

    for i in range(startLong, endLong + 1):
        probLongDict[longRange[i]] = cumPopLong[i]

    longSynDataRange = np.linspace(longRange[startLong], longRange[endLong + 1], endLong - startLong + 2)

    synLong = []
    for i in range(synDataNum):
        synProb = random.random()
        # print(startLat, endLat)
        # print('synProb=', synProb)
        for p in range(startLong, endLong + 1):
            if synProb < cumPopLong[p]:
                # print(' cumPopLat[p]=', cumPopLat[p])
                # print('lessRange', latRange[p])
                syndata = longRange[p] + longDelta * random.random()
                # print('syndata=', syndata)
                synLong.append(syndata)
                break

    for syn in synLong:
        for i in range(numLongItr - 1):
            if longRange[i] <= syn < longRange[i + 1]:
                citySynLong[i] = citySynLong[i] + 1
                break
    citySynSumLong = sum(citySynLong)
    citySynLong = [citySynN / citySynSumLong for citySynN in citySynLong]

    print('xLong=',xLong)
    print('popLong=', popLong)
    print('cityLong=', cityLong)
    print('citySynLong=', citySynLong)
    'figure'
    # plt.figure()
    # plt.plot(xLong, popLong,color='#0F5C9C', marker='^',linestyle='-', markersize=7, markeredgewidth=1.3, fillstyle='none' ,label='population')
    # plt.plot(xLong, cityLong, color='#50A4CF',marker='D',linestyle='-',markersize=6.5, markeredgewidth=1.3, fillstyle='none', label='city')
    # plt.plot(xLong, citySynLong,  color='#008E89',marker='s',linestyle='-',markersize=6.5, markeredgewidth=1.3, fillstyle='none', label='synthetic data')
    # plt.xlabel('longitude (degree)')
    # plt.ylabel('proportion (%)')
    # plt.xticks(longRange[0::4])
    # plt.legend()
    # plt.show()



    '2.3 city data synthesis'
    # synGsInfo = ['Num', 'Name', 'Lat', 'Long']
    synGsInfo = []

    for i in range(synDataNum):
        synGsInfo.append([str(i), 'SynCity_' + str(i), synLat[i] / 180 + 0.5, synLong[i] / 360 + 0.5])

    # plt.figure()
    # plt.scatter(synLat, synLong)
    # plt.show()

    xLength = 1
    yLength = 1
    xNum = 33
    cirRadius = xLength / xNum * 0.5
    print('cirRadius=', cirRadius)
    yNum = int(np.floor(yLength / cirRadius * 0.5))
    print('yNum=', yNum)

    xstart = 0 + cirRadius
    ystart = 0 + cirRadius
    cirCenter = []
    circleDict = {}
    circleCountDict = {}
    circleNum = 0
    satLlaDist = {}  # satLlaDist={'Name':[Lat,Long,attitude]}
    satLla = []  # satLla=['Lat','Long','attitude','Name'] #satInfo
    # y是lat x是long
    # 圆圈和点的绘图

    # figure, axes = plt.subplots()
    for xnum in range(xNum):
        for ynum in range(yNum):

            # draw_circle = plt.Circle((xstart + 2 * cirRadius * xnum, ystart + 2 * cirRadius * ynum), cirRadius,
            #                          fill=False)
            # axes.add_artist(draw_circle)

            cirCenter.append([xstart + 2 * cirRadius * xnum, ystart + 2 * cirRadius * ynum])
            circleName = 'Sat' + str(circleNum // yNum) + '_' + str(circleNum % yNum)

            circleDict[circleName] = [xstart + 2 * cirRadius * xnum, ystart + 2 * cirRadius * ynum]
            circleCountDict[circleName] = 0
            satLla.append([ystart + 2 * cirRadius * ynum, xstart + 2 * cirRadius * xnum, 0, circleName])
            satLlaDist[circleName] = [ystart + 2 * cirRadius * ynum, xstart + 2 * cirRadius * xnum, 0]
            circleNum = circleNum + 1

    # for point in synGsInfo:
    #     plt.scatter(point[3], point[2], s=20, alpha=0.8)
    
    # plt.title('Circle')
    # plt.xlim([0, xLength])
    # plt.ylim([0, yLength])
    # axes.set_aspect(1)
    # plt.show()



    # print(circleDict)

    ISLink = ISL_Calculate(satLlaDist, xNum, yNum, 1, 0)
    # print(ISLink)

    # GSLLink = ['gsLength', 'gs', 'sat', gsLat(y), gsLon(x), satLat(y), satLon(x)]
    # synGsInfo = ['Num', 'Name', 'Lat', 'Long']
    GSLLink = []

    for city in synGsInfo:
        for i in range(xNum * yNum):
            circleName = 'Sat' + str(i // yNum) + '_' + str(i % yNum)
            center = circleDict[circleName]
            length = np.sqrt((city[3] - center[0]) ** 2 + (city[2] - center[1]) ** 2)
            if length <= cirRadius:
                circleCountDict[circleName] = circleCountDict[circleName] + 1
                gsllink = [length, city[1], circleName, city[2], city[3], center[1], center[0]]
                GSLLink.append(gsllink)

    # print(circleCountDict)


    '2.4 create Graph'
    satGraph = satellite_Graph_Generate(satLla, ISLink)
    gslGraph = GSL_Graph_Generate(GSLLink)

    'compose the graph'
    SatTerrain = nx.Graph()
    SatTerrain = nx.compose(SatTerrain, gslGraph)
    SatTerrain = nx.compose(satGraph, gslGraph)

    #nx.write_gml(SatTerrain, 'satGraphgml')

    '2.5 scale-free prove'
    degree_sequence = sorted([d for n, d in SatTerrain.degree()], reverse=True)
    print('degree_sequence=',degree_sequence)

    degrees, degreeFreq = DegreeDistribution(SatTerrain, plotDistriution=False, plotGraphTopology=False, save=False)

    degreeFreqStart4 = degreeFreq[4:]
    degreeFreqStart4 = [i / sum(degreeFreqStart4) for i in degreeFreqStart4]


    #degreeFreqStart4 = degreeFreq[4:]
    '''
    print('degrees=', degrees)
    print('degreeFreq=', degreeFreqStart4)
    print('')
    plt.figure()
    degrees = list(range(1, len(degreeFreqStart4) + 1))
    plt.loglog(degrees, degreeFreqStart4, color='#B2B2B2',marker='.', alpha=0.3, label='linear binning', linestyle='none')
    
    degreeBinning, degreeFreqBinning = LogBinning(degreeFreqStart4)
    print('degreeBinning=',degreeBinning)
    print('degreeFreqBinning=',degreeFreqBinning)
    plt.loglog(degreeBinning, degreeFreqBinning, '.', label='log binning')
    plt.xlabel('degree')
    plt.ylabel('frequency')
    plt.legend()
    a = plt.axes([.19, .18, .28, .28])
    plt.plot(degrees,degreeFreqStart4,color='#B2B2B2')
    plt.plot(degreeBinning, degreeFreqBinning,color='#125E9D')
    # plt.xticks()
    # plt.yticks()
    '''

    degrees, degreeFreq = DegreeDistribution(SatTerrain, plotDistriution=False, plotGraphTopology=False, save=False)
    plt.figure()
    degrees=degrees[1:]
    degreeFreq=degreeFreq[1:]
    degreeBinning2, degreeFreqBinning2 = LogBinning(degreeFreq)
    print('degrees=', degrees)
    print('degreeFreq=', degreeFreq)
    print('degreeBinning2=', degreeBinning2)
    print('degreeFreqBinning2=', degreeFreqBinning2)
    plt.loglog(degrees, degreeFreq, '.', color='#B2B2B2', alpha=0.5, label='linear')
    plt.loglog(degreeBinning2, degreeFreqBinning2, '.', label='log binning')
    plt.xlabel('k\'')
    plt.ylabel('p(k\')')
    plt.legend()
    a = plt.axes([.19, .18, .28, .28])
    plt.plot(degrees, degreeFreq, color='#B2B2B2')
    plt.plot(degreeBinning2, degreeFreqBinning2, color='#125E9D')
    # plt.xticks()
    # plt.yticks()
    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / 'figure'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'scaleFreeProve_BinAndLinear.png')
    plt.show()

    '==========================sub figure==============================='

    '2.6 P-Calcu'
    # fit = powerlaw.Fit(degree_sequence, discrete=True)
    # print('fit.power_law.xmin=',fit.power_law.xmin)
    # print('fit.power_law.alpha=',fit.power_law.alpha)
    # print('fit.power_law.D=', fit.power_law.D)
    # R, p = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True, nested=None)
    # print('R=',R,'p=',p)

    '2.7 show the original data'

    zero_count = degreeFreqStart4.count(0)
    print('zero_count=',zero_count)
    for i in range(zero_count):
        zero_index = degreeFreqStart4.index(0)
        degreeFreqStart4.pop(zero_index)
        degrees.pop(zero_index)


    print('degreeFreqStart4=',degreeFreqStart4)
    print('degreeBinning=', degreeBinning)
    print('degreeFreqBinning=', degreeFreqBinning)

    plt.figure()
    plt.loglog(degrees,degreeFreqStart4,'r.',alpha=0.5)
    plt.loglog(degreeBinning, degreeFreqBinning, 'b.',alpha=0.5)


    xdata=degreeBinning+degrees[0:20]
    ydata=degreeFreqBinning+degreeFreqStart4[0:20]
    plt.loglog(xdata, ydata, 'g.')
    plt.show()