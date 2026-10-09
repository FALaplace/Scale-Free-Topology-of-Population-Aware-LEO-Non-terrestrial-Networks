from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
INPUT_DATA_DIR = Path(r"F:\BaiduNetdiskDownload\毕业代码\NetworkSurvivability\satData")

DATA_DIR = Path(__file__).resolve().parent.parent / "satData"

import matplotlib.pyplot as plt

from RemoveNodes import *

def PowerLaw(x, a, b):
    return a * x ** (-b)


if __name__ == '__main__':
    timeStep=1440
    random.seed(12345678)
    startTime = 0
    IntervalNum = 0  # should start from 0
    timeInterval = 60

    RunConPath_Shell1= '72_22_550_55_5000_60ms/'
    RunConPath_Shell2 = '32_50_1110_55_5000_60ms/'
    RunConPath_Shell3 = '8_50_1130_55_5000_60ms/'


    GraphFilePath_Shell1 = str(INPUT_DATA_DIR) + '/'  + RunConPath_Shell1 + 'Graph/'
    GraphFilePath_Shell2 = str(INPUT_DATA_DIR) + '/'  + RunConPath_Shell2 + 'Graph/'
    GraphFilePath_Shell3 = str(INPUT_DATA_DIR) + '/'  + RunConPath_Shell3 + 'Graph/'


    #curvefit
    poptData=[]
    degreeBinningData = []
    degreeFreqBinningData = []
    xdata=list(range(timeStep+1))

    # curvefitDataPathShell1 = 'satData/' + RunConPath_Shell1 + 'CurveFit/'
    # curvefitDataPathShell1 = 'satData/' + RunConPath_Shell1 + 'CurveFit/'


    # for IntervalNum in xdata:
    for IntervalNum in [3]:
        # read gml file
        print(IntervalNum)

        SatTerrain_Shell1 = nx.read_gml(
            GraphFilePath_Shell1 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
        SatTerConGraph_Shell1 = largestConnectedGraph(SatTerrain_Shell1)  # this step makes the graph frozen

        SatTerrain_Shell2 = nx.read_gml(
            GraphFilePath_Shell2 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
        SatTerConGraph_Shell2 = largestConnectedGraph(SatTerrain_Shell2)  # this step makes the graph frozen

        SatTerrain_Shell3 = nx.read_gml(
            GraphFilePath_Shell3 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
        SatTerConGraph_Shell3 = largestConnectedGraph(SatTerrain_Shell3)  # this step makes the graph frozen

        print(GraphFilePath_Shell1 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
        print(GraphFilePath_Shell2 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
        print(GraphFilePath_Shell3 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')

        SatTerrain = nx.compose(SatTerConGraph_Shell1, SatTerConGraph_Shell2)
        SatTerrain = nx.compose(SatTerrain, SatTerConGraph_Shell3)
        SatTerrain = largestConnectedGraph(SatTerrain)  # this step makes the graph frozen

        # curve fit
        degreesS1, degreeFreqS1 = DegreeDistribution(SatTerConGraph_Shell1, plotDistriution=False,
                                                     plotGraphTopology=False,
                                                     save=False)
        degreesS2, degreeFreqS2 = DegreeDistribution(SatTerConGraph_Shell2, plotDistriution=False,
                                                     plotGraphTopology=False,
                                                     save=False)
        degreesS3, degreeFreqS3 = DegreeDistribution(SatTerConGraph_Shell3, plotDistriution=False,
                                                     plotGraphTopology=False,
                                                     save=False)
        degreesAll, degreeFreqAll = DegreeDistribution(SatTerrain, plotDistriution=False, plotGraphTopology=False,
                                                 save=False)

        # 度从4
        'shell1'
        degreeFreqS1 = degreeFreqS1[4:]
        degreeFreqS1 = [freq / sum(degreeFreqS1) for freq in degreeFreqS1]
        degreesS1 = list(range(1, len(degreeFreqS1) + 1))
        degreeBinningS1, degreeFreqBinningS1 = LogBinning(degreeFreqS1)
        print('degreeFreqS1=',degreeFreqS1)
        print('degreeFreqBinningS1',degreeFreqBinningS1)

        # discard zero
        zero_count = degreeFreqS1.count(0)
        for i in range(zero_count):
            zero_index = degreeFreqS1.index(0)
            degreeFreqS1.pop(zero_index)
            degreesS1.pop(zero_index)




        'shell2'
        degreeFreqS2 = degreeFreqS2[4:]
        degreeFreqS2 = [freq / sum(degreeFreqS2) for freq in degreeFreqS2]
        degreesS2 = list(range(1, len(degreeFreqS2) + 1))
        degreeBinningS2, degreeFreqBinningS2 = LogBinning(degreeFreqS2)

        # discard zero
        zero_count = degreeFreqS2.count(0)
        for i in range(zero_count):
            zero_index = degreeFreqS2.index(0)
            degreeFreqS2.pop(zero_index)
            degreesS2.pop(zero_index)

        'shell3'
        degreeFreqS3 = degreeFreqS3[4:]
        degreeFreqS3 = [freq / sum(degreeFreqS3) for freq in degreeFreqS3]
        degreesS3 = list(range(1, len(degreeFreqS3) + 1))
        degreeBinningS3, degreeFreqBinningS3 = LogBinning(degreeFreqS3)

        # discard zero
        zero_count = degreeFreqS3.count(0)
        for i in range(zero_count):
            zero_index = degreeFreqS3.index(0)
            degreeFreqS3.pop(zero_index)
            degreesS3.pop(zero_index)

        'shellAll'
        degreeFreqAll=degreeFreqAll[4:]
        degreeFreqAll = [freq / sum(degreeFreqAll) for freq in degreeFreqAll]
        degreesAll = list(range(1, len(degreeFreqAll) + 1))
        degreeBinningAll, degreeFreqBinningAll = LogBinning(degreeFreqAll)
        # discard zero
        zero_count = degreeFreqAll.count(0)
        for i in range(zero_count):
            zero_index = degreeFreqAll.index(0)
            degreeFreqAll.pop(zero_index)
            degreesAll.pop(zero_index)





        # plt.loglog(degreesAll, degreeFreqAll, 'g.', label='log binning(All)')
        # print(degreesAll)

        'step 1'
        # degreeFreqStart4S1 = degreeFreqS1[4:]
        # degreeFreqStart4S1 = [freq / sum(degreeFreqStart4S1) for freq in degreeFreqStart4S1]  # normalization

        # degreeFreqStart4S2 = degreeFreqS2[4:]
        # degreeFreqStart4S2 = [freq / sum(degreeFreqStart4S2) for freq in degreeFreqStart4S2]  # normalization

        # degreeFreqStart4S3 = degreeFreqS3[4:]
        # degreeFreqStart4S3 = [freq / sum(degreeFreqStart4S3) for freq in degreeFreqStart4S3]  # normalization

        # degreeFreqStart4All = degreeFreqAll[4:]
        # degreeFreqStart4All = [freq / sum(degreeFreqStart4All) for freq in degreeFreqStart4All]  # normalization

        'step 2'
        # degreesS1 = list(range(1, len(degreeFreqStart4S1) + 1))
        # degreesS2 = list(range(1, len(degreeFreqStart4S2) + 1))
        # degreesS3 = list(range(1, len(degreeFreqStart4S3) + 1))
        #degreesAll = list(range(1, len(degreeFreqStart4All) + 1))

        'step 3'
        # degreeBinningS1, degreeFreqBinningS1 = LogBinning(degreeFreqStart4S1)
        # degreeBinningS2, degreeFreqBinningS2 = LogBinning(degreeFreqStart4S2)
        # degreeBinningS3, degreeFreqBinningS3 = LogBinning(degreeFreqStart4S3)
        # degreeBinningAll, degreeFreqBinningAll = LogBinning(degreeFreqAll)

        XdegreeS1 = degreeBinningS1[0:] + degreesS1[0:20]
        YdegreeFreqS1 = degreeFreqBinningS1[0:] + degreeFreqS1[0:20]

        poptS1, pcovS1 = curve_fit(PowerLaw, XdegreeS1, YdegreeFreqS1, maxfev=50000)
        yPowerlawS1 = [PowerLaw(i, poptS1[0], poptS1[1]) for i in XdegreeS1]
        print('poptS1=', poptS1)

        XdegreeS2 = degreeBinningS2[0:] + degreesS2[0:50]
        YdegreeFreqS2 = degreeFreqBinningS2[0:] + degreeFreqS2[0:50]

        XdegreeS3 = degreeBinningS3[0:] + degreesS3[0:10]
        YdegreeFreqS3 = degreeFreqBinningS3[0:] + degreeFreqS3[0:10]

        poptS3, pcovS3 = curve_fit(PowerLaw, XdegreeS3, YdegreeFreqS3, maxfev=50000)
        yPowerlawS3 = [PowerLaw(i, poptS3[0], poptS3[1]) for i in XdegreeS3]
        print('poptS3=', poptS3)

        XdegreeAll=degreeBinningAll[2:]+degreesAll[5:50]
        YdegreeFreqAll=degreeFreqBinningAll[2:]+degreeFreqAll[5:50]

        '''
        degreeBinningData.append(degreeBinning)
        degreeFreqBinningData.append(degreeFreqBinning)
        '''

        'step 4'
        # plt.loglog(degreeBinningS1, degreeFreqBinningS1, 'r-', label='log binning(Shell 1)')
        # plt.loglog(degreeBinningS1, degreeFreqBinningS1, 'r.')
        # plt.loglog(degreeBinningS2, degreeFreqBinningS2, 'b-', label='log binning(Shell 2)')
        # plt.loglog(degreeBinningS2, degreeFreqBinningS2, 'b.')
        # plt.loglog(degreeBinningS3, degreeFreqBinningS3, 'g-', label='log binning(Shell 3)')
        # plt.loglog(degreeBinningS3, degreeFreqBinningS3, 'g.')
        # plt.loglog(degreeBinningAll, degreeFreqBinningAll, 'm-', label='log binning')
        # plt.loglog(degreeBinningAll, degreeFreqBinningAll, 'm.')

        endDegree = -1
        figureX = 6
        'S1'
        plt.figure(figsize=(figureX,figureX*9/16))
        # plt.loglog(degreeBinningS1, degreeFreqBinningS1, '.', label='log binning')
        plt.loglog(XdegreeS1, yPowerlawS1, 'r.', label='log binning(pl)')
        plt.loglog(XdegreeS1, YdegreeFreqS1, color='#B2B2B2',marker='.', alpha=0.3, label='linear binning', linestyle='none')
        plt.xlabel('k')
        plt.ylabel('p(k)')
        plt.tight_layout()
        plt.legend()
        #plt.savefig('MultiLayer/figure/S1.png')

        # 'S2'
        '''
        plt.figure(figsize=(figureX,figureX*9/16))
        plt.loglog(degreeBinningS2, degreeFreqBinningS2, '.', label='bin')
        plt.loglog(degreesS2[0:endDegree], degreeFreqS2[0:endDegree], color='#B2B2B2',marker='.', alpha=0.3, label='linear binning', linestyle='none')
        #plt.loglog(XdegreeS2, YdegreeFreqS2, 'm.', label='log binning(All)')
        plt.xlabel('k')
        plt.ylabel('p(k)')
        plt.tight_layout()
        plt.legend()
        '''
        #plt.savefig('MultiLayer/figure/S2.png')
        #
        # 'S3'
        plt.figure(figsize=(figureX,figureX*9/16))
        plt.loglog(XdegreeS3, yPowerlawS3, 'r.', label='log binning(pl)')
        plt.loglog(XdegreeS3, YdegreeFreqS3, color='#B2B2B2',marker='.', alpha=0.3, label='linear binning', linestyle='none')
        #plt.loglog(XdegreeS3, YdegreeFreqS3, 'm.', label='log binning(All)')
        plt.xlabel('k')
        plt.ylabel('p(k)')
        plt.tight_layout()
        plt.legend()
        #plt.savefig('MultiLayer/figure/S3.png')
        #
        #
        # 'All'
        '''
        plt.figure(figsize=(figureX,figureX*9/16))
        plt.loglog(degreeBinningAll, degreeFreqBinningAll, '.', label='bin')
        plt.loglog(degreesAll[0:endDegree], degreeFreqAll[0:endDegree], color='#B2B2B2',marker='.', alpha=0.3, label='linear binning', linestyle='none')
        #plt.loglog(XdegreeAll, YdegreeFreqAll, 'm.', label='log binning(All)')
        plt.xlabel('k')
        plt.ylabel('p(k)')
        plt.legend()
        plt.tight_layout()
        #plt.savefig('MultiLayer/figure/ALL.png')
        '''
        plt.show()



