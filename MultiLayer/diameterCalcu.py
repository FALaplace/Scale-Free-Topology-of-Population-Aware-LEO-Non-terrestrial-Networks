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
    RunConPath_Shell123 = 'StarlinkShell123/'

    GraphFilePath_Shell1 = str(INPUT_DATA_DIR) + '/'  + RunConPath_Shell1 + 'Graph/'
    GraphFilePath_Shell2 = str(INPUT_DATA_DIR) + '/'  + RunConPath_Shell2 + 'Graph/'
    GraphFilePath_Shell3 = str(INPUT_DATA_DIR) + '/'  + RunConPath_Shell3 + 'Graph/'

    DiameterDataPathShell1 = str(DATA_DIR) + '/'  + RunConPath_Shell1 + 'Diameter/'
    DiameterDataPathShell2 = str(DATA_DIR) + '/'  + RunConPath_Shell2 + 'Diameter/'
    DiameterDataPathShell3 = str(DATA_DIR) + '/'  + RunConPath_Shell3 + 'Diameter/'
    DiameterDataPathShell123 = str(DATA_DIR) + '/'  + RunConPath_Shell123 + 'Diameter/'

    diameterShell1List = []
    diameterShell2List = []
    diameterShell3List = []
    diameterShell123List = []


    #curvefit
    poptData=[]
    degreeBinningData = []
    degreeFreqBinningData = []
    xdata=list(range(timeStep+1))

    # curvefitDataPathShell1 = 'satData/' + RunConPath_Shell1 + 'CurveFit/'
    # curvefitDataPathShell1 = 'satData/' + RunConPath_Shell1 + 'CurveFit/'


    # for IntervalNum in xdata:
    for IntervalNum in range(0,200):
        # read gml file
        print('IntervalNum=',IntervalNum)
        SatTerrain_Shell1 = nx.read_gml(
            GraphFilePath_Shell1 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
        SatTerConGraph_Shell1 = largestConnectedGraph(SatTerrain_Shell1)  # this step makes the graph frozen

        SatTerrain_Shell2 = nx.read_gml(
            GraphFilePath_Shell2 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
        SatTerConGraph_Shell2 = largestConnectedGraph(SatTerrain_Shell2)  # this step makes the graph frozen

        SatTerrain_Shell3 = nx.read_gml(
            GraphFilePath_Shell3 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
        SatTerConGraph_Shell3 = largestConnectedGraph(SatTerrain_Shell3)  # this step makes the graph frozen

        # print(GraphFilePath_Shell1 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
        # print(GraphFilePath_Shell2 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
        # print(GraphFilePath_Shell3 + 'satTerrainGraph/' + str(startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')

        SatTerrain = nx.compose(SatTerConGraph_Shell1, SatTerConGraph_Shell2)
        SatTerrain = nx.compose(SatTerrain, SatTerConGraph_Shell3)
        SatTerrain = largestConnectedGraph(SatTerrain)  # this step makes the graph frozen

        # diameterShell1 = nx.diameter(SatTerConGraph_Shell1)
        # diameterShell2 = nx.diameter(SatTerConGraph_Shell2)
        diameterShell3 = nx.diameter(SatTerConGraph_Shell3)
        # diameterShell123 = nx.diameter(SatTerrain)


        # diameterShell1List.append([diameterShell1])
        # print('diameterShell1=', diameterShell1)
        #
        # diameterShell2List.append([diameterShell2])
        # print('diameterShell2=', diameterShell2)

        diameterShell3List.append([diameterShell3])
        print('diameterShell3=', diameterShell3)

        # diameterShell123List.append([diameterShell123])
        # print('diameterShell123=', diameterShell123)





    # writeMatrixList(diameterShell1List, DiameterDataPathShell1 + 'Diameter.txt')
    # writeMatrixList(diameterShell2List, DiameterDataPathShell2 + 'Diameter.txt')
    Path(DiameterDataPathShell3).mkdir(parents=True, exist_ok=True)
    writeMatrixList(diameterShell3List, DiameterDataPathShell3 + 'Diameter.txt')
    # writeMatrixList(diameterShell123List, DiameterDataPathShell123 + 'Diameter.txt')



