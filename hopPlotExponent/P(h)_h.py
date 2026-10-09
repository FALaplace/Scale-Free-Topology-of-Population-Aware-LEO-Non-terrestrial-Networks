from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
INPUT_DATA_DIR = Path(r"F:\BaiduNetdiskDownload\毕业代码\NetworkSurvivability\satData")

DATA_DIR = Path(__file__).resolve().parent.parent / "satData"

import matplotlib.pyplot as plt

from RemoveNodes import *

def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)

    return [node for node, length in path_lengths.items() if length <= n]



if __name__ == '__main__':
    timeStep = 1440
    random.seed(12345678)
    startTime = 0
    IntervalNum = 0  # should start from 0
    timeInterval = 60

    RunConPath_Shell1 = '72_22_550_55_5000_60ms/'
    RunConPath_Shell2 = '32_50_1110_55_5000_60ms/'
    RunConPath_Shell3 = '8_50_1130_55_5000_60ms/'
    RunConPath_Shell123 = 'StarlinkShell123/'

    GraphFilePath_Shell1 = str(INPUT_DATA_DIR) + '/'  + RunConPath_Shell1 + 'Graph/'
    GraphFilePath_Shell2 = str(INPUT_DATA_DIR) + '/'  + RunConPath_Shell2 + 'Graph/'
    GraphFilePath_Shell3 = str(INPUT_DATA_DIR) + '/'  + RunConPath_Shell3 + 'Graph/'

    SatTerrain_Shell1 = nx.read_gml(GraphFilePath_Shell1 + 'satTerrainGraph/' + str(
        startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
    SatTerConGraph_Shell1 = largestConnectedGraph(SatTerrain_Shell1)  # this step makes the graph frozen
    #
    SatTerrain_Shell2 = nx.read_gml(
        GraphFilePath_Shell2 + 'satTerrainGraph/' + str(
            startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
    SatTerConGraph_Shell2 = largestConnectedGraph(SatTerrain_Shell2)  # this step makes the graph frozen

    # SatTerrain_Shell3 = nx.read_gml(
    #     GraphFilePath_Shell3 + 'satTerrainGraph/' + str(
    #         startTime + IntervalNum * timeInterval) + '_ms_satTerrainGraphgml')
    # SatTerConGraph_Shell3 = largestConnectedGraph(SatTerrain_Shell3)  # this step makes the graph frozen

    SatTerrain = nx.compose(SatTerConGraph_Shell1, SatTerConGraph_Shell2)
    # SatTerrain = nx.compose(SatTerrain, SatTerConGraph_Shell3)
    SatTerrain = largestConnectedGraph(SatTerrain)  # this step makes the graph frozen

    PhList=[]
    #
    # G = nx.Graph()
    # G.add_edges_from([('v1', 'v2'), ('v2', 'v4'), ('v1', 'v3'),('v4', 'v5')])
    # nx.draw(G, with_labels=True)
    # plt.show()
    diameter = nx.diameter(SatTerrain)
    # diameter = 21
    print(diameter)
    print()
    for d in range(diameter + 1):
        ph=0
        print('d=',d)
        for node in SatTerrain.nodes:
            # print(neighborhood(SatTerConGraph_Shell1, node,d))
            ph=ph+len(neighborhood(SatTerrain, node,d))
        PhList.append(ph)
    print(PhList)
    xList=range(0,diameter+1)
    plt.plot(xList,PhList)
    plt.plot(xList, PhList,'.')
    plt.show()

# shell1 [4645, 28609, 300503, 824261, 1813983, 3337167, 5746967, 8810441, 12293801, 15380145, 17616383, 19208857, 20300695, 20993259, 21360835, 21509735, 21557673, 21570647, 21574543, 21575759, 21576017, 21576025]
# shell2 [6406, 83232, 1425044, 3160812, 8141090, 14037390, 21824104, 28322256, 33361330, 36356496, 38293556, 39499906, 40208036, 40631662, 40865744, 40974636, 41020874, 41033470, 41036336, 41036802, 41036836]
# shell3 [3467, 16107, 484987, 1094801, 2540965, 4518067, 6900363, 8939385, 10239011, 11030735, 11474895, 11713617, 11853079, 11941853, 11994037, 12015577, 12019557, 12020023, 12020089]
# shell 1+2+3 [8415, 121845, 1648173, 4374431, 11530091, 22158969, 35381747, 46186027, 54528239, 60688311, 65026795, 67793761, 69400253, 70228361, 70605243, 70748455, 70795449, 70808763, 70811715, 70812201, 70812225]
