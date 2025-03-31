#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <ctime>
#include <thread>
#include <mutex>
#include <algorithm>

using namespace std;

struct Graph {
    vector<vector<pair<long, double>>> adjList;  
    unordered_map<long, long> nodeToIndex;       
    vector<long> indexToNode;                    
    
    void addEdge(long src, long dst, double prob) {
        if (nodeToIndex.find(src) == nodeToIndex.end()) {
            nodeToIndex[src] = adjList.size();
            indexToNode.push_back(src);
            adjList.emplace_back();
        }
        if (nodeToIndex.find(dst) == nodeToIndex.end()) {
            nodeToIndex[dst] = adjList.size();
            indexToNode.push_back(dst);
            adjList.emplace_back();
        }
        adjList[nodeToIndex[src]].emplace_back(nodeToIndex[dst], prob);
    }
};

Graph loadGraph(const string &fileName) {
    Graph graph;
    ifstream file(fileName);
    string line;
    long nnodes = 0, nedges = 0;
    
    vector<pair<long, long>> edges;
    vector<double> probs;
    edges.reserve(1000000);  
    
    while (getline(file, line)) {
        if (line[0] == '#') {
            istringstream iss(line);
            iss.ignore(1);
            iss >> nnodes >> nedges;
        } else {
            long src, dst;
            double prob;
            if (istringstream(line) >> src >> dst >> prob) {
                graph.addEdge(src, dst, prob);
            }
        }
    }
    return graph;
}

int simulateInfluence(const Graph &graph, long startNode, const unordered_set<long> &globalInfluenced, 
                     unordered_set<long> &newlyInfluenced, mt19937 &gen, uniform_real_distribution<> &dis) {
    vector<bool> activated(graph.adjList.size(), false);
    queue<long> q;
    
    if (globalInfluenced.find(startNode) == globalInfluenced.end()) {
        q.push(graph.nodeToIndex.at(startNode));
        activated[graph.nodeToIndex.at(startNode)] = true;
        newlyInfluenced.insert(startNode);
    }

    while (!q.empty()) {
        long nodeIdx = q.front();
        q.pop();

        for (const auto &neighbor : graph.adjList[nodeIdx]) {
            long neighborIdx = neighbor.first;
            double prob = neighbor.second;
            if (!activated[neighborIdx]) {
                if (dis(gen) < prob) {
                    activated[neighborIdx] = true;
                    q.push(neighborIdx);
                    newlyInfluenced.insert(graph.indexToNode[neighborIdx]);
                }
            }
        }
    }

    return newlyInfluenced.size();
}

void processNodeBatch(const Graph &graph, const vector<long> &nodes, 
                     const unordered_set<long> &globalInfluenced,
                     unordered_map<long, int> &influenceScores,
                     mt19937 &gen, uniform_real_distribution<> &dis,
                     mutex &mtx) {
    for (long node : nodes) {
        unordered_set<long> newlyInfluenced;
        int influence = simulateInfluence(graph, node, globalInfluenced, newlyInfluenced, gen, dis);
        lock_guard<mutex> lock(mtx);
        influenceScores[node] = influence;
    }
}

set<long> selectKNodes(Graph &graph, int k, int numThreads = 4) {
    set<long> selectedNodes;
    unordered_set<long> globalInfluenced;
    vector<thread> threads;
    mutex mtx;
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < k; ++i) {
        unordered_map<long, int> influenceScores;
        vector<long> candidates;
        
        for (const auto &nodeIndex : graph.nodeToIndex) {
            long node = nodeIndex.first;
            if (selectedNodes.find(node) == selectedNodes.end()) {
                candidates.push_back(node);
            }
        }
        
        // Process candidates in parallel
        int batchSize = (candidates.size() + numThreads - 1) / numThreads;
        for (int j = 0; j < numThreads; ++j) {
            int start = j * batchSize;
            if (start < candidates.size()) {
                int end = min(start + batchSize, (int)candidates.size());
                vector<long> batch(candidates.begin() + start, candidates.begin() + end);
                threads.emplace_back(processNodeBatch, ref(graph), batch, ref(globalInfluenced),
                                   ref(influenceScores), ref(gen), ref(dis), ref(mtx));
            }
        }
        
        for (auto &thread : threads) {
            thread.join();
        }
        threads.clear();
        
        long bestNode = -1;
        int maxInfluence = 0;
        for (const auto &score : influenceScores) {
            if (score.second > maxInfluence) {
                maxInfluence = score.second;
                bestNode = score.first;
            }
        }
        
        if (bestNode != -1) {
            selectedNodes.insert(bestNode);
            unordered_set<long> newlyInfluenced;
            simulateInfluence(graph, bestNode, globalInfluenced, newlyInfluenced, gen, dis);
            globalInfluenced.insert(newlyInfluenced.begin(), newlyInfluenced.end());
        }
    }
    
    return selectedNodes;
}

// Function to run multiple instances and select best nodes
set<long> runMultipleInstances(Graph &graph, int k, int numInstances) {
    unordered_map<long, int> nodeFrequency;  
    unordered_map<long, double> nodeSpread; 
    
    for (int instance = 0; instance < numInstances; ++instance) {
        set<long> selectedNodes = selectKNodes(graph, k);
        
        // Count frequency of selected nodes
        for (long node : selectedNodes) {
            nodeFrequency[node]++;
        }
    }

    vector<pair<long, int>> sortedNodes(nodeFrequency.begin(), nodeFrequency.end());
    sort(sortedNodes.begin(), sortedNodes.end(), 
         [](const pair<long, int>& a, const pair<long, int>& b) {
             return a.second > b.second;
         });
    
    set<long> bestNodes;
    for (int i = 0; i < min(50, (int)sortedNodes.size()); ++i) {
        bestNodes.insert(sortedNodes[i].first);
    }
    
    return bestNodes;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <graph_file> <output_file> k <# of instances>" << endl;
        return 1;
    }
    
    string fileName = argv[1];
    string outputFile = argv[2];
    int k = stoi(argv[3]);
    int numInstances = stoi(argv[4])/5;
    
    Graph graph = loadGraph(fileName);
    
    set<long> bestNodes = runMultipleInstances(graph, k, numInstances);
    
    ofstream outFile(outputFile);
    if (!outFile) {
        cerr << "Error opening output file!" << endl;
        return 1;
    }
    
    for (long node : bestNodes) {
        outFile << node << '\n';
    }
    
    outFile.close();
    return 0;
}
