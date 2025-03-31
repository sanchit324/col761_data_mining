#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <random>
#include <unordered_set>
#include<unordered_map>
#include <ctime>

using namespace std;

struct Graph {
    map<long, vector<long>> adjList;
    map<pair<long, long>, double> probs;
    void addEdge(long src, long dst) {
        adjList[src].push_back(dst);
        //adjList[dst].push_back(src);
    }
};

Graph loadGraph(const string &fileName) {
    Graph graph;
    ifstream file(fileName);
    string line;
    long nnodes = 0, nedges = 0;
    double prob;
    vector<pair<long, long>> edges;
    map<pair<long, long>, double> probs;
    while (getline(file, line)) {
        istringstream iss(line);
        if (line[0] == '#') {
            iss.ignore(1); // Skip "# Nodes: "
            iss >> nnodes;
            iss >> nedges;
            cout << "Nodes: " << nnodes << " Edges: " << nedges << endl;
        } else {
            long src, dst;
            if (iss >> src >> dst >> prob) {
                edges.emplace_back(src, dst);
                probs[{src, dst}] = prob;
            }
        }
    }
    file.close();
    for (const auto &edge : edges) {
        graph.addEdge(edge.first, edge.second);
    }
    graph.probs = move(probs);
    return graph;
}

int simulateInfluence(const Graph &graph, long startNode, const unordered_set<long> &globalInfluenced, unordered_set<long> &newlyInfluenced) {
    unordered_set<long> activated = globalInfluenced;  
    queue<long> q;
    
    if (activated.find(startNode) == activated.end()) {
        q.push(startNode);
        activated.insert(startNode);
        newlyInfluenced.insert(startNode);
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    while (!q.empty()) {
        long node = q.front();
        q.pop();

        for (long neighbor : graph.adjList.at(node)) {
            if (activated.find(neighbor) == activated.end()) {  
                double prob = graph.probs.at({node, neighbor});
                if (dis(gen) < prob) {  
                    activated.insert(neighbor);
                    q.push(neighbor);
                    newlyInfluenced.insert(neighbor);
                }
            }
        }
    }

    return newlyInfluenced.size();  
}

set<long> selectKNodes(Graph &graph, int k) {
    set<long> selectedNodes;
    unordered_set<long> globalInfluenced;
    
    for (int i = 0; i < k; ++i) {
        long bestNode = -1;
        int maxNewInfluence = 0;
        unordered_set<long> bestNewlyInfluenced;

        for (const auto& entry : graph.adjList) {
            long node = entry.first;  // Extract node manually
            if (selectedNodes.find(node) != selectedNodes.end()) continue;  
        
            unordered_set<long> newlyInfluenced;
            int newInfluence = simulateInfluence(graph, node, globalInfluenced, newlyInfluenced);
            
            if (newInfluence > maxNewInfluence) {
                maxNewInfluence = newInfluence;
                bestNode = node;
                bestNewlyInfluenced = newlyInfluenced;
            }
        }

        if (bestNode != -1) {
            selectedNodes.insert(bestNode);
            globalInfluenced.insert(bestNewlyInfluenced.begin(), bestNewlyInfluenced.end());
            //cout << "Selected Node: " << bestNode << " with " << maxNewInfluence << " new influences\n";
        }
    }
    
    return selectedNodes;
}



int main(int argc, char* argv[]){
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <graph_file> " << "<output_file> "<<"k "<<"<# of instances>"<<endl;
        return 1;
    }
    unordered_map<long,long> seedSet;
    string fileName = argv[1];
    string outputFile = argv[2];
    int k = stoi(argv[3]);
    Graph graph = loadGraph(fileName);
    set<long> influentialNodes = selectKNodes(graph, k);


    ofstream outFile(outputFile); 

    if (!outFile) {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    for (long node : influentialNodes) {
        outFile << node <<endl;
    }

    outFile.close(); // Close the file
    return 0;

    
}