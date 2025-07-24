#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <cmath>

using namespace std;

int main() {
    // Open the input file
    ifstream input("input.txt");
    ofstream output("output.txt");
    
    if (!input.is_open() || !output.is_open()) {
        cerr << "Error opening files!" << endl;
        return 1;
    }
    
    string line;
    regex pattern("node_1=\\((\\d+),(\\d+)\\), node_2=\\((\\d+),(\\d+)\\)");
    
    while (getline(input, line)) {
        smatch matches;
        
        if (regex_search(line, matches, pattern) && matches.size() == 5) {
            int x1 = stoi(matches[1]);
            int y1 = stoi(matches[2]);
            int x2 = stoi(matches[3]);
            int y2 = stoi(matches[4]);
            
            // Calculate Manhattan distance
            int manhattan_distance = abs(x2 - x1) + abs(y2 - y1);
            
            // Write to output file
            output << "node_1=(" << x1 << "," << y1 << "), "
                  << "node_2=(" << x2 << "," << y2 << "), "
                  << "man_dis=" << manhattan_distance << endl;
        }
    }
    
    input.close();
    output.close();
    
    return 0;
}