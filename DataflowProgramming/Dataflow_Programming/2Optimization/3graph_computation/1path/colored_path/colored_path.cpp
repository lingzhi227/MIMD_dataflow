// Given an m x n grid. Each cell of the grid has a sign pointing to the next cell you should visit if you are currently in this cell. The sign of grid[i][j] can be:

// 1 which means go to the cell to the right. (i.e go from grid[i][j] to grid[i][j + 1])
// 2 which means go to the cell to the left. (i.e go from grid[i][j] to grid[i][j - 1])
// 3 which means go to the lower cell. (i.e go from grid[i][j] to grid[i + 1][j])
// 4 which means go to the upper cell. (i.e go from grid[i][j] to grid[i - 1][j])
// Notice that there could be some signs on the cells of the grid that point outside the grid.

// You will initially start at the upper left cell (0, 0). A valid path in the grid is a path that starts from the upper left cell (0, 0) and ends at the bottom-right cell (m - 1, n - 1) following the signs on the grid. The valid path does not have to be the shortest.

// You can modify the sign on a cell with cost = 1. You can modify the sign on a cell one time only.

// Return the minimum cost to make the grid have at least one valid path.



#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <climits>
#include <string>
#include <cctype>
using namespace std;

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    string line;
    
    // Process each non-empty line from the input file.
    while(getline(fin, line)) {
        if(line.empty()) continue;
        
        // Expected line format: grid = [[...], [...], ...]
        // We'll keep the original line to output later.
        string originalLine = line;
        // Remove the prefix "grid = " to get the grid string.
        size_t pos = line.find("grid = ");
        if(pos != string::npos) {
            string gridStr = line.substr(pos + 7);
            vector<vector<int>> grid;
            vector<int> row;
            string num;
            // Parse the grid string character by character.
            for (char c : gridStr) {
                if(isdigit(c)) {
                    num.push_back(c);
                } else {
                    if(!num.empty()){
                        row.push_back(stoi(num));
                        num.clear();
                    }
                    // When we encounter a closing bracket, it indicates end of a row.
                    if(c == ']') {
                        if(!row.empty()){
                            grid.push_back(row);
                            row.clear();
                        }
                    }
                }
            }
            
            // Dimensions of the grid.
            int m = grid.size();
            int n = grid.empty() ? 0 : grid[0].size();
            
            // Set up 0-1 BFS.
            vector<vector<int>> dist(m, vector<int>(n, INT_MAX));
            deque<pair<int, int>> dq;
            dist[0][0] = 0;
            dq.push_back({0, 0});
            
            // The directions: 1->right, 2->left, 3->down, 4->up.
            vector<int> dx = {0, 0, 1, -1};
            vector<int> dy = {1, -1, 0, 0};
            
            while(!dq.empty()){
                auto [x, y] = dq.front();
                dq.pop_front();
                int cost = dist[x][y];
                // Try all four possible moves.
                for (int k = 0; k < 4; k++){
                    int nx = x + dx[k];
                    int ny = y + dy[k];
                    // Check if the neighbor is within bounds.
                    if(nx < 0 || nx >= m || ny < 0 || ny >= n) continue;
                    // If the current cell's sign (1-indexed) equals k+1, no cost is added.
                    int newCost = cost + ((grid[x][y] == k + 1) ? 0 : 1);
                    if(newCost < dist[nx][ny]){
                        dist[nx][ny] = newCost;
                        // If no cost, push to the front of deque.
                        if(grid[x][y] == k + 1)
                            dq.push_front({nx, ny});
                        else
                            dq.push_back({nx, ny});
                    }
                }
            }
            
            int ans = dist[m-1][n-1];
            // Write output in the required format.
            fout << originalLine << ", cost=" << ans << "\n";
        }
    }
    
    fin.close();
    fout.close();
    return 0;
}
