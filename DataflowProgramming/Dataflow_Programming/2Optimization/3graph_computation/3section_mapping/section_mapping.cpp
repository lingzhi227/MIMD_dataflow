// given an input constains multiple PE grid sections
// given a grid dimension (m, n) representing m rows n columns of PEs
// the program need to decide whether the sections can be fit into the grid
// if true, the program need to generate a mapping solution to the output
// start means the north-left corner coordinate of the PE of corresponding section, end means the right-south corner coordinate of the PE of the section


#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <regex>
#include <algorithm>
#include <cmath>

using namespace std;

// Structure to represent a grid section
struct GridSection {
    int id;         // Grid section ID
    int rows;       // Number of rows
    int cols;       // Number of columns
    int startRow;   // Starting row coordinate
    int startCol;   // Starting column coordinate
    
    // Constructor
    GridSection(int id, int rows, int cols) : id(id), rows(rows), cols(cols), startRow(-1), startCol(-1) {}
};

// Function to parse input file
vector<GridSection> parseInput(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening input file: " << filename << endl;
        exit(1);
    }
    
    vector<GridSection> sections;
    string line;
    regex pattern("grid_(\\d+): \\((\\d+),(\\d+)\\)");
    smatch matches;
    
    while (getline(file, line)) {
        if (regex_search(line, matches, pattern)) {
            int id = stoi(matches[1]);
            int rows = stoi(matches[2]);
            int cols = stoi(matches[3]);
            sections.push_back(GridSection(id, rows, cols));
        }
    }
    
    file.close();
    return sections;
}

// Function to check if placing a grid section at a specific position is valid
bool isValidPlacement(const vector<vector<bool>>& grid, int startRow, int startCol, int rows, int cols) {
    int totalRows = grid.size();
    int totalCols = grid[0].size();
    
    // Check if the section fits within the grid boundaries
    if (startRow + rows > totalRows || startCol + cols > totalCols) {
        return false;
    }
    
    // Check if the section overlaps with any other section
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (grid[startRow + i][startCol + j]) {
                return false;
            }
        }
    }
    
    return true;
}

// Function to place a grid section at a specific position
void placeSection(vector<vector<bool>>& grid, GridSection& section, int startRow, int startCol) {
    for (int i = 0; i < section.rows; ++i) {
        for (int j = 0; j < section.cols; ++j) {
            grid[startRow + i][startCol + j] = true;
        }
    }
    
    section.startRow = startRow;
    section.startCol = startCol;
}

// Function to try fitting all the grid sections in the grid
bool fitSections(vector<vector<bool>>& grid, vector<GridSection>& sections, int index) {
    // If all sections have been placed, return true
    if (index == sections.size()) {
        return true;
    }
    
    GridSection& section = sections[index];
    int totalRows = grid.size();
    int totalCols = grid[0].size();
    
    // Try placing the current section at all possible positions
    for (int i = 0; i < totalRows; ++i) {
        for (int j = 0; j < totalCols; ++j) {
            if (isValidPlacement(grid, i, j, section.rows, section.cols)) {
                placeSection(grid, section, i, j);
                
                // Recursively try to place the next section
                if (fitSections(grid, sections, index + 1)) {
                    return true;
                }
                
                // If placing the next section is not possible, backtrack
                for (int r = 0; r < section.rows; ++r) {
                    for (int c = 0; c < section.cols; ++c) {
                        grid[i + r][j + c] = false;
                    }
                }
                
                section.startRow = -1;
                section.startCol = -1;
            }
        }
    }
    
    // If the current section couldn't be placed anywhere, return false
    return false;
}

// Function to write output to file
void writeOutput(const string& filename, int m, int n, const vector<GridSection>& sections, int totalArea) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening output file: " << filename << endl;
        exit(1);
    }
    
    // Add verbal information at the beginning
    file << "Total area needed for all sections: " << totalArea << endl;
    file << "Using grid dimensions: (" << m << "," << n << ") = " << m * n << " PEs" << endl;
    file << "Successfully mapped all sections!" << endl << endl;
    
    file << "for (m,n)=(" << m << "," << n << ")" << endl << endl;
    
    for (const auto& section : sections) {
        file << "grid_" << section.id << ": (" << section.rows << "," << section.cols 
             << "), start=(" << section.startRow << "," << section.startCol 
             << "), end=(" << (section.startRow + section.rows - 1) << "," 
             << (section.startCol + section.cols - 1) << ")" << endl;
    }
    
    // Add grid visualization to the output file
    file << endl << "Visualization:" << endl;
    
    // Create a grid for visualization
    vector<vector<int>> visualGrid(m, vector<int>(n, -1));
    
    // Fill the grid with section IDs
    for (const auto& section : sections) {
        if (section.startRow >= 0 && section.startCol >= 0) {
            for (int i = 0; i < section.rows; ++i) {
                for (int j = 0; j < section.cols; ++j) {
                    visualGrid[section.startRow + i][section.startCol + j] = section.id;
                }
            }
        }
    }
    
    // Write the grid to the file
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (visualGrid[i][j] == -1) {
                file << ". ";
            } else {
                file << visualGrid[i][j] << " ";
            }
        }
        file << endl;
    }
    
    file.close();
    cout << "Output written to " << filename << endl;
}

// Function to display the current mapping
void displayMapping(const vector<vector<bool>>& grid, const vector<GridSection>& sections) {
    cout << "Current mapping visualization:" << endl;
    
    // Create a grid for visualization
    int rows = grid.size();
    int cols = grid[0].size();
    vector<vector<int>> visualGrid(rows, vector<int>(cols, -1));
    
    // Fill the grid with section IDs
    for (const auto& section : sections) {
        if (section.startRow >= 0 && section.startCol >= 0) {
            for (int i = 0; i < section.rows; ++i) {
                for (int j = 0; j < section.cols; ++j) {
                    visualGrid[section.startRow + i][section.startCol + j] = section.id;
                }
            }
        }
    }
    
    // Display the grid
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (visualGrid[i][j] == -1) {
                cout << ". ";
            } else {
                cout << visualGrid[i][j] << " ";
            }
        }
        cout << endl;
    }
}

// Calculate total area needed for all sections
int calculateTotalArea(const vector<GridSection>& sections) {
    int totalArea = 0;
    for (const auto& section : sections) {
        totalArea += section.rows * section.cols;
    }
    return totalArea;
}

int main() {
    string inputFile = "input.txt";
    string outputFile = "output.txt";
    
    // Parse input file
    vector<GridSection> sections = parseInput(inputFile);
    
    // Calculate total area needed
    int totalArea = calculateTotalArea(sections);
    
    // Ask for grid dimensions
    int m, n;
    cout << "Enter grid dimensions (format: either 'm n' or '(m,n)'): ";
    
    string input;
    cin >> input;
    
    // Handle different input formats
    if (input.front() == '(' && input.find(',') != string::npos) {
        // Format: (m,n)
        sscanf(input.c_str(), "(%d,%d)", &m, &n);
    } else {
        // Just the first number
        m = stoi(input);
        
        // Get the second number
        cin >> n;
    }
    
    if (m <= 0 || n <= 0) {
        cerr << "Invalid grid dimensions. Both m and n must be positive." << endl;
        return 1;
    }
    
    // Create grid
    vector<vector<bool>> grid(m, vector<bool>(n, false));
    
    // Try to fit all sections
    if (fitSections(grid, sections, 0)) {
        // Write successful results to file only
        writeOutput(outputFile, m, n, sections, totalArea);
    } else {
        cout << "Failed to fit all sections in the grid of size (" << m << "," << n << ")." << endl;
        cout << "Total area needed: " << totalArea << endl;
        cout << "Grid capacity: " << m * n << endl;
        
        // Suggest a minimum grid size
        int minRows = 0, minCols = 0;
        for (const auto& section : sections) {
            minRows = max(minRows, section.rows);
            minCols = max(minCols, section.cols);
        }
        
        cout << "The minimum grid size needed is at least (" << minRows << "," << minCols << ")." << endl;
        cout << "Consider increasing the grid size. A total area of at least " 
             << totalArea << " PEs is needed." << endl;
             
        // Try to find a better grid size
        int suggestedM = ceil(sqrt(totalArea));
        int suggestedN = ceil(totalArea / (double)suggestedM);
        
        cout << "A suggested grid size could be (" << suggestedM << "," << suggestedN << ")." << endl;
    }
    
    return 0;
}