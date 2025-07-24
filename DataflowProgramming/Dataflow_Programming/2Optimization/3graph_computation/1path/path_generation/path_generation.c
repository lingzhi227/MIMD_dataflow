// on an m x n grid. 
// start at the top-left corner (i.e., PE[0][0]). 
// end to the bottom-right corner (i.e., PE[m - 1][n - 1]). 
// move either (e)ast or (s)outh at any point in time.

// Given the two integers m and n, 
// return the number of possible unique paths from start to end.
// print the possible unique paths in the format: PATH{eeeessees....}, e means east one step, s means south one step. and the PATH{} is recorded steps take


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Function to calculate combinations
unsigned long long combination(int t, int k) {
    if (k < 0 || k > t) return 0;
    if (k == 0 || k == t) return 1;
    if (k > t - k) {
        k = t - k;
    }
    unsigned long long result = 1;
    for (int i = 1; i <= k; i++) {
        result *= (t - k + i);
        result /= i;
    }
    return result;
}

// Function to parse input line
bool parse_input_line(char *line, int *m, int *n, int *start_x, int *start_y, int *end_x, int *end_y) {
    return sscanf(line, "m=%d, n=%d, start=(%d,%d), end=(%d,%d)", m, n, start_x, start_y, end_x, end_y) == 6;
}

// Function to generate paths for a given case
void generate_paths(FILE *f, int m, int n, int start_x, int start_y, int end_x, int end_y, int case_num) {
    int t = (end_x - start_x) + (end_y - start_y); // Total steps
    int k = (end_x - start_x); // Number of 's' steps

    unsigned long long count = combination(t, k);

    // Write case information
    fprintf(f, "Case %d: m=%d, n=%d, start=(%d,%d), end=(%d,%d)\n", case_num, m, n, start_x, start_y, end_x, end_y);
    fprintf(f, "Total paths: %llu\n", count);

    if (k == 0) {
        // All 'e's case
        char *path = (char *)malloc(t + 1);
        if (!path) {
            perror("malloc failed");
            return;
        }
        memset(path, 'e', t);
        path[t] = '\0';
        fprintf(f, "PATH{%s}\n", path);
        free(path);
    } else if (k == t) {
        // All 's's case
        char *path = (char *)malloc(t + 1);
        if (!path) {
            perror("malloc failed");
            return;
        }
        memset(path, 's', t);
        path[t] = '\0';
        fprintf(f, "PATH{%s}\n", path);
        free(path);
    } else {
        // Generate combinations
        int *c = (int *)malloc(k * sizeof(int));
        if (!c) {
            perror("malloc failed");
            return;
        }
        for (int i = 0; i < k; i++) {
            c[i] = i;
        }

        char *buffer = (char *)malloc(t + 1);
        if (!buffer) {
            perror("malloc failed");
            free(c);
            return;
        }
        memset(buffer, 'e', t);
        buffer[t] = '\0';

        bool has_next = true;
        while (has_next) {
            // Set 's's in the buffer
            for (int i = 0; i < k; i++) {
                buffer[c[i]] = 's';
            }
            fprintf(f, "PATH{%s}\n", buffer);

            // Reset 's's back to 'e's
            for (int i = 0; i < k; i++) {
                buffer[c[i]] = 'e';
            }

            // Generate next combination
            int i;
            for (i = k - 1; i >= 0; i--) {
                if (c[i] < t - k + i) {
                    break;
                }
            }
            if (i == -1) {
                has_next = false;
            } else {
                c[i]++;
                for (int j = i + 1; j < k; j++) {
                    c[j] = c[j - 1] + 1;
                }
            }
        }

        free(buffer);
        free(c);
    }
    fprintf(f, "\n"); // Separate cases with a blank line
}

int main() {
    FILE *input_file = fopen("input.txt", "r");
    if (!input_file) {
        perror("Failed to open input.txt");
        return 1;
    }

    FILE *output_file = fopen("output.txt", "w");
    if (!output_file) {
        perror("Failed to open output.txt");
        fclose(input_file);
        return 1;
    }

    char line[256];
    int case_num = 1;

    // Read input file line by line
    while (fgets(line, sizeof(line), input_file)) {
        int m, n, start_x, start_y, end_x, end_y;

        // Parse input line
        if (!parse_input_line(line, &m, &n, &start_x, &start_y, &end_x, &end_y)) {
            printf("Invalid input format in case %d: %s", case_num, line);
            continue;
        }

        // Validate input
        if (m <= 0 || m > 994 || n <= 0 || n > 750 ||
            start_x < 0 || start_x >= m || start_y < 0 || start_y >= n ||
            end_x < 0 || end_x >= m || end_y < 0 || end_y >= n) {
            printf("Invalid input values in case %d: %s", case_num, line);
            continue;
        }

        // Generate paths for the current case
        generate_paths(output_file, m, n, start_x, start_y, end_x, end_y, case_num);
        case_num++;
    }

    fclose(input_file);
    fclose(output_file);
    printf("Results have been written to output.txt.\n");
    return 0;
}