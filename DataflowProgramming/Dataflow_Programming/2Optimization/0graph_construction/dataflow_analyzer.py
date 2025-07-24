# usage: python3 dataflow_analyzer.py test.c


import re
import sys
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class OMPTaskParser:
    def __init__(self, filename=None, code_string=None):
        self.filename = filename
        self.input_code = code_string
        self.code = ""
        self.tasks = []
        self.variables = set()
        self.dependencies = []
        self.graph = nx.DiGraph()
        
    def read_file(self):
        """Read the C file content"""
        if self.input_code:
            self.code = self.input_code
            return True
            
        try:
            with open(self.filename, 'r') as file:
                self.code = file.read()
            return True
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    
    def extract_tasks(self):
        """Extract OpenMP task directives and their dependencies"""
        # Look for the pattern of pragma omp task with depend clauses
        lines = self.code.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this line is a task pragma
            if line.startswith('#pragma omp task') and 'depend' in line:
                task_id = len(self.tasks) + 1
                pragma_line = line
                
                # If the depend clauses continue to next line, combine them
                while i + 1 < len(lines) and not lines[i].strip().endswith('{') and '{' not in lines[i].strip():
                    i += 1
                    pragma_line += ' ' + lines[i].strip()
                
                # Extract the dependency information
                depend_pattern = r'depend\((\w+):\s*([^)]+)\)'
                depend_matches = re.finditer(depend_pattern, pragma_line)
                
                task_deps = []
                for dep_match in depend_matches:
                    dep_type = dep_match.group(1)  # 'in' or 'out'
                    var_names = dep_match.group(2).strip()
                    
                    # Handle multiple variables in one clause (comma-separated)
                    for var in var_names.split(','):
                        var = var.strip()
                        self.variables.add(var)
                        task_deps.append((dep_type, var))
                
                # Find the task body by tracking opening and closing braces
                body_start = i
                while body_start < len(lines) and '{' not in lines[body_start]:
                    body_start += 1
                
                if body_start >= len(lines):
                    i += 1
                    continue
                
                open_braces = 0
                body_end = body_start
                
                # Count the opening braces in the first line
                open_braces = lines[body_start].count('{')
                
                # Find where the task body ends
                while body_end < len(lines) and open_braces > 0:
                    body_end += 1
                    if body_end >= len(lines):
                        break
                    open_braces += lines[body_end].count('{')
                    open_braces -= lines[body_end].count('}')
                
                # Extract the task body
                body_lines = lines[body_start:body_end+1]
                task_body = '\n'.join(body_lines)
                
                # Try to extract a meaningful task name from operations in the body
                operation_pattern = r'=\s*([^;]+);'
                operation_match = re.search(operation_pattern, task_body)
                
                task_name = f"Task_{task_id}"
                if operation_match:
                    operation = operation_match.group(1).strip()
                    task_name = f"Task_{task_id} ({operation})"
                
                self.tasks.append({
                    'id': task_id,
                    'name': task_name,
                    'dependencies': task_deps,
                    'body': task_body
                })
                
                i = body_end
            i += 1
    
    def build_dependency_graph(self):
        """Build a directed graph of task dependencies"""
        # Track which task produces each variable
        producers = {}
        
        # First pass: register all tasks and output variables
        for task in self.tasks:
            task_id = task['id']
            task_name = task['name']
            
            # Add the task to the graph
            self.graph.add_node(task_id, name=task_name, label=task_name)
            
            # Register this task as the producer of its output variables
            for dep_type, var in task['dependencies']:
                if dep_type == 'out':
                    producers[var] = task_id
        
        # Second pass: add edges for dependencies
        for task in self.tasks:
            task_id = task['id']
            
            for dep_type, var in task['dependencies']:
                if dep_type == 'in' and var in producers:
                    producer_id = producers[var]
                    if producer_id != task_id:  # Avoid self-loops
                        self.graph.add_edge(producer_id, task_id, variable=var)
                        self.dependencies.append((producer_id, task_id, var))
    
    def visualize_graph(self, output_filename=None):
        """Create a visualization of the dependency graph"""
        plt.figure(figsize=(10, 8))
        
        # If graph is empty, display a message
        if len(self.graph.nodes()) == 0:
            plt.text(0.5, 0.5, "No tasks with dependencies found", 
                     horizontalalignment='center', fontsize=14)
            plt.axis('off')
            
            if output_filename:
                plt.savefig(output_filename)
                print(f"Graph saved as {output_filename}")
            else:
                plt.show()
            return
        
        # Create node positions - arrange tasks in a pipeline-like layout
        pos = {}
        nodes = sorted(self.graph.nodes())
        for i, node in enumerate(nodes):
            pos[node] = (i, 0)  # Horizontal layout
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color="lightblue", alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, width=2, arrowsize=20, alpha=0.7)
        
        # Add labels to nodes
        node_labels = {node: data['name'] for node, data in self.graph.nodes(data=True)}
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10)
        
        # Add labels to edges (dependency variables)
        edge_labels = {(u, v): data['variable'] for u, v, data in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("OpenMP Task Dataflow Graph")
        plt.axis('off')
        
        if output_filename:
            plt.savefig(output_filename)
            print(f"Graph saved as {output_filename}")
        else:
            plt.show()
    
    def print_task_details(self):
        """Print detailed information about the tasks and dependencies"""
        print("OpenMP Task Analysis Report:")
        print("===========================")
        
        print("\nTask Details:")
        for task in self.tasks:
            print(f"\nTask {task['id']}: {task['name']}")
            print("  Dependencies:")
            for dep_type, var in task['dependencies']:
                print(f"    {dep_type}: {var}")
            print("  Implementation:")
            # Format the body for better readability
            body_lines = task['body'].split('\n')
            formatted_body = '\n    '.join(body_lines)
            print(f"    {formatted_body}")
        
        print("\nDependency Relationships:")
        if not self.dependencies:
            print("  No dependencies found between tasks")
        else:
            for src, dst, var in self.dependencies:
                src_name = [t['name'] for t in self.tasks if t['id'] == src][0]
                dst_name = [t['name'] for t in self.tasks if t['id'] == dst][0]
                print(f"  {src_name} → {dst_name} via {var}")
    
    def analyze(self):
        """Perform the full analysis process"""
        if not self.read_file():
            return False
        
        self.extract_tasks()
        self.build_dependency_graph()
        return True

def main():
    if len(sys.argv) < 2:
        # For testing, use the embedded code
        test_code = """
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void process_data_pipeline(int input_data[], int output_data[], int n) {
    int *intermediate1 = (int*)malloc(n * sizeof(int));
    int *intermediate2 = (int*)malloc(n * sizeof(int));

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Process data elements one by one through the pipeline
            for (int i = 0; i < n; i++) {
                // Node 1: Square operation
                #pragma omp task depend(out: intermediate1[i])
                {
                    intermediate1[i] = input_data[i] * input_data[i];
                    printf("Thread %d: Node 1 squared data[%d] = %d to get %d\\n", 
                           omp_get_thread_num(), i, input_data[i], intermediate1[i]);
                }

                // Node 2: Double operation
                #pragma omp task depend(in: intermediate1[i]) depend(out: intermediate2[i])
                {
                    intermediate2[i] = intermediate1[i] * 2;
                    printf("Thread %d: Node 2 doubled data[%d] = %d to get %d\\n", 
                           omp_get_thread_num(), i, intermediate1[i], intermediate2[i]);
                }

                // Node 3: Add operation (adding the original value to the doubled-square)
                #pragma omp task depend(in: intermediate2[i]) depend(out: output_data[i])
                {
                    output_data[i] = intermediate2[i] + input_data[i];
                    printf("Thread %d: Node 3 added data[%d] = %d to %d to get %d\\n", 
                           omp_get_thread_num(), i, input_data[i], intermediate2[i], output_data[i]);
                }
            }
        }
    }

    free(intermediate1);
    free(intermediate2);
}

int main() {
    int n = 5;
    int input_data[5] = {1, 2, 3, 4, 5};
    int output_data[5];

    printf("Starting pipeline processing with %d data elements\\n", n);

    process_data_pipeline(input_data, output_data, n);

    printf("\\nFinal results:\\n");
    for (int i = 0; i < n; i++) {
        // output = input + 2*(input^2)
        printf("data[%d] = %d → result = %d\\n", i, input_data[i], output_data[i]);
    }

    return 0;
}
        """
        parser = OMPTaskParser(code_string=test_code)
        print("Using embedded test code")
    else:
        filename = sys.argv[1]
        parser = OMPTaskParser(filename=filename)
        print(f"Analyzing file: {filename}")
    
    if parser.analyze():
        parser.print_task_details()
        parser.visualize_graph("omp_dataflow_graph.png")
    else:
        print(f"Failed to analyze OpenMP tasks")

if __name__ == "__main__":
    main()