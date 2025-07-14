# Data Structures & Algorithms Visualizer

A comprehensive Python application that provides interactive visualizations for various data structures and algorithms. This tool is designed to help students, educators, and developers understand how different data structures work and how algorithms manipulate them in real-time.

## Features

### Supported Data Structures
- **Arrays**: Dynamic arrays with various sorting and searching algorithms
- **Stacks**: LIFO (Last In, First Out) data structure with push, pop, and peek operations
- **Queues**: FIFO (First In, First Out) data structure with enqueue and dequeue operations
- **Binary Trees**: Binary search trees with insertion, deletion, and traversal algorithms
- **Linked Lists**: Singly linked lists with insertion, deletion, and search operations
- **Graphs**: Undirected weighted graphs with pathfinding and traversal algorithms

### Supported Algorithms

#### Array Algorithms
- **Sorting Algorithms**:
  - Bubble Sort
  - Selection Sort
  - Insertion Sort
  - Merge Sort
  - Quick Sort
  - Heap Sort
- **Search Algorithms**:
  - Linear Search
  - Binary Search

#### Stack Operations
- Push (add element)
- Pop (remove top element)
- Peek (view top element)
- Display current stack

#### Queue Operations
- Enqueue (add element)
- Dequeue (remove front element)
- Front (view front element)
- Display current queue

#### Binary Tree Operations
- Insert node
- Delete node
- Search for value
- **Traversal Methods**:
  - Inorder traversal
  - Preorder traversal
  - Postorder traversal
  - Breadth-First Search (BFS)
  - Depth-First Search (DFS)

#### Linked List Operations
- Insert node
- Delete node
- Search for value
- Display entire list

#### Graph Algorithms
- **Dijkstra's Algorithm**: Find shortest path between two nodes
- **Breadth-First Search (BFS)**: Graph traversal
- **Depth-First Search (DFS)**: Graph traversal

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Dependencies
```bash
pip install matplotlib numpy
```

Note: `tkinter` usually comes pre-installed with Python. If you encounter issues, you may need to install it separately depending on your system.

### Installation Steps
1. Clone or download the repository
2. Install the required dependencies
3. Run the application:
   ```bash
   python main.py
   ```

## Usage

### Getting Started
1. Launch the application
2. Select a data structure from the dropdown menu
3. Choose an algorithm from the algorithm dropdown
4. Adjust the speed slider to control animation speed
5. Use the control buttons to interact with the visualizer

### Control Buttons
- **Generate Data**: Creates random data for the selected data structure
- **Add Element**: Manually add an element to the current data structure
- **Start**: Begin executing the selected algorithm
- **Reset**: Clear the current data structure and stop any running algorithm

### Speed Control
Use the speed slider to adjust the animation speed:
- Left (0.1): Fastest execution
- Right (2.0): Slowest execution

### Interactive Features
- **Real-time Visualization**: Watch algorithms execute step-by-step
- **Color Coding**: Different colors highlight current operations and comparisons
- **Status Updates**: Bottom status bar shows current operation and results
- **Dynamic Input**: Add custom values to data structures during runtime

## Examples

### Array Sorting
1. Select "Array" from the data structure dropdown
2. Choose a sorting algorithm (e.g., "Bubble Sort")
3. Click "Generate Data" to create a random array
4. Click "Start" to watch the sorting process
5. Red bars indicate elements being compared/swapped

### Graph Pathfinding
1. Select "Graph" from the data structure dropdown
2. Choose "Dijkstra" algorithm
3. Click "Generate Data" to create a random graph
4. Click "Start" and enter start and goal nodes
5. Watch as the algorithm finds the shortest path

### Tree Traversal
1. Select "Binary Tree" from the data structure dropdown
2. Add elements using "Add Element" button
3. Choose a traversal method (e.g., "Inorder")
4. Click "Start" to see the traversal order
5. Orange highlights show the current node being processed

## Technical Details

### Architecture
- **GUI Framework**: tkinter for the user interface
- **Visualization**: matplotlib for creating interactive charts and graphs
- **Threading**: Multi-threaded execution to prevent UI freezing during algorithm execution
- **Data Structures**: Custom implementations of TreeNode and ListNode classes

### Key Classes
- `DataStructureVisualizer`: Main application class
- `TreeNode`: Binary tree node implementation
- `ListNode`: Linked list node implementation

### Performance Considerations
- Algorithms run in separate threads to maintain UI responsiveness
- Visualization updates are throttled based on the speed setting
- Memory-efficient implementations suitable for educational purposes

## Educational Value

This visualizer is particularly useful for:
- **Computer Science Students**: Understanding algorithm complexity and behavior
- **Educators**: Teaching data structures and algorithms concepts
- **Interview Preparation**: Visualizing common coding interview problems
- **Algorithm Analysis**: Comparing different algorithm performance characteristics

## Customization

The application can be extended by:
- Adding new data structures in the `setup_ui()` method
- Implementing additional algorithms in their respective operation methods
- Modifying visualization styles in the `visualize_*()` methods
- Adjusting timing and animation parameters

## Troubleshooting

### Common Issues
1. **Import Error**: Ensure all required packages are installed
2. **Performance Issues**: Reduce array size or increase speed setting
3. **UI Freezing**: Algorithms run in separate threads; wait for completion
4. **Visualization Not Updating**: Check that matplotlib backend is properly configured

### System Requirements
- **RAM**: Minimum 4GB recommended for smooth operation
- **Display**: Minimum 1024x768 resolution
- **Python Version**: 3.7 or higher

## License

This project is open-source and available for educational use. Feel free to modify and distribute according to your needs.

## Contributing

Contributions are welcome! Areas for improvement include:
- Additional data structures (AVL trees, hash tables, etc.)
- More algorithm implementations
- Enhanced visualization features
- Performance optimizations
- Mobile/web versions

## Support

For issues, questions, or suggestions:
- Check the troubleshooting section
- Review the code comments for implementation details
- Create issues for bugs or feature requests